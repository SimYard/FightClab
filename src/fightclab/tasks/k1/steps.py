"""K-1 footwork and jabs generated from the K-1 coach's measures, on the stance and guard of the Varga clips.

    uv run python -m fightclab.tasks.k1.steps OUT.csv forward:4 stop:2 back:4 left:4 right:4 stop:2 jab:1 step_jab:1.5
    uv run python -m fightclab.tasks.k1.steps OUT.csv stop:1 jab:1@1.60,0.06 step_jab:1.5@1.56,0.05,0.22 (aimed)
    uv run python -m fightclab.tasks.k1.steps OUT.csv random:600      (random commands for ~600 s, to train on)

Each move command moves the stance in step-drags: the foot nearer the direction (the lead foot going forward or left,
the rear foot going back or right) steps STEP_LENGTH with its sole lifted STEP_LIFT, then the other foot follows as
far. The pelvis follows the feet's midpoint at the stance's height and lean; arms, torso and head hold the guard; the
legs are solved by inverse kinematics. Strike commands (``strikes.STRIKES``: jab, jab_body, step_jab) throw a jab
from that guard, see ``strike``; ``catch`` brings the rear glove in front of the face (``catch.py``). Rows are at
30 fps (root pos, quat xyzw, joints) with OUT.labels.txt naming
each row's command; ``motion_npz t800 OUT.csv OUT.npz`` makes it a motion for the tracking task.

The video reference brought the coach's demo habits along: 46 cm lunges, a rocking pelvis, dragged and hanging feet,
and seams where pieces were joined. The stance is still the clips' median, corrected where the K-1 coach measured it
off (STANCE_WIDTH, LEAD_TOE_OUT, REAR_HEEL).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from fightclab.motion import ROBOTS, leg_chains, lowest, model, reach, set_pose, soles
from fightclab.tasks.k1 import catch
from fightclab.tasks.k1.footwork import OPPONENT_YAW, STANCE
from fightclab.tasks.k1.footwork_motion import FPS, GUARD_POSE
from fightclab.tasks.k1.strikes import (
    FOREARM_WEIGHT,
    HIP_TURN,
    LEAD_ARM,
    OPPONENT,
    REAR_HEEL_UP,
    REAR_TURN,
    SHIFT_LEAD,
    SHIFTS,
    STEP_LATE,
    STRIKES,
    TARGETS,
    TORSO,
    TURN_BACK,
    TURN_LEAD,
    Target,
    aim,
    arm_chain,
    body_ramp,
    contact,
    guard_weights,
    profile,
)

ROBOT = ROBOTS["t800"]
HEIGHT = 1.75  # m, T800 standing
STEP_LENGTH = 0.18  # m: the K-1 coach's 10-25 cm; the tracked robot made 17 cm steps on its own
STEP_LIFT = 0.04  # m of sole clearance at mid-step (the coach: 3-6 cm)
STEP_JAB_LIFT = 0.04  # m, as in the footwork: at the coach's 2.5 cm skim the 1.4 m/s foot grazed the floor, feet_slip
# made the step short, and the pelvis stayed back (the RL review)
SWING, GAP, REST = 0.22, 0.06, 0.12  # s: the first foot's step, the pause, the second's step takes SWING, then rest
# s: the step jab's lead foot, out earlier to land at the same time. In SWING the pelvis following it needed a floor
# friction of about 1 (the RL review: the trained robot never asked for more than the lowest drawn, 0.3).
STEP_SWING = 0.3
STANCE_WIDTH = 0.28 * HEIGHT  # m between the feet (the coach: 0.24-0.32 H; the clips crept to 0.36 H)
LEAD_TOE_OUT = np.radians(10.0)  # the lead foot points this far out from the line to the opponent (clips: 14-31 in)
REAR_HEEL = np.radians(10.0)  # the rear foot's heel up (toe down), on its ball
# frames: the pelvis' path smoothed over this. Over 9 (0.3 s) it followed each step's 9 cm in 0.22 s, which needs a
# floor friction of 0.6.
PELVIS_WINDOW = 15
# m the pelvis sits below the clips' stance: their rear leg was nearly straight, and with the pelvis following the
# feet it could not reach its foot during the lead step.
PELVIS_DROP = 0.04
# Unit moves in the opponent's frame (forward, left) and which foot steps first (0 left/lead, 1 right/rear).
COMMANDS = {"forward": ((1, 0), 0), "back": ((-1, 0), 1), "left": ((0, 1), 0), "right": ((0, -1), 1), "stop": None}
CATCH = "catch"  # the rear-hand catch of a head jab (``catch.py``)
# The skills of random_commands and their shares of its skill slots: every skill learned so far, always (a fine-tune
# on some of them forgot the rest), a new one given about 35% and every old one 0.5% or more (docs/duel.md, "Growing
# the skill library"). The approach jab came first (Varga steps in with it at 5:31.8 of "Combining Footwork &
# Punches"); trained alone, its step leaked into the standing jab (a shuffle forward). These shares and SKILL_SLOT give
# the counts mkc2 was trained on: a strike after half the moves, a catch after 35%.
SKILL_SHARES = {"step_jab": 0.2, "jab": 0.2, "jab_body": 0.2, CATCH: 0.4}
SKILL_SLOT = 0.85  # of the moves followed by a skill


class Aim(NamedTuple):
    """Where a strike goes: its target's height and side (``strikes.Target``) and, stepping in, the step's length."""

    height: float
    side: float
    step: float = STEP_LENGTH


# Training strikes go anywhere on the target, so the tracker hits where the duel policy aims, not three fixed points
# (m: target height, side). The head: chin to forehead of a T800 in guard, whose face's centre is at 1.56-1.64 m and
# 0.045-0.075 m toward the centre line (the K-1 coach; the old fixed 1.50 m ran under the head's collision sphere).
AIMS = {"head": ((1.50, 1.66), (0.0, 0.09)), "body": ((1.15, 1.30), (0.0, 0.06))}
STEP_RANGE = (0.10, 0.25)  # m, the step jab's step (the K-1 coach's 10-25 cm): reach beyond standing range
Command = tuple[str, float] | tuple[str, float, Aim]


@dataclass(frozen=True)
class Stance:
    """The stance in the pelvis' heading frame (x along the heading, y left)."""

    feet: NDArray[np.float64]
    """(2, 2): the left (lead) and right (rear) foot relative to their midpoint."""
    pelvis: NDArray[np.float64]
    """(2,): the pelvis relative to the feet's midpoint."""
    height: float
    pitch: float
    """rad, the pelvis' forward lean."""
    rear_yaw: float
    """rad, the rear foot's turn from the heading."""


# The footwork clips' still frames (both feet planted, pelvis slower than 5 cm/s) at their median, the feet spread to
# STANCE_WIDTH. Measured once from the cleaned clips (``footwork_motion``), which are not in the repo.
CLIP_STANCE = Stance(
    feet=np.array(((0.09678666041367329, 0.22507186044898703), (-0.09678666041367329, -0.22507186044898703))),
    pelvis=np.array((0.03891281486309335, -0.012186716368568815)),
    height=0.994756,
    pitch=0.13029068303122204,
    rear_yaw=-0.11917156659851422,
)


def plan(
    commands: list[Command], st: Stance
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str], list[tuple[int, str, Aim | None]]]:
    """The feet on the floor (2, N, 2), the soles' lift (2, N), each frame's command, each strike's first frame and aim.

    The world here has the pelvis heading along +x. A strike without an aim goes to its ``strikes.TARGETS`` target.
    """
    yaw0 = OPPONENT_YAW  # the opponent's direction from the pelvis' heading (+x)
    to_world = np.array(((np.cos(yaw0), -np.sin(yaw0)), (np.sin(yaw0), np.cos(yaw0))))
    feet = st.feet.copy()
    pos, lift, labels = [], [], []
    strikes: list[tuple[int, str, Aim | None]] = []

    def hold(seconds: float, name: str) -> None:
        for _ in range(round(seconds * FPS)):
            pos.append(feet.copy())
            lift.append(np.zeros(2))
            labels.append(name)

    def swing(f: int, move: NDArray[np.float64], name: str, height: float = STEP_LIFT, seconds: float = SWING) -> None:
        n = round(seconds * FPS)
        start = feet[f].copy()
        for k in range(1, n + 1):
            u = k / n
            feet[f] = start + move * (10 * u**3 - 15 * u**4 + 6 * u**5)  # minimum jerk
            pos.append(feet.copy())
            up = np.zeros(2)
            up[f] = height * np.sin(np.pi * u)
            lift.append(up)
            labels.append(name)

    cycle = 2 * SWING + GAP + REST
    extension = profile()[0]
    peak, n = int(np.argmax(extension)), round(STEP_SWING * FPS)
    for name, seconds, *rest in commands:
        if name == CATCH:  # the feet hold; the rear glove catches (``catch.py``)
            strikes.append((len(pos), name, None))
            hold(max(seconds, catch.FRAMES / FPS), name)
            continue
        if name in STRIKES:  # the feet hold, or the lead foot steps in to land with the fist and the rear follows
            aim = rest[0] if rest else None
            strikes.append((len(pos), name, aim))
            if STRIKES[name][1]:  # the coach: it lands with the fist, never before; the rear follows within 0.3 s
                move = to_world @ np.array((1.0, 0.0)) * (aim.step if aim else STEP_LENGTH)
                hold((peak + 1 + STEP_LATE - n) / FPS, name)
                swing(0, move, name, STEP_JAB_LIFT, STEP_SWING)
                hold(GAP, name)
                swing(1, move, name)
            hold(max(seconds, len(extension) / FPS) - (len(pos) - strikes[-1][0]) / FPS, name)
            continue
        spec = COMMANDS[name]
        if spec is None:
            hold(seconds, name)
            continue
        (fwd, left), first = spec
        move = to_world @ np.array((fwd, left), dtype=float) * STEP_LENGTH
        cycles = max(int(seconds // cycle), 1)
        for _ in range(cycles):
            swing(first, move, name)
            hold(GAP, name)
            swing(1 - first, move, name)
            hold(REST, name)
        hold(seconds - cycles * cycle, name)
    return np.array(pos).transpose(1, 0, 2), np.array(lift).T, labels, strikes


def aimed(name: str, aim: Aim | None) -> Target:
    """The strike's target, moved to ``aim``'s height and side."""
    target = TARGETS[STRIKES[name][0]]
    return target if aim is None else target._replace(height=aim.height, side=aim.side)


def build(commands: list[Command], st: Stance = CLIP_STANCE) -> tuple[NDArray[np.float64], list[str]]:
    """The motion's rows (30 fps: root pos, quat xyzw, joints) and each row's command."""
    feet, lift, labels, strikes = plan(commands, st)
    m, d, adr = model(ROBOT)
    legs = leg_chains(m, ROBOT)
    sole = soles(m, ROBOT)
    # The lead toe out from the opponent's line, the rear foot as in the clips with its heel up.
    turns = [Rotation.from_euler("Z", OPPONENT_YAW + LEAD_TOE_OUT), Rotation.from_euler("ZY", (st.rear_yaw, REAR_HEEL))]
    rows = np.zeros((feet.shape[1], 7 + len(adr)))
    rows[:, :2] = savgol_filter(feet.mean(0) + st.pelvis, PELVIS_WINDOW, 2, axis=0)
    rows[:, 2] = st.height - PELVIS_DROP
    rows[:, 3:7] = Rotation.from_euler("ZYX", (0.0, st.pitch, 0.0)).as_quat()
    # Each strike turns the hips, leans and shifts the pelvis toward the opponent (down too for a body jab), and turns
    # the rear foot out with its heel up; the legs follow below.
    extension = profile()[0]
    toward = np.array((np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW)))
    rear = np.zeros(len(rows))  # how far into the strike the rear foot is, 0 to 1
    for start, name, point in strikes:
        if name not in STRIKES:  # a catch moves nothing but the rear arm
            continue
        target = aimed(name, point)
        late = STEP_LATE if STRIKES[name][1] else 0
        ramp = body_ramp(extension, target.lead, target.back, late=late)
        span = slice(start, start + len(ramp))
        # The pelvis toward the target (``strikes.SHIFTS``), tops out ``coast`` frames after contact. Stepping in, the
        # weight stays forward until the rear foot is down (the coach: it unwound and backed up). It starts with the
        # fist even in a body jab: pushed forward while falling into the drop, the zero-moment point reached the rear
        # ball's edge.
        follow = round(GAP * FPS) + round(SWING * FPS) if late else 0
        size, coast, back = SHIFTS[name]
        if coast is None:
            coast = int(np.argmax(extension)) - int(np.argmax(extension > 0.05)) + SHIFT_LEAD  # noqa: PLR2004
        shift = body_ramp(extension, SHIFT_LEAD, back, max(follow - coast, 0), late, top=coast)
        assert shift[-1] == ramp[-1] == 0, "a body ramp outlasts the strike: raise strikes.PAD"
        rows[span, :2] += np.outer(shift, toward) * size
        rows[span, 2] -= ramp * target.drop
        rows[span, 3:7] = Rotation.from_euler(
            "ZYX", np.c_[HIP_TURN * ramp, st.pitch + target.lean * ramp, 0 * ramp]
        ).as_quat()
        rear[span] = ramp
    # The legs start bent, from the clips' median pose: a straight knee is singular and the solver stays stuck there.
    joints = np.array([GUARD_POSE.get(n, STANCE[n]) for n in ROBOT.joints])
    # Each foot body's height with its sole on the floor, turned as it will be: the feet are put down on the first
    # frame, turned, and measured.
    set_pose(m, d, adr, rows[0], joints)
    floor = [d.xpos[leg[0]][2] - lowest(m, d, sole[f]) - PELVIS_DROP for f, leg in enumerate(legs)]
    for _ in range(5):
        for f, leg in enumerate(legs):
            reach(m, d, leg, np.r_[feet[f, 0], floor[f]], turns[f])
    floor = [d.xpos[leg[0]][2] - lowest(m, d, sole[f]) for f, leg in enumerate(legs)]
    joints = d.qpos[adr].copy()

    def rear_turn(x: float) -> Rotation:  # the rear foot, ``x`` of the way into a strike
        return Rotation.from_euler("ZY", (st.rear_yaw + REAR_TURN * x, REAR_HEEL + REAR_HEEL_UP * x))

    # The rear foot's height with its toe on the floor, turned out and heel up as at a strike's contact.
    reach(m, d, legs[1], d.xpos[legs[1][0]].copy(), rear_turn(1.0))
    floor_up = d.xpos[legs[1][0]][2] - lowest(m, d, sole[1])
    for i in range(len(rows)):
        set_pose(m, d, adr, rows[i], joints)
        for _ in range(5 if i == 0 else 1):  # the first frame converges from the clips' pose
            reach(m, d, legs[0], np.r_[feet[0, i], floor[0] + lift[0, i]], turns[0])
            z = floor[1] + (floor_up - floor[1]) * rear[i]
            reach(m, d, legs[1], np.r_[feet[1, i], z + lift[1, i]], rear_turn(rear[i]))
        joints = d.qpos[adr].copy()
        rows[i, 7:] = joints
    for name, angle in GUARD_POSE.items():
        rows[:, 7 + ROBOT.joints.index(name)] = angle
    for start, name, point in strikes:
        if name in STRIKES:
            strike(m, d, adr, rows[start : start + len(extension)], aimed(name, point))
        else:
            catch.catch(m, d, adr, rows[start:])
    return rows, labels


def strike(
    m: mujoco.MjModel, d: mujoco.MjData, adr: NDArray[np.int_], rows: NDArray[np.float64], target: Target
) -> None:
    """A jab in ``rows`` (the strike's frames: the body already moving with it, the arms in guard).

    The waist turns on the clip's timing (``strikes.profile``); the lead fist goes straight from the guard to the
    contact pose (``strikes.contact``) on the opponent's centre line at ``target.height``, and back, the palm turning
    down from half extension; the arm by inverse kinematics. The fist's path is relative to the pelvis, which moves
    with the strike.
    """
    extension, turn = profile()
    arm = arm_chain(m)
    cols = 7 + np.array([ROBOT.joints.index(n) for n in LEAD_ARM])
    shoulder = m.body("LINK_SHOULDER_PITCH_L").id
    rows[:, 7 + ROBOT.joints.index(TORSO)] += turn[np.argmax(np.abs(turn))] * body_ramp(extension, TURN_LEAD, TURN_BACK)

    set_pose(m, d, adr, rows[0])
    guard, hold = d.xpos[arm.body] - rows[0, :3], rows[0, cols].copy()
    roll, twist, forearm = (LEAD_ARM.index(n) for n in ("J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L", "J17_ELBOW_YAW_L"))
    peak = int(np.argmax(extension))
    set_pose(m, d, adr, rows[peak])
    toward = np.array((np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW)))
    aim_at = np.r_[rows[peak, :2] + OPPONENT * toward + target.side * np.array((-toward[1], toward[0])), target.height]
    hit = contact(m, d, arm, (d.xpos[shoulder].copy(), aim_at), hold) - rows[peak, :3]
    struck = d.qpos[arm.qadr].copy()  # the arm at contact; the shoulder and forearm go there from half extension
    weight = guard_weights()
    weight[forearm] = FOREARM_WEIGHT
    joints = hold
    for i in range(len(rows)):
        set_pose(m, d, adr, rows[i])
        d.qpos[arm.qadr] = joints
        s = np.clip((extension[i] - 0.5) / 0.5, 0.0, 1.0)  # the fist turns over from half extension
        prior = hold.copy()
        blend = s * s * (3 - 2 * s)
        for j in (roll, twist, forearm):
            prior[j] = hold[j] + (struck[j] - hold[j]) * blend
        miss = aim(m, d, arm, rows[i, :3] + guard + extension[i] * (hit - guard), prior=prior, weight=weight)
        assert miss < 0.01, f"the jab's fist misses its path by {miss:.3f} m"  # noqa: PLR2004
        joints = d.qpos[arm.qadr].copy()
        rows[i, cols] = joints


def random_commands(seconds: float, seed: int = 0) -> list[Command]:
    """Moves of 1-4 s in random order with skills and stops between them; about ``seconds`` long.

    SKILL_SLOT of the moves are followed by a skill drawn by SKILL_SHARES, half by a stop. Each strike goes to a random
    ``AIMS`` point, stepping in by a random ``STEP_RANGE`` step.
    """
    unknown = set(SKILL_SHARES) - {*STRIKES, CATCH}
    if unknown:
        raise ValueError(f"no such skills: {sorted(unknown)}")
    skills, weights = list(SKILL_SHARES), np.array(list(SKILL_SHARES.values()))
    rng = np.random.default_rng(seed)
    out: list[Command] = [("stop", 1.0)]
    moves = [c for c in COMMANDS if c != "stop"]
    while sum(c[1] for c in out) < seconds:
        out.append((str(rng.choice(moves)), float(rng.uniform(1.0, 4.0))))
        if rng.random() < SKILL_SLOT:
            name = str(rng.choice(skills, p=weights / weights.sum()))
            if name == CATCH:
                out.append((CATCH, float(rng.uniform(0.6, 1.2))))
            else:
                (low, high), (left, right) = AIMS[STRIKES[name][0]]
                point = Aim(
                    float(rng.uniform(low, high)), float(rng.uniform(left, right)), float(rng.uniform(*STEP_RANGE))
                )
                out.append((name, float(rng.uniform(0.8, 1.5)), point))
        if rng.random() < 0.5:  # noqa: PLR2004
            out.append(("stop", float(rng.uniform(0.5, 2.0))))
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1])
    args = sys.argv[2:]
    if args[0].startswith("random:"):
        commands = random_commands(float(args[0].split(":")[1]))
    else:  # NAME:SECONDS, a strike optionally @HEIGHT,SIDE[,STEP]
        commands = []
        for a in args:
            head, _, at = a.partition("@")
            name, seconds = head.split(":")
            point = Aim(*(float(x) for x in at.split(","))) if at else None
            commands.append((name, float(seconds), point) if point else (name, float(seconds)))
    rows, labels = build(commands)
    np.savetxt(out, rows, delimiter=",", fmt="%.6f")
    out.with_suffix(".labels.txt").write_text("\n".join(labels))
    print(f"{len(rows) / FPS:.1f} s -> {out}")
