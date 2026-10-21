"""K-1 footwork generated from the K-1 coach's measures, on the stance and guard of the Varga clips.

    uv run python -m fightclab.tasks.k1.steps OUT.csv forward:4 stop:2 back:4 left:4 right:4 stop:2
    uv run python -m fightclab.tasks.k1.steps OUT.csv random:600      (random commands for ~600 s, to train on)

Each command moves the stance in step-drags: the foot nearer the direction (the lead foot going forward or left, the
rear foot going back or right) steps STEP_LENGTH with its sole lifted STEP_LIFT, then the other foot follows as far.
The pelvis follows the feet's midpoint at the stance's height and lean; arms, torso and head hold the guard; the
legs are solved by inverse kinematics. Rows are at 30 fps (root pos, quat xyzw, joints) with OUT.labels.txt naming
each row's command; ``motion_npz t800 OUT.csv OUT.npz`` makes it a motion for the tracking task.

The video reference brought the coach's demo habits along: 46 cm lunges, a rocking pelvis, dragged and hanging feet,
and seams where pieces were joined. The stance is still the clips' median, corrected where the K-1 coach measured it
off (STANCE_WIDTH, LEAD_TOE_OUT, REAR_HEEL).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from fightclab.tasks.helicopter_kick.video_motion import ROBOTS, _lowest, leg_chains, reach
from fightclab.tasks.k1.footwork import OPPONENT_YAW, STANCE
from fightclab.tasks.k1.footwork_motion import FPS, GUARD_POSE

ROBOT = ROBOTS["t800"]
HEIGHT = 1.75  # m, T800 standing
STEP_LENGTH = 0.18  # m: the K-1 coach's 10-25 cm; the tracked robot made 17 cm steps on its own
STEP_LIFT = 0.04  # m of sole clearance at mid-step (the coach: 3-6 cm)
SWING, GAP, REST = 0.22, 0.06, 0.12  # s: the first foot's step, the pause, the second's step takes SWING, then rest
STANCE_WIDTH = 0.28 * HEIGHT  # m between the feet (the coach: 0.24-0.32 H; the clips crept to 0.36 H)
LEAD_TOE_OUT = np.radians(10.0)  # the lead foot points this far out from the line to the opponent (clips: 14-31 in)
REAR_HEEL = np.radians(10.0)  # the rear foot's heel up (toe down), on its ball
PELVIS_WINDOW = 9  # frames: the pelvis' path smoothed over this
# m the pelvis sits below the clips' stance: their rear leg was nearly straight, and with the pelvis following the
# feet it could not reach its foot during the lead step.
PELVIS_DROP = 0.04
# Unit moves in the opponent's frame (forward, left) and which foot steps first (0 left/lead, 1 right/rear).
COMMANDS = {"forward": ((1, 0), 0), "back": ((-1, 0), 1), "left": ((0, 1), 0), "right": ((0, -1), 1), "stop": None}


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


def plan(commands: list[tuple[str, float]], st: Stance) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Per frame: the feet on the floor (2, N, 2), the soles' lift (2, N) and the command.

    The world here has the pelvis heading along +x.
    """
    yaw0 = OPPONENT_YAW  # the opponent's direction from the pelvis' heading (+x)
    to_world = np.array(((np.cos(yaw0), -np.sin(yaw0)), (np.sin(yaw0), np.cos(yaw0))))
    feet = st.feet.copy()
    pos, lift, labels = [], [], []

    def hold(seconds: float, name: str) -> None:
        for _ in range(round(seconds * FPS)):
            pos.append(feet.copy())
            lift.append(np.zeros(2))
            labels.append(name)

    def swing(f: int, move: NDArray[np.float64], name: str) -> None:
        start = feet[f].copy()
        n = round(SWING * FPS)
        for k in range(1, n + 1):
            u = k / n
            feet[f] = start + move * (10 * u**3 - 15 * u**4 + 6 * u**5)  # minimum jerk
            pos.append(feet.copy())
            up = np.zeros(2)
            up[f] = STEP_LIFT * np.sin(np.pi * u)
            lift.append(up)
            labels.append(name)

    cycle = 2 * SWING + GAP + REST
    for name, seconds in commands:
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
    return np.array(pos).transpose(1, 0, 2), np.array(lift).T, labels


def build(commands: list[tuple[str, float]], st: Stance = CLIP_STANCE) -> tuple[NDArray[np.float64], list[str]]:
    """The motion's rows (30 fps: root pos, quat xyzw, joints) and each row's command."""
    feet, lift, labels = plan(commands, st)
    m = ROBOT.spec().compile()
    d = mujoco.MjData(m)
    adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ROBOT.joints])
    legs = leg_chains(m, ROBOT)
    shapes = (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)
    soles = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == leg[0] and m.geom_type[g] in shapes] for leg in legs]
    # The lead toe out from the opponent's line, the rear foot as in the clips with its heel up.
    turns = [Rotation.from_euler("Z", OPPONENT_YAW + LEAD_TOE_OUT), Rotation.from_euler("ZY", (st.rear_yaw, REAR_HEEL))]
    rows = np.zeros((feet.shape[1], 7 + len(adr)))
    rows[:, :2] = savgol_filter(feet.mean(0) + st.pelvis, PELVIS_WINDOW, 2, axis=0)
    rows[:, 2] = st.height - PELVIS_DROP
    rows[:, 3:7] = Rotation.from_euler("ZYX", (0.0, st.pitch, 0.0)).as_quat()
    # The legs start bent, from the clips' median pose: a straight knee is singular and the solver stays stuck there.
    joints = np.array([GUARD_POSE.get(n, STANCE[n]) for n in ROBOT.joints])
    # Each foot body's height with its sole on the floor, turned as it will be: the feet are put down on the first
    # frame, turned, and measured.
    d.qpos[:3], d.qpos[3:7], d.qpos[adr] = rows[0, :3], rows[0, [6, 3, 4, 5]], joints
    mujoco.mj_kinematics(m, d)
    floor = [d.xpos[leg[0]][2] - min(_lowest(m, d, g) for g in soles[f]) - PELVIS_DROP for f, leg in enumerate(legs)]
    for _ in range(5):
        for f, leg in enumerate(legs):
            reach(m, d, leg, np.r_[feet[f, 0], floor[f]], turns[f])
    floor = [d.xpos[leg[0]][2] - min(_lowest(m, d, g) for g in soles[f]) for f, leg in enumerate(legs)]
    joints = d.qpos[adr].copy()
    for i in range(len(rows)):
        d.qpos[:3], d.qpos[3:7], d.qpos[adr] = rows[i, :3], rows[i, [6, 3, 4, 5]], joints
        for _ in range(5 if i == 0 else 1):  # the first frame converges from the clips' pose
            for f, leg in enumerate(legs):
                reach(m, d, leg, np.r_[feet[f, i], floor[f] + lift[f, i]], turns[f])
        joints = d.qpos[adr].copy()
        rows[i, 7:] = joints
    for name, angle in GUARD_POSE.items():
        rows[:, 7 + ROBOT.joints.index(name)] = angle
    return rows, labels


def random_commands(seconds: float, seed: int = 0) -> list[tuple[str, float]]:
    """Commands of 1-4 s in random order (a stop between moves half the time), about ``seconds`` long."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, float]] = [("stop", 1.0)]
    moves = [c for c in COMMANDS if c != "stop"]
    while sum(s for _, s in out) < seconds:
        out.append((str(rng.choice(moves)), float(rng.uniform(1.0, 4.0))))
        if rng.random() < 0.5:  # noqa: PLR2004
            out.append(("stop", float(rng.uniform(0.5, 2.0))))
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1])
    args = sys.argv[2:]
    if args[0].startswith("random:"):
        commands = random_commands(float(args[0].split(":")[1]))
    else:
        commands = [(a.split(":")[0], float(a.split(":")[1])) for a in args]
    rows, labels = build(commands)
    np.savetxt(out, rows, delimiter=",", fmt="%.6f")
    out.with_suffix(".labels.txt").write_text("\n".join(labels))
    print(f"{len(rows) / FPS:.1f} s -> {out}")
