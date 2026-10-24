"""K-1 strikes for the footwork generator (``steps.py``): the jab, to the K-1 coach's measures, on Varga's timing.

``jab_t800.csv`` holds one jab from Gabriel Varga's "Combining Footwork & Punches" (8:54.5, in profile), through
GVHMR, ``tools/gvhmr_to_csv.py`` and ``video_motion --ground`` (30 fps rows: root pos, quat xyzw, joints). Only its
timing and torso turn are used (``profile``: the lead fist's extension, 0.17 s out and 0.17 s back, and an 18 deg
waist turn). Its arm joints are not (GMR trades shoulder roll for yaw from frame to frame), nor its pelvis: Varga
jabs while stepping back.

The rest follows the K-1 coach's review of a first version that "hit with the forearm, a slap": the elbow bent 44 deg
at contact, the upper arm below horizontal, the fist on its side, the hips still and the body sliding as one block.
Now the fist travels a straight line from the guard to a contact pose with the elbow ELBOW_CONTACT bent, on the line
from the shoulder to the target, palm down (``contact``); on the way the arm is solved by inverse kinematics with the
palm turning down from half extension, the shoulder's yaw held near the guard so the forearm does the twist (``aim``).
The hips turn (HIP_TURN), the pelvis leans and shifts toward the target and, for a body jab, drops on bent knees, all
on minimum-jerk ramps (``body_ramp``) that peak at contact; the rear heel rises and turns out with the hips.
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from fightclab.motion import ROBOTS, model, set_pose

H = 1.75  # m, T800
CLIP = Path(__file__).with_name("jab_t800.csv")
LEAD_ARM = ("J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L")
REAR_ARM = ("J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R")
TORSO = "J12_TORSO_YAW"
SPEED = (
    0.75  # times the clip's timing: 0.23 s out (the coach: 0.25 or less), the elbow opening at ~19 rad/s (its M107-15
)
# motor gives full torque to 14 and none at 25.6; at the clip's own speed it needed 25)
ELBOW_CONTACT = np.radians(-15.0)  # the elbow at contact (the coach: bent 10-20 deg)
ELBOW_MAX = np.radians(-5.0)  # never locked or hyperextended (the model allows +15)
PALM_WEIGHT = 0.3  # the palm-down residual against metres of fist position
# How hard the IK holds the shoulder near the guard: its yaw, or the IK twists the shoulder instead of the forearm
# and swings the elbow; its roll, or the elbow flies out sideways (it went to 90 deg, a hook).
TWIST_WEIGHT, ROLL_WEIGHT = 0.3, 0.2
# On the way the forearm's twist follows a schedule (``strike``) instead of a palm residual: with both the shoulder's
# yaw and the forearm free to turn the palm, the IK traded one for the other into their limits.
FOREARM_WEIGHT = 2.0
MAX_STEP = 0.1  # rad per IK iteration: with the fist weighted LINE_WEIGHT, unlimited steps jumped into joint limits
HIP_TURN = np.radians(-13.0)  # pelvis yaw at contact, the lead hip forward (the coach: -12 to -15)
SHIFT_LEAD = 1  # frames the pelvis' shift toward the target starts before the fist, in every jab
REAR_TURN, REAR_HEEL_UP = np.radians(-7.0), np.radians(10.0)  # the rear foot turns out with the hips, its heel rises
OPPONENT = 1.0  # m from the pelvis to the opponent's centre line, along which the fist aims
# The waist turn: the clip's size, on the body's timing. On the clip's own it started 0.3 s before the fist and gave
# the jab away (the coach), and was still 4.6 deg off when the fist was home.
TURN_LEAD, TURN_BACK = 1, 8
PAD = 8  # frames in guard after the clip, so that every body ramp is back to 0 within a strike
# The step jab lands the fist this many frames before the lead foot, with the body still moving (the coach: it had
# stopped when the fist landed); the K-1 order is fist, then foot.
STEP_LATE = 2
LINE_WEIGHT = 10.0  # the fist's place against the joints' holds: at 1 the holds won, the jab missed 11 cm right


class Target(NamedTuple):
    """Where a jab goes and what the body does for it."""

    height: float
    """m above the floor of the aim point on a T800 opponent in the same stance."""
    drop: float
    """m the pelvis gets lower, on bent knees (a body jab does not punch down)."""
    lean: float
    """rad of extra forward lean, from the hips (T800 has no waist pitch)."""
    lead: int
    """frames the body's ramp starts before the fist leaves the guard (the coach: 1 at most, or it telegraphs; the
    body jab's level change leads by 3 at most and doubles as a feint)."""
    back: int
    """frames the body takes to return after contact (kept under 0.7 g)."""
    side: float = 0.0
    """m the aim point sits left of the centre line (toward the lead side)."""


TARGETS = {
    "head": Target(1.50, 0.0, np.radians(4.5), lead=1, back=7),  # nose and chin level
    # Solar plexus, on the sternum and right chest: on the centre line the opponent's lead elbow cleared it by 1-3 cm.
    # The drop at the coach's shallowest (0.05-0.07 H): at 0.06 H, falling into it while shifting forward needed a
    # floor friction of 0.74.
    "body": Target(1.25, 0.05 * H, np.radians(5.5), lead=3, back=9, side=0.03),
}
# Commands: the target and whether the lead foot steps in with the jab (landing with the fist).
STRIKES = {"jab": ("head", False), "jab_body": ("body", False), "step_jab": ("head", True)}


class Shift(NamedTuple):
    """The pelvis toward the target in a strike, on top of the feet."""

    size: float
    """m in all."""
    top: int | None
    """frames from contact to the top; None: as many as from the start to contact, fastest at contact."""
    back: int
    """frames back to the guard."""


# Moving at contact, the rest after it, back by contact + 0.4 s. Stopped at contact (0.022 H, back in 0.23-0.3 s) the
# jab braked the zero-moment point 5 cm past the lead toe and came back with it 12 cm behind the rear ball: no robot
# does that on its feet, and the trained one kept its pelvis still. Carried on to 0.035 H and back over 0.5 s, the
# body still went in after the fist was home, with 0.67 of the weight on a lead leg that then can't check a low kick,
# and was forward when a counter lands, 0.4 s after contact. These are the K-1 coach's: 0.015-0.025 H at contact,
# 8-17 kg m/s of body momentum standing (25-40 stepping in), balance margins of 3 cm and more. The body jab's drop
# already carries the body forward, and its start was the tightest point of all; the step jab holds its weight
# forward until the rear foot is down.
SHIFTS = {
    "jab": Shift(0.028 * H, 5, 9),
    "jab_body": Shift(0.022 * H, None, 8),
    "step_jab": Shift(0.035 * H, None, 9),
}


@cache
def profile(speed: float = SPEED) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per output frame (the clip's 30 fps, ``speed`` times faster): the lead fist's extension and the torso's turn."""
    m, d, adr = model(ROBOTS["t800"])
    wrist, torso = m.body("LINK_WRIST_END_L").id, m.body("LINK_WAIST_YAW").id
    pos, yaw = [], []
    for r in np.loadtxt(CLIP, delimiter=","):
        set_pose(m, d, adr, r)
        frame = d.xmat[torso].reshape(3, 3)
        pos.append(frame.T @ (d.xpos[wrist] - d.xpos[torso]))  # in the torso's frame: the turn is not reach
        yaw.append(np.arctan2(frame[1, 0], frame[0, 0]))
    out = np.array(pos) - pos[0]
    peak = out[np.linalg.norm(out, axis=1).argmax()]
    extension = np.clip(out @ peak / (peak @ peak), 0.0, 1.0)  # the 1 cm pull-back before it is dropped
    turn = np.unwrap(yaw) - yaw[0]
    turn -= np.linspace(0.0, turn[-1], len(turn))  # back where it started
    t = np.arange(0.0, len(turn) - 1 + 1e-9, speed)
    rest = np.zeros(PAD)  # in guard, while the step jab's body finishes
    return np.r_[np.interp(t, np.arange(len(turn)), extension), rest], np.r_[
        np.interp(t, np.arange(len(turn)), turn), rest
    ]


def body_ramp(
    extension: NDArray[np.float64], lead: int, back: int, hold: int = 0, late: int = 0, *, top: int = 0
) -> NDArray[np.float64]:
    """The body's part in a strike: 0 in guard, 1 ``top`` frames after contact, by minimum jerk.

    Contact is ``late`` frames after the fist's full reach. Up from ``lead`` frames before the fist leaves the guard;
    held ``hold`` frames after the top, then down over ``back`` frames.
    """
    onset, peak = int(np.argmax(extension > 0.05)) + late, int(np.argmax(extension)) + late  # noqa: PLR2004
    top += peak
    frames = np.arange(len(extension))
    up, down = (frames - (onset - lead)) / (top - onset + lead), (frames - top - hold) / back
    return np.where(frames <= top, jerk(up), 1.0 - jerk(down))


def jerk(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimum jerk from 0 to 1 as ``s`` goes from 0 to 1, flat outside."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Arm(NamedTuple):
    """An arm for inverse kinematics."""

    body: int
    """the wrist end."""
    qadr: NDArray[np.int_]
    dadr: NDArray[np.int_]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    """the joints' ranges, the elbow's capped at ELBOW_MAX."""
    palm: NDArray[np.float64]
    """the palm's normal in the wrist end's frame (it faces the thigh with all joints at zero)."""


def arm_chain(m: mujoco.MjModel, side: str = "L") -> Arm:
    """The lead (``L``) or rear (``R``) arm of ``m``."""
    joints = [m.joint(n).id for n in (LEAD_ARM if side == "L" else REAR_ARM)]
    body = m.body(f"LINK_WRIST_END_{side}").id
    d = mujoco.MjData(m)
    d.qpos[3] = 1.0
    mujoco.mj_kinematics(m, d)
    lo, hi = m.jnt_range[joints].T.copy()
    hi[3] = ELBOW_MAX  # the elbow's pitch, fourth in either arm
    palm = d.xmat[body].reshape(3, 3).T @ np.array((0.0, -1.0, 0.0))
    return Arm(body, m.jnt_qposadr[joints], m.jnt_dofadr[joints], lo, hi, palm)


Residual = Callable[
    [mujoco.MjData, NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
]


def _solve(m: mujoco.MjModel, d: mujoco.MjData, arm: Arm, residual: Residual, steps: int = 60) -> None:
    """Damped least squares on the arm's joints in ``d``; ``residual(d, jp, jr)`` gives (error, Jacobian) rows."""
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(steps):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacBody(m, d, jp, jr, arm.body)
        e, jac = residual(d, jp[:, arm.dadr], jr[:, arm.dadr])
        step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(len(e)), e)
        step *= min(1.0, MAX_STEP / max(np.abs(step).max(), 1e-12))
        d.qpos[arm.qadr] = np.clip(d.qpos[arm.qadr] + step, arm.lo, arm.hi)
    mujoco.mj_kinematics(m, d)


def _palm_rows(
    d: mujoco.MjData, arm: Arm, jr: NDArray[np.float64], want: NDArray[np.float64], weight: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    palm = d.xmat[arm.body].reshape(3, 3) @ arm.palm
    return weight * (want - palm), weight * -np.cross(palm, jr.T).T


def contact(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    arm: Arm,
    line: tuple[NDArray[np.float64], NDArray[np.float64]],
    hold: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The arm in ``d`` at contact; returns the fist's position.

    The fist on ``line`` (from the shoulder to the aim point), the elbow ELBOW_CONTACT bent, the palm down, the
    shoulder held near ``hold`` with ``guard_weights``.
    """
    shoulder, aim_at = line
    toward = (aim_at - shoulder) / np.linalg.norm(aim_at - shoulder)
    across = np.eye(3) - np.outer(toward, toward)
    prior, weight = hold.copy(), guard_weights()
    prior[LEAD_ARM.index("J16_ELBOW_PITCH_L")], weight[LEAD_ARM.index("J16_ELBOW_PITCH_L")] = ELBOW_CONTACT, 3.0

    def residual(
        d: mujoco.MjData, jp: NDArray[np.float64], jr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        e_palm, j_palm = _palm_rows(d, arm, jr, np.array((0.0, 0.0, -1.0)), PALM_WEIGHT)
        e = np.r_[LINE_WEIGHT * across @ (shoulder - d.xpos[arm.body]), e_palm, weight * (prior - d.qpos[arm.qadr])]
        return e, np.r_[LINE_WEIGHT * across @ jp, j_palm, np.diag(weight)]

    _solve(m, d, arm, residual, steps=200)
    return d.xpos[arm.body].copy()


def aim(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    arm: Arm,
    pos: NDArray[np.float64],
    *,
    prior: NDArray[np.float64],
    weight: NDArray[np.float64],
) -> float:
    """The arm in ``d`` with its fist at ``pos``, each joint held near ``prior`` by ``weight``; the miss (m)."""

    def residual(
        d: mujoco.MjData, jp: NDArray[np.float64], jr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        del jr
        e = np.r_[LINE_WEIGHT * (pos - d.xpos[arm.body]), weight * (prior - d.qpos[arm.qadr])]
        return e, np.r_[LINE_WEIGHT * jp, np.diag(weight)]

    _solve(m, d, arm, residual)
    return float(np.linalg.norm(pos - d.xpos[arm.body]))


def guard_weights() -> NDArray[np.float64]:
    """Per lead-arm joint, how hard the IK holds it near the guard.

    The shoulder's roll keeps the elbow in (else the IK raises the arm sideways, a hook); its yaw leaves the palm's
    turn to the forearm.
    """
    return np.array([0.0, ROLL_WEIGHT, TWIST_WEIGHT, 0.0, 0.0])
