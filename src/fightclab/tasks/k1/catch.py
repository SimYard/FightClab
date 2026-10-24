"""The rear-hand catch of a head jab, for the footwork generator (``steps.py``), to the K-1 coach's measures.

The rear glove (the end of the right forearm: the robot has no hands) goes from the guard to an intercept point on
the incoming fist's line, 0.17 m in front of the head and just right of the lead glove, holds there past the jab's
full reach and comes back the same way. It goes forward first and in after (the sideways part starts IN_LAG of the
way out): moving straight there, the glove cut 1-1.5 cm into the head. Nothing else moves: the stance, pelvis,
torso, head and the lead glove, which covers the other half of the face, stay as in the guard.

The coach measured it against the approved jab at standing range (0.965 m pelvis to pelvis), in the defender's
opponent frame at its head: x at the opponent, y to the left, z up. The jab's fist reaches the intercept 0.12 s
after leaving its guard, so the glove has to start 0.05-0.08 s before the fist moves, on the attacker's hip turn or
on rhythm; a reaction to the fist comes too late. The duel policy times the command.
"""

import mujoco
import numpy as np
from numpy.typing import NDArray

from fightclab.motion import set_pose
from fightclab.tasks.k1.footwork import OPPONENT_YAW
from fightclab.tasks.k1.strikes import REAR_ARM, aim, arm_chain, jerk

HEAD = "LINK_HEAD_YAW"
# m, the right wrist end's origin at the catch (its glove sphere 0.190 m forward, 0.021 m right, 0.051 m up): within
# 2.4 cm of the fist's line for head jabs aimed 1.56-1.64 m high, which the defender can't tell apart in time. Set for
# a chin-high jab, a jab to the face's centre met its top edge at 27 deg and slid into the forehead (the coach). Left
# of this the shoulder's roll is at its limit (0.384 rad) and the lead glove, 8 mm away, covers the rest.
INTERCEPT = np.array((0.201, -0.056, 0.000))
# The shoulder's roll stays this far from its limit: a target at the stop leaves the tracker no room and sends a hit on
# the glove into the stop (the coach). The glove then holds 0.5 cm off INTERCEPT, and the gloves still overlap seen
# from the front.
ROLL_MARGIN = np.radians(5.0)
# The arm at the catch (the IK at INTERCEPT, the roll ROLL_MARGIN short): the forearm 56 deg up, 2.1 cm from the head.
CATCH_POSE = dict(zip(REAR_ARM, (-1.424, 0.297, 0.659, -1.521, -0.209), strict=True))
PRIOR_WEIGHT = 0.3  # how hard the IK holds each joint near the path between the guard and CATCH_POSE
OUT, HOLD, BACK = 4, 5, 6  # frames at 30 fps: out in 0.133 s, in place 0.05 s before contact to 0.1 s after, back 0.2 s
IN_LAG = 0.25  # of the way out before the glove starts in
FRAMES = OUT + HOLD + BACK


def _progress(i: int) -> NDArray[np.float64]:
    """How far toward the intercept the glove is in frame ``i``: forward and up, and in."""
    if i <= OUT:
        u = i / OUT
    elif i <= OUT + HOLD:
        u = 1.0
    else:
        u = 1.0 - (i - OUT - HOLD) / BACK  # the way out, reversed: in comes back first
    return jerk(np.array((u, (u - IN_LAG) / (1 - IN_LAG))))


def catch(m: mujoco.MjModel, d: mujoco.MjData, adr: NDArray[np.int_], rows: NDArray[np.float64]) -> None:
    """The catch in ``rows`` (its frames, the body still, the arms in guard); ``adr``: the rows' joints in qpos."""
    arm = arm_chain(m, "R")
    arm.hi[REAR_ARM.index("J19_SHOULDER_ROLL_R")] -= ROLL_MARGIN
    cols = 7 + np.array([list(adr).index(q) for q in arm.qadr])
    head = m.body(HEAD).id
    x = np.array((np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW), 0.0))
    frame = np.stack((x, np.cross((0.0, 0.0, 1.0), x), (0.0, 0.0, 1.0)), axis=1)  # columns: x, y (left), z
    set_pose(m, d, adr, rows[0])
    guard = frame.T @ (d.xpos[arm.body] - d.xpos[head])
    hold = d.qpos[arm.qadr].copy()
    pose = np.array([CATCH_POSE[n] for n in REAR_ARM])
    weight = np.full(len(REAR_ARM), PRIOR_WEIGHT)
    joints = hold
    for i in range(min(FRAMES + 1, len(rows))):
        set_pose(m, d, adr, rows[i])
        d.qpos[arm.qadr] = joints
        out, inward = _progress(i)
        local = guard + (INTERCEPT - guard) * np.array((out, inward, out))
        miss = aim(m, d, arm, d.xpos[head] + frame @ local, prior=hold + (pose - hold) * out, weight=weight)
        assert miss < 0.01, f"the catch's glove misses its path by {miss:.3f} m"  # noqa: PLR2004
        joints = d.qpos[arm.qadr].copy()
        rows[i, cols] = joints
