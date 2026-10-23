"""Robot motions as CSV rows at 30 fps (root pos, root quat xyzw, joints), and the cleanup every skill shares.

Reconstructions jitter, float and sink, and slide planted feet along the floor: ``clean_ground`` smooths and grounds
a motion, ``lock_feet`` holds its planted feet still. The helicopter kick's own retiming is in its ``video_motion``.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from fightclab.robots import asimov, h2, pm01, r1, t1, t800
from fightclab.robots.g1 import G1_JOINTS


@dataclass(frozen=True)
class Robot:
    """What the cleanup needs to know about a robot."""

    spec: Callable[[], mujoco.MjSpec]
    waist: tuple[str, ...]
    """Joints damped by WAIST_SCALE."""
    csv_joints: tuple[str, ...] | None = None
    """The CSV's joint columns if GMR writes them in another order than the model's."""
    feet: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    """The last body of each leg."""
    root: str = "pelvis"

    @cached_property
    def joints(self) -> tuple[str, ...]:
        """The CSV's joint columns, in order."""
        if self.csv_joints:
            return self.csv_joints
        m = self.spec().compile()  # not spec.joints: after MjSpec.delete it can hold stale entries
        return tuple(m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE)


ROBOTS = {
    "g1": Robot(g1_constants.get_spec, ("waist_roll_joint", "waist_pitch_joint"), G1_JOINTS),  # Unitree SDK order
    "r1": Robot(r1.get_spec, ("waist_roll_joint",)),
    "h2": Robot(
        h2.get_spec, ("waist_roll_joint", "waist_pitch_joint"), feet=("left_ankle_pitch_link", "right_ankle_pitch_link")
    ),
    "pm01": Robot(pm01.get_spec, (), feet=("LINK_FOOT_L", "LINK_FOOT_R"), root="LINK_BASE"),  # waist yaw only
    "t800": Robot(t800.get_spec, (), feet=("LINK_FOOT_L", "LINK_FOOT_R"), root="LINK_BASE"),  # torso yaw only
    "t1": Robot(t1.get_spec, (), feet=("left_foot_link", "right_foot_link"), root="Trunk"),  # waist yaw only
    "asimov": Robot(asimov.get_spec, (), root="pelvis_link"),  # waist yaw only
}

FPS = 30
YAW_WINDOW, TILT_WINDOW, JOINT_WINDOW = 21, 31, 9  # Savitzky-Golay windows, frames
TILT_SCALE = 1.0  # pelvis roll/pitch kept: leaning away is what lifts the kick's heel to head height
WAIST_SCALE = 0.5  # waist roll/pitch kept


def model(robot: Robot) -> tuple[mujoco.MjModel, mujoco.MjData, NDArray[np.int_]]:
    """The robot's model and data, and the qpos address of each CSV joint."""
    m = robot.spec().compile()
    return m, mujoco.MjData(m), np.array([m.jnt_qposadr[m.joint(n).id] for n in robot.joints])


def set_pose(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    adr: NDArray[np.int_],
    row: NDArray[np.float64],
    joints: NDArray[np.float64] | None = None,
) -> None:
    """``d`` in a row's pose, with ``joints`` instead of the row's if given, and its kinematics."""
    d.qpos[:3], d.qpos[3:7], d.qpos[adr] = row[:3], row[[6, 3, 4, 5]], (row[7:] if joints is None else joints)
    mujoco.mj_kinematics(m, d)


def soles(m: mujoco.MjModel, robot: Robot) -> list[list[int]]:
    """Per foot, what ``lowest`` measures: the spheres, capsules and boxes of its body."""
    shapes = (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)
    feet = [m.body(foot).id for foot in robot.feet]
    return [[g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] in shapes] for b in feet]


def lowest(m: mujoco.MjModel, d: mujoco.MjData, geoms: list[int]) -> float:
    """Height of the lowest point of ``geoms``, spheres, capsules and boxes."""

    def bottom(g: int) -> float:
        z, rz, size = d.geom_xpos[g, 2], np.abs(d.geom_xmat[g].reshape(3, 3)[2]), m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            return float(z - rz @ size)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return float(z - rz[2] * size[1] - size[0])
        return float(z - size[0])

    return min(bottom(g) for g in geoms)


def heights(rows: NDArray[np.float64], robot: Robot) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per frame: height of the lowest point of the feet's primitive geoms, and of the center of mass."""
    m, d, adr = model(robot)
    feet = [g for foot in soles(m, robot) for g in foot]
    pelvis = m.body(robot.root).id
    low, com = np.zeros(len(rows)), np.zeros(len(rows))
    for i, r in enumerate(rows):
        set_pose(m, d, adr, r)
        mujoco.mj_comPos(m, d)
        low[i] = lowest(m, d, feet)
        com[i] = d.subtree_com[pelvis, 2]
    return low, com


def fit(rows: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """Smooth the spin, smooth and damp the pelvis tilt and waist roll/pitch, filter the jitter."""
    out = rows.copy()
    yaw, pitch, roll = Rotation.from_quat(rows[:, 3:7]).as_euler("ZYX").T
    yaw = savgol_filter(np.unwrap(yaw), YAW_WINDOW, 2)
    pitch = savgol_filter(np.unwrap(pitch), TILT_WINDOW, 2) * TILT_SCALE
    roll = savgol_filter(np.unwrap(roll), TILT_WINDOW, 2) * TILT_SCALE
    out[:, 3:7] = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1)).as_quat()  # xyzw
    out[:, :3] = savgol_filter(rows[:, :3], JOINT_WINDOW, 3, axis=0)
    out[:, 7:] = savgol_filter(rows[:, 7:], JOINT_WINDOW, 3, axis=0)
    joints = robot.joints
    out[:, [7 + joints.index(j) for j in robot.waist]] *= WAIST_SCALE
    return out


def lift_sunk_feet(rows: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """Raise the pelvis wherever the smoothing pushed a foot into the floor, smoothly."""
    lift = np.clip(-heights(rows, robot)[0], 0.0, None)
    lift = np.maximum(savgol_filter(lift, 9, 2), lift)  # smooth, but never less than needed
    out = rows.copy()
    out[:, 2] += lift
    return out


def ground(rows: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """For motions without flight: the lowest foot on the floor in every frame, the correction smoothed.

    Video reconstructions float and sink by centimetres as the estimated height drifts.
    """
    offset = savgol_filter(-heights(rows, robot)[0], 15, 2)
    out = rows.copy()
    out[:, 2] += offset
    return lift_sunk_feet(out, robot)


def clean_ground(rows: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """A 30 fps clip of a motion that never leaves the floor (stance, steps, kicks): fitted and grounded."""
    return ground(fit(rows, robot), robot)


LOCK_HEIGHT = 0.03  # m: a sole this close to the floor is down
HOVER = 0.08  # m: a foot this low that moves slower than LOCK_SPEED is planted (put down on the floor if hovering)
LOCK_SPEED = 0.25  # m/s
STEP_MAX = 0.45  # s: the longest a step keeps its foot up (reconstructions left feet hanging for seconds)
LOCK_MIN = 0.1  # s: the shortest contact phase kept
STEP_MIN = 0.02  # m a foot travels between contacts for it to be a step
STEP_LIFT = 0.04  # m: the least a step lifts the sole (the K-1 coach's bar is 3 cm; GVHMR flattens steps to slides)
LOCK_MAX = 0.15  # m: a contact phase whose foot would move farther than this to hold still is not really planted
IK_STEPS = 10
IK_TOL = 1e-4  # m and rad
IK_MISS = 0.02  # m: a leg whose foot ends farther than this from its target keeps its original joints
DROP_MAX = 0.04  # m the pelvis may come down so that a straight leg reaches the floor


def runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """(start, stop) of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def contacts(
    rows: NDArray[np.float64], robot: Robot
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Per foot and frame: the foot body's position (2, N, 3), its sole's height and whether it is planted (2, N)."""
    m, d, adr = model(robot)
    feet = [m.body(foot).id for foot in robot.feet]
    geoms = soles(m, robot)
    pos, low = np.zeros((2, len(rows), 3)), np.zeros((2, len(rows)))
    for i, r in enumerate(rows):
        set_pose(m, d, adr, r)
        for f, b in enumerate(feet):
            pos[f, i] = d.xpos[b]
            low[f, i] = lowest(m, d, geoms[f])
    speed = np.linalg.norm(np.gradient(pos[..., :2], axis=1), axis=-1) * FPS
    planted = (low < HOVER) & (speed < LOCK_SPEED)
    for f in range(2):
        for a, b in runs(planted[f]):
            if b - a < LOCK_MIN * FPS:
                planted[f, a:b] = False
    return pos, low, planted


STEADY_WINDOW = 31  # frames (~1 s): the pelvis' roll, pitch and height smoothed over this for a steady stance


def steady_pelvis(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """``rows`` with the pelvis' roll, pitch and height smoothed over STEADY_WINDOW; heading and path kept."""
    out = rows.copy()
    yaw, pitch, roll = Rotation.from_quat(rows[:, 3:7]).as_euler("ZYX").T
    pitch, roll = (savgol_filter(np.unwrap(a), STEADY_WINDOW, 1) for a in (pitch, roll))
    out[:, 3:7] = Rotation.from_euler("ZYX", np.stack((yaw, pitch, roll), axis=1)).as_quat()
    out[:, 2] = savgol_filter(rows[:, 2], STEADY_WINDOW, 1)
    return out


Leg = tuple[int, NDArray[np.int_], NDArray[np.int_], NDArray[np.float64]]


def leg_chains(m: mujoco.MjModel, robot: Robot) -> list[Leg]:
    """Per foot: its body, and the qpos addresses, dof addresses and ranges (2, n) of the hinges up to the pelvis."""
    legs = []
    for foot in robot.feet:
        body, joints = m.body(foot).id, []
        while body:
            joints += [
                j for j in range(m.njnt) if m.jnt_bodyid[j] == body and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            ]
            body = m.body_parentid[body]
        hinges = np.array(joints)
        legs.append((m.body(foot).id, m.jnt_qposadr[hinges], m.jnt_dofadr[hinges], m.jnt_range[hinges].T))
    return legs


def reach(m: mujoco.MjModel, d: mujoco.MjData, leg: Leg, pos: NDArray[np.float64], turn: Rotation) -> float:
    """Damped least squares on ``leg``'s joints in ``d`` until its foot is at ``pos`` turned ``turn``; the miss (m)."""
    body, qadr, dadr, (lo, hi) = leg
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(IK_STEPS):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        e = np.r_[pos - d.xpos[body], (turn * Rotation.from_quat(d.xquat[body][[1, 2, 3, 0]]).inv()).as_rotvec()]
        if np.abs(e).max() < IK_TOL:
            break
        mujoco.mj_jacBody(m, d, jp, jr, body)
        jac = np.vstack((jp[:, dadr], jr[:, dadr]))
        d.qpos[qadr] = np.clip(d.qpos[qadr] + jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e), lo, hi)
    mujoco.mj_kinematics(m, d)
    return float(np.linalg.norm(pos - d.xpos[body]))


def lock_feet(rows: NDArray[np.float64], robot: Robot, steady: bool = False) -> NDArray[np.float64]:
    """Planted feet stay put: video reconstructions slide them along the floor, and a tracking policy copies that.

    In each contact phase (a foot under HOVER moving slower than LOCK_SPEED) the foot is held at its median position
    with its sole on the floor (unless that is over LOCK_MAX away: then it was not planted); between phases the
    correction fades linearly. Reconstructions also flatten the steps between: a foot that travels over STEP_MIN
    between two contacts rises at least STEP_LIFT (a half sine) while it travels, at most STEP_MAX, and while the
    other foot stands; lifting both made the body float, lifting through whole swings left feet hanging.

    The legs' joints are re-solved by damped least squares so each foot reaches its new position with its orientation
    unchanged (a raised heel stays raised); a leg that cannot get there within IK_MISS keeps its joints. The pelvis
    stays (``steady``: it stops rocking, see ``steady_pelvis``), except where a straight leg could not put a planted
    foot down: there it comes down as far as needed, smoothly, and the legs are solved again. Last, the lower sole
    goes back on the floor (``ground``).
    """
    pos, low, planted = contacts(rows, robot)
    target = pos.copy()
    frames = np.arange(len(rows))
    lift, arcs = np.zeros((2, len(rows)), dtype=bool), target[..., 2].copy()
    down = planted | (low < LOCK_HEIGHT)
    for f in range(2):
        shift = np.full((len(rows), 3), np.nan)
        phases = []
        for a, b in runs(planted[f]):
            held = np.median(pos[f, a:b, :2], axis=0) - pos[f, a:b, :2]
            if np.linalg.norm(held, axis=-1).max() < LOCK_MAX:
                shift[a:b] = np.c_[held, -low[f, a:b]]  # held still, its sole on the floor
                phases.append((a, b))
        known = np.flatnonzero(~np.isnan(shift[:, 0]))
        for c in range(3):
            target[f, :, c] += np.interp(frames, known, shift[known, c]) if len(known) else 0.0
        for (_, a), (b, _) in itertools.pairwise(phases):  # the swing between two contacts
            if np.linalg.norm(target[f, b, :2] - target[f, a - 1, :2]) > STEP_MIN:
                # The step: where the foot really travels, at most STEP_MAX around its fastest frame.
                v = np.linalg.norm(np.gradient(target[f, a - 1 : b + 1, :2], axis=0), axis=-1)
                moving = np.flatnonzero(v > 0.3 * v.max())
                half = round(STEP_MAX * FPS / 2)
                peak = a - 1 + int(np.argmax(v))
                s = max(a - 1 + moving[0] - 2, peak - half, a - 1)
                e = min(a - 1 + moving[-1] + 2, peak + half, b)
                u = (frames[s : e + 1] - s) / max(e - s, 1)
                floor = np.interp(frames[s : e + 1], (a - 1, b), pos[f, [a - 1, b], 2] - low[f, [a - 1, b]])
                arcs[f, s : e + 1] = np.maximum(target[f, s : e + 1, 2], floor + STEP_LIFT * np.sin(np.pi * u))
                lift[f, s : e + 1] = down[1 - f, s : e + 1]  # the other foot is down
    # Where both would lift, the faster foot steps and the other stays down.
    speed = np.linalg.norm(np.gradient(target[..., :2], axis=1), axis=-1)
    both = lift.all(0)
    lift[0] &= ~both | (speed[0] >= speed[1])
    lift[1] &= ~both | (speed[1] > speed[0])
    target[..., 2] = np.where(lift, arcs, target[..., 2])
    m, d, adr = model(robot)
    legs = leg_chains(m, robot)
    keeps = []  # the feet's own orientation, per frame
    for r in rows:
        set_pose(m, d, adr, r)
        keeps.append([Rotation.from_quat(d.xquat[body][[1, 2, 3, 0]]) for body, *_ in legs])

    def solve(out: NDArray[np.float64]) -> NDArray[np.float64]:
        """Re-solve ``out``'s legs on its pelvis in place; returns how far each planted foot stays above its target."""
        short = np.zeros(len(rows))
        for i, r in enumerate(rows):
            set_pose(m, d, adr, out[i], r[7:])
            for f, leg in enumerate(legs):
                body, qadr = leg[0], leg[1]
                start = d.qpos[qadr].copy()
                reach(m, d, leg, target[f, i], keeps[i][f])
                if planted[f, i] and np.linalg.norm(target[f, i, :2] - d.xpos[body][:2]) < IK_MISS / 2:  # only short
                    short[i] = max(short[i], d.xpos[body][2] - target[f, i, 2])
                if np.linalg.norm(target[f, i] - d.xpos[body]) > IK_MISS:
                    d.qpos[qadr] = start
            mujoco.mj_kinematics(m, d)
            out[i, 7:] = d.qpos[adr]
        return short

    out = steady_pelvis(rows) if steady else rows.copy()
    short = solve(out)
    # A planted foot a straight leg cannot put down: the pelvis comes down, smoothly, and the legs are solved again.
    drop = np.clip(short, 0.0, DROP_MAX)
    if drop.max() > IK_TOL:
        drop = np.maximum(savgol_filter(maximum_filter1d(drop, STEADY_WINDOW // 2), STEADY_WINDOW, 1), drop)
        out[:, 2] -= drop
        solve(out)
    return ground(out, robot)


def joint_speed(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per frame, the fastest joint's speed (rad/s)."""
    return np.abs(np.gradient(rows[:, 7:], axis=0)).max(1) * FPS


def floating(rows: NDArray[np.float64], robot: Robot) -> float:
    """Share of frames with both soles over 3 cm up (footwork has no flight)."""
    _, low, _ = contacts(rows, robot)
    return float((low.min(0) > 0.03).mean())  # noqa: PLR2004


def skating(rows: NDArray[np.float64], robot: Robot) -> float:
    """Share of frames with a sole under 1.5 cm whose foot moves faster than 0.15 m/s (the K-1 coach: under 5%)."""
    pos, low, _ = contacts(rows, robot)
    speed = np.linalg.norm(np.gradient(pos[..., :2], axis=1), axis=-1) * FPS
    return float((speed[low < 0.015] > 0.15).mean())  # noqa: PLR2004
