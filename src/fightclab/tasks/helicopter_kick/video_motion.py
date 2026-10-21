"""Helicopter kick reference from video: a Ginger Ninja tutorial clip through GVHMR and GMR, fitted to a robot.

Pipeline, each step in its own environment (``<robot>`` is g1, r1, h2, pm01, t800, t1 or asimov):
1. Cut the kick to a 30 fps clip with no cuts, the whole body in frame, a static camera.
2. GVHMR (~/GVHMR): ``python tools/demo/demo.py --video=<clip> -s`` -> hmr4d_results.pt
3. GMR (~/GMR): ``~/GMR/.venv/bin/python tools/gvhmr_to_csv.py <robot> <hmr4d_results.pt> /tmp/raw.csv``
4. This module: ``uv run python -m fightclab.tasks.helicopter_kick.video_motion <robot> /tmp/raw.csv /tmp/video.csv``
5. ``uv run python -m fightclab.tasks.helicopter_kick.motion_npz <robot> /tmp/video.csv``

The timing below was tuned on G1 and is used for every robot.

What the cleanup fixes:
- the tutorial shows the kick in slow motion, about 2x (a 1.2 s flight would need a 1.8 m jump);
- monocular height drifts: on the ground the lowest foot point is put on the floor, using the contact
  phases read off the video (TAKEOFF and LANDING, in clip seconds);
- G1 cannot jump 42 cm like the person, so the flight is time-compressed to FLIGHT seconds, in which the
  center of mass flies a parabola;
- the push-off (PUSH_START to TAKEOFF) is retimed so the center of mass rises from the crouch with constant
  acceleration and reaches the flight's takeoff speed (~2.7 m/s) while the feet are still down; PUSH_SPEED
  sets its length. Before, the reference reached that speed only after takeoff, accelerating upward in the
  air, and tracking policies rose on their toes but never flew.

Fitting to G1's motors (check with ``python -m fightclab.feasibility``): as reconstructed, the kick needed up
to 5x the torque of G1's waist motors. The heavy torso has to follow a pelvis that tumbles, spins at up to
12 rad/s and swings asymmetric arms, and two small 5020 motors carry all of it. So the spin-up before
takeoff is slowed down, the yaw and the pelvis tilt smoothed, the waist roll/pitch damped and the
reconstruction jitter filtered out.

The kick stays head-high on purpose (heel ~1.4 m; a G1 opponent's head is at 1.23-1.29 m). A head kick needs
both the high jump and the lean away from the kicking leg, and those still ask the knees and the waist for
about 1.5x their torque. Damping the lean and smoothing harder fits the motors, but the heel then only gets
to 0.7-1.1 m. The tracking task tolerates the shortfall and lets the policy find how high G1 can kick.
"""

import itertools
import sys
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from fightclab.feasibility import report
from fightclab.robots import asimov, h2, pm01, r1, t1, t800
from fightclab.tasks.helicopter_kick.reference import JOINTS


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
    flight: float | None = None
    """FLIGHT for this robot, if not G1's."""
    push_speed: float | None = None
    """PUSH_SPEED for this robot, if not G1's."""

    @property
    def joints(self) -> tuple[str, ...]:
        """The CSV's joint columns, in order."""
        if self.csv_joints:
            return self.csv_joints
        m = self.spec().compile()  # not spec.joints: after MjSpec.delete it can hold stale entries
        return tuple(m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE)


ROBOTS = {
    "g1": Robot(g1_constants.get_spec, ("waist_roll_joint", "waist_pitch_joint"), JOINTS),  # Unitree SDK order
    "r1": Robot(r1.get_spec, ("waist_roll_joint",)),
    "h2": Robot(
        h2.get_spec, ("waist_roll_joint", "waist_pitch_joint"), feet=("left_ankle_pitch_link", "right_ankle_pitch_link")
    ),
    "pm01": Robot(pm01.get_spec, (), feet=("LINK_FOOT_L", "LINK_FOOT_R"), root="LINK_BASE"),  # waist yaw only
    "t800": Robot(t800.get_spec, (), feet=("LINK_FOOT_L", "LINK_FOOT_R"), root="LINK_BASE"),  # torso yaw only
    # T1's crouch is shallow and its ankles barely point the toes (20 deg), so its center of mass rises only 10 cm in
    # the push-off, against G1's 21. That reaches ~1.3 m/s, a 0.3 s flight (~10 cm hop), not G1's 0.55 s; faster
    # push-offs were too short to survive the smoothing and left the takeoff speed short again.
    "t1": Robot(
        t1.get_spec, (), feet=("left_foot_link", "right_foot_link"), root="Trunk", flight=0.30, push_speed=1.2
    ),  # waist yaw only
    # Asimov's knee bends 86 deg and its ankles +-20: its push-off reaches ~1.5 m/s, a 0.4 s flight, not G1's 0.55 s.
    "asimov": Robot(asimov.get_spec, (), root="pelvis_link", flight=0.40, push_speed=1.2),  # waist yaw only
}

FPS = 30  # of the clip and of the output
G = 9.81
SLOWMO = 2.0
TAKEOFF, LANDING = 1.55, 2.75  # clip seconds, read off the video
FLIGHT = 0.55  # s on G1: a ~37 cm pelvis rise instead of the person's 42 cm
SPIN_UP_SLOW = 1.6  # the spin-up on the ground runs this much slower than the person
PUSH_START = 1.15  # clip second the push-off starts: the crouch's bottom
PUSH_SPEED = 1.0  # push-off length: 0.2 s at 1.0, ~1.7 g; at 2.0 it was 0.1 s and ~4.6 g, too fast to follow
YAW_WINDOW, TILT_WINDOW, JOINT_WINDOW = 21, 31, 9  # Savitzky-Golay windows, frames
TILT_SCALE = 1.0  # pelvis roll/pitch kept: leaning away is what lifts the heel to head height
WAIST_SCALE = 0.5  # waist roll/pitch kept
HOLD = 0.4  # s of the final pose appended, so the motion settles instead of stopping dead
SUNK = 0.005  # m below the floor that the summary reports as a sunk foot


def heights(rows: NDArray[np.float64], robot: Robot) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per frame: height of the lowest point of the feet's primitive geoms, and of the center of mass."""
    m = robot.spec().compile()
    d = mujoco.MjData(m)
    adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in robot.joints])
    soles = {m.body(foot).id for foot in robot.feet}
    shapes = (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)
    feet = [g for g in range(m.ngeom) if m.geom_bodyid[g] in soles and m.geom_type[g] in shapes]
    pelvis = m.body(robot.root).id
    low, com = np.zeros(len(rows)), np.zeros(len(rows))
    for i, r in enumerate(rows):
        d.qpos[:3], d.qpos[3:7], d.qpos[adr] = r[:3], (r[6], r[3], r[4], r[5]), r[7:]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        low[i] = min(_lowest(m, d, g) for g in feet)
        com[i] = d.subtree_com[pelvis, 2]
    return low, com


def _lowest(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> float:
    """Height of a sphere's, capsule's or box's lowest point."""
    z, rz, size = d.geom_xpos[g, 2], np.abs(d.geom_xmat[g].reshape(3, 3)[2]), m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        return float(z - rz @ size)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(z - rz[2] * size[1] - size[0])
    return float(z - size[0])


def push_start() -> float:
    """Output second the push-off starts."""
    return PUSH_START / SLOWMO * SPIN_UP_SLOW


def jump(robot: Robot | None = None) -> tuple[float, float]:
    """The robot's flight time and push-off speed."""
    if robot is None:
        return FLIGHT, PUSH_SPEED
    return robot.flight or FLIGHT, robot.push_speed or PUSH_SPEED


def phases(robot: Robot | None = None) -> tuple[float, float]:
    """Takeoff and landing in output (robot) seconds."""
    flight, push_speed = jump(robot)
    t0 = push_start() + (TAKEOFF - PUSH_START) / SLOWMO / push_speed
    return t0, t0 + flight


def clip_time(t: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """Clip seconds shown at output seconds: spin-up slowed, push-off sped up, flight squeezed, landing real time."""
    t0, t1 = phases(robot)
    flight, push_speed = jump(robot)
    t_push = push_start()
    spin_up = t * SLOWMO / SPIN_UP_SLOW
    push = PUSH_START + (t - t_push) * SLOWMO * push_speed
    air = TAKEOFF + (t - t0) / flight * (LANDING - TAKEOFF)
    post = LANDING + (t - t1) * SLOWMO
    return np.select([t < t_push, t < t0, t < t1], [spin_up, push, air], post)


def resample(rows: NDArray[np.float64], src: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows linearly interpolated at clip frame positions ``src``; quaternions renormalized."""
    i = np.clip(np.floor(src).astype(int), 0, len(rows) - 2)
    w = (src - i)[:, None]
    q0, q1 = rows[i, 3:7], rows[i + 1, 3:7]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    out = (1 - w) * rows[i] + w * rows[i + 1]
    out[:, 3:7] = (1 - w) * q0 + w * q1
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=-1, keepdims=True)
    return out


def even_push(rows: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """Retime the push-off so the grounded center of mass rises with constant acceleration from the crouch.

    The person's extension starts slow and ends fast; sped up enough to reach the takeoff speed, it asked for
    ~4.5 g in its last frames. Rising evenly from rest, the same height and takeoff speed take ~1.7 g.
    """
    t = np.arange(len(rows)) / FPS
    t_push, t0 = push_start(), phases(robot)[0]
    low, com = heights(rows, robot)
    push = (t >= t_push) & (t <= t0)
    height = np.maximum.accumulate((com - low)[push])  # the grounded center of mass, made monotonic
    u = (t[push] - t[push][0]) / (t[push][-1] - t[push][0])
    target = height[0] + (height[-1] - height[0]) * u**2
    out = rows.copy()
    out[push] = resample(rows, np.interp(target, height, t[push]) * FPS)
    return out


def ground_and_fly(rows: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """Lowest foot on the floor on the ground; in the air the center of mass flies a parabola.

    The parabola is for the center of mass, not the pelvis: the kicking leg swings up in the air, and with a
    ballistic pelvis the center of mass would rise with it, pushed by nothing.
    """
    t = np.arange(len(rows)) / FPS
    t0, t1 = phases(robot)
    low, com = heights(rows, robot)
    above = com - rows[:, 2]  # center of mass over the pelvis, set by the pose
    z = rows[:, 2] - low
    air = (t > t0) & (t < t1)
    for ground in (t <= t0, t >= t1):  # smoothed apart, so the takeoff's kink stays out of the air
        z[ground] = savgol_filter(z[ground], JOINT_WINDOW, 3)
    com = z + above
    i0, i1 = np.flatnonzero(air)[[0, -1]] + (-1, 1)  # the ground frames the flight joins
    flight = t[i1] - t[i0]
    v0 = (com[i1] - com[i0]) / flight + G * flight / 2
    tau = t[air] - t[i0]
    z[air] = com[i0] + v0 * tau - G * tau**2 / 2 - above[air]
    out = rows.copy()
    out[:, 2] = z
    return out


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
    m = robot.spec().compile()
    d = mujoco.MjData(m)
    adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in robot.joints])
    shapes = (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)
    feet = [m.body(foot).id for foot in robot.feet]
    geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] in shapes] for b in feet]
    pos, low = np.zeros((2, len(rows), 3)), np.zeros((2, len(rows)))
    for i, r in enumerate(rows):
        d.qpos[:3], d.qpos[3:7], d.qpos[adr] = r[:3], (r[6], r[3], r[4], r[5]), r[7:]
        mujoco.mj_kinematics(m, d)
        for f, b in enumerate(feet):
            pos[f, i] = d.xpos[b]
            low[f, i] = min(_lowest(m, d, g) for g in geoms[f])
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
    m = robot.spec().compile()
    d = mujoco.MjData(m)
    adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in robot.joints])
    legs = leg_chains(m, robot)
    keeps = []  # the feet's own orientation, per frame
    for r in rows:
        d.qpos[:3], d.qpos[3:7], d.qpos[adr] = r[:3], (r[6], r[3], r[4], r[5]), r[7:]
        mujoco.mj_kinematics(m, d)
        keeps.append([Rotation.from_quat(d.xquat[body][[1, 2, 3, 0]]) for body, *_ in legs])

    def solve(out: NDArray[np.float64]) -> NDArray[np.float64]:
        """Re-solve ``out``'s legs on its pelvis in place; returns how far each planted foot stays above its target."""
        short = np.zeros(len(rows))
        for i, r in enumerate(rows):
            d.qpos[:3], d.qpos[3:7], d.qpos[adr] = out[i, :3], out[i, [6, 3, 4, 5]], r[7:]
            mujoco.mj_kinematics(m, d)
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


def clean(rows: NDArray[np.float64], robot: Robot) -> NDArray[np.float64]:
    """A 30 fps clip-time robot motion -> 30 fps robot-time rows, fitted, grounded, with a G1-sized flight.

    Grounding comes after the smoothing: smoothing a grounded, ballistic height spread the takeoff's kink
    into the air, where the reference then kept accelerating upward with nothing to push against.
    """
    _t0, t1 = phases(robot)
    end = t1 + (len(rows) / FPS - LANDING) / SLOWMO
    t = np.arange(0.0, end, 1.0 / FPS)
    out = resample(rows, clip_time(t, robot) * FPS)
    out = np.concatenate([out, np.repeat(out[-1:], round(HOLD * FPS), axis=0)])
    return lift_sunk_feet(ground_and_fly(even_push(fit(out, robot), robot), robot), robot)


if __name__ == "__main__":
    name, raw, out = sys.argv[1:4]
    robot = ROBOTS[name]
    grounded = sys.argv[4:] == ["--ground"]  # a motion without flight: the kick's timing does not apply
    rows = (clean_ground if grounded else clean)(np.loadtxt(raw, delimiter=","), robot)
    np.savetxt(out, rows, delimiter=",", fmt="%.6f")
    low, com = heights(rows, robot)
    yaw = np.degrees(np.unwrap(2 * np.arctan2(rows[:, 5], rows[:, 6])))
    print(f"{len(rows)} frames, {len(rows) / FPS:.2f} s -> {out}")
    print(f"pelvis {rows[:, 2].min():.2f}..{rows[:, 2].max():.2f} m, turn {yaw[-1] - yaw[0]:+.0f} deg")
    if not grounded:  # the push-off has to reach the flight's takeoff speed while the feet are still down
        vz = np.diff(com) * FPS
        i0 = int(phases(robot)[0] * FPS)
        print(
            "center of mass speed up between frames around takeoff:",
            " ".join(f"{v:+.1f}" for v in vz[i0 - 4 : i0 + 3]),
            "m/s",
        )
    print(f"feet below the floor by >5 mm in {(low < -SUNK).sum()} frames, worst {low.min() * 1000:.0f} mm")
    if name == "g1":  # ponytail: the inverse-dynamics report knows G1 only; generalize when tuning another robot
        print("G1 motors:")
        report(rows, FPS)
