"""Reference motion for the helicopter kick, built from keyframes.

Each keyframe sets the pelvis position and heading, where the feet are (planted on the floor, or swinging
relative to the pelvis) and where the palms are relative to the torso. Every keyframe is solved by
damped least-squares IK; joints and the pelvis are then splined between keyframes, with a ballistic
pelvis height in the air.

The kick (orthodox, spinning right, see ``KickProgressCfg.direction``): guard -> load the left foot and turn
on it while the right leg steps around -> push off the left -> split in the air with the right heel high ->
land on the left foot -> pivot on it and step the right into the guard, facing the same way as at the start.
Timing and the turn budget follow a Ginger Ninja tutorial clip reconstructed with GVHMR and GMR.

Make the motion file the tracking task loads:
    uv run python -m fightclab.tasks.helicopter_kick.reference /tmp/helicopter_kick.csv
    MUJOCO_GL=egl WANDB_MODE=disabled uv run -m mjlab.scripts.csv_to_npz --input-file /tmp/helicopter_kick.csv \
        --output-name helicopter_kick --input-fps 30 --output-fps 50 --render True
    cp /tmp/motion.npz src/fightclab/tasks/helicopter_kick/helicopter_kick.npz
csv_to_npz saves /tmp/motion.npz before it tries to upload to WandB, so the failed upload can be ignored.
"""

import itertools
import math
import sys
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline

from fightclab.robots.g1 import GUARD_FEET, GUARD_HANDS, GUARD_KEYFRAME, GUARD_PELVIS_HEIGHT

FPS = 30
G = 9.81
# Joint order of mjlab's csv_to_npz (Unitree's SDK order).
JOINTS = (
    *(f"left_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    *(f"right_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *(f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")),
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    *(f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")),
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class Foot:
    """A foot target.

    Planted: on the floor at world (x, y) with the toes at ``yaw`` degrees.
    Swinging: (x, y, z) from the pelvis, in its heading frame.
    """

    pos: tuple[float, ...]
    planted: bool
    yaw: float = 0.0


def planted(x: float, y: float, yaw: float = 0.0) -> Foot:
    """A foot flat on the floor."""
    return Foot((x, y), True, yaw)


def swing(x: float, y: float, z: float) -> Foot:
    """A foot in the air, relative to the pelvis."""
    return Foot((x, y, z), False)


@dataclass(frozen=True)
class Key:
    """One keyframe. ``z`` None means ballistic: the pelvis is in the air."""

    name: str
    t: float
    xy: tuple[float, float]
    z: float | None
    yaw: float  # degrees, negative turns right
    feet: tuple[Foot, Foot]  # left, right
    hands: tuple[tuple[float, float, float], tuple[float, float, float]]  # left, right palm in torso frame


TUCK = ((0.26, 0.06, 0.22), (0.26, -0.06, 0.22))  # palms at the chest, to spin fast
# Van Damme's arms at the strike (Bloodsport): the arm away from the kicking leg straight out and up as a
# counterweight, the kicking side's arm bent in at the chest.
STRIKE = ((0.0, 0.48, 0.40), (0.25, -0.08, 0.20))
GL, GR = GUARD_FEET
TAKEOFF, LANDING = 0.80, 1.22
TAKEOFF_Z = 0.76
LAND_L = (0.25, 0.10)  # where the left foot lands; it pivots there into the final guard
FINAL_XY = (LAND_L[0] - GL[0], LAND_L[1] - GL[1])  # pelvis in the final guard
FINAL = (planted(*LAND_L, -360), planted(FINAL_XY[0] + GR[0], FINAL_XY[1] + GR[1], -360))

# Turn budget from the Ginger Ninja clip through GVHMR: about 90 degrees on the ground before takeoff, 160 in
# the air, the last 110 after landing, pivoting on the left foot while the right steps into the guard.
# Earlier versions asked for 240-270 in the air; tracking on G1 with Unitree's motor curves never managed it.
# On one foot the pelvis stands over it, or the reference cannot balance. No crouch below 0.69 m: lower,
# the rear ankle hits its dorsiflexion limit and the foot cannot stay flat.
KEYS = (
    Key("guard", 0.00, (0.0, 0.0), GUARD_PELVIS_HEIGHT, 0, (planted(*GL), planted(*GR)), GUARD_HANDS),
    Key(
        "load left",
        0.30,
        (0.09, 0.06),
        0.69,
        -20,
        (planted(*GL, -10), planted(*GR)),
        ((0.25, -0.05, 0.25), (0.0, -0.35, 0.20)),
    ),
    Key(
        "turn on left foot",
        0.60,
        (GL[0] - 0.02, GL[1] - 0.02),
        0.69,
        -55,
        (planted(*GL, -50), swing(-0.05, -0.25, -0.50)),
        ((0.25, -0.05, 0.25), (0.0, -0.35, 0.20)),
    ),
    Key("push off", TAKEOFF, GL, TAKEOFF_Z, -90, (planted(*GL, -90), swing(0.25, -0.35, -0.30)), TUCK),
    Key(
        "legs open",
        0.92,
        (0.19, 0.12),
        None,
        -140,
        (swing(0.0, 0.45, -0.35), swing(0.0, -0.55, -0.25)),
        ((0.10, 0.35, 0.30), TUCK[1]),
    ),
    # Kicking leg above the pelvis, the other leg lower, Van Damme's arms (Bloodsport, Ginger Ninja tutorial).
    # The heel points front-left here, about where it does in the clip.
    Key("strike", 1.03, (0.22, 0.11), None, -200, (swing(0.15, 0.60, -0.30), swing(0.0, -0.66, 0.12)), STRIKE),
    Key("legs close", 1.13, (0.24, 0.10), None, -235, (swing(0.05, 0.20, -0.50), swing(0.10, -0.40, -0.25)), TUCK),
    # Land on the left foot, as in the clip, and keep turning on it.
    Key("touch down left", LANDING, LAND_L, TAKEOFF_Z, -250, (planted(*LAND_L, -250), swing(0.05, -0.35, -0.40)), TUCK),
    Key("pivot", 1.40, LAND_L, 0.70, -310, (planted(*LAND_L, -300), swing(-0.08, -0.25, -0.50)), GUARD_HANDS),
    Key("step into guard", 1.55, FINAL_XY, 0.69, -360, FINAL, GUARD_HANDS),
    Key("guard", 1.85, FINAL_XY, GUARD_PELVIS_HEIGHT, -360, FINAL, GUARD_HANDS),
    Key("hold", 2.30, FINAL_XY, GUARD_PELVIS_HEIGHT, -360, FINAL, GUARD_HANDS),
)

# Ballistic flight between takeoff and landing at the same pelvis height.
V0 = G * (LANDING - TAKEOFF) / 2


def pelvis_z(t: float) -> float:
    """Ballistic pelvis height in the air."""
    tau = t - TAKEOFF
    return TAKEOFF_Z + V0 * tau - G * tau**2 / 2


def yaw_quat(yaw: float) -> NDArray[np.float64]:
    """MuJoCo wxyz quaternion for a heading in radians."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def yaw_mat(yaw: float) -> NDArray[np.float64]:
    """Rotation matrix for a heading in radians."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Solver:
    """Damped least-squares IK on G1 with the pelvis pose given and the waist at zero."""

    def __init__(self) -> None:
        """Load G1 on a lit floor (for the storyboard) and pick the joints IK may move."""
        spec = mujoco.MjSpec.from_file(str(G1_XML))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(5, 5, 0.1), rgba=(0.3, 0.4, 0.5, 1))
        spec.worldbody.add_light(pos=(0, 0, 4), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8))
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        hinges = [self.m.joint(n).id for n in JOINTS]
        movable = [j for j in hinges if not self.m.joint(j).name.startswith("waist")]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in movable])
        self.cols = [self.m.jnt_dofadr[j] for j in movable]
        self.lo, self.hi = self.m.jnt_range[movable].T * 0.9
        self.out_adr = np.array([self.m.jnt_qposadr[j] for j in hinges])
        legs = hinges[:12]
        self.leg_qadr = np.array([self.m.jnt_qposadr[j] for j in legs])
        self.leg_cols = [self.m.jnt_dofadr[j] for j in legs]
        self.leg_lo, self.leg_hi = self.m.jnt_range[legs].T * 0.9
        self.guard = np.array([GUARD_KEYFRAME.joint_pos.get(n, 0.0) for n in JOINTS])
        self.torso = self.m.body("torso_link").id

    def _jac(self, s: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, s)
        return jp, jr

    def _plant(self, s: int, foot: Foot) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        """Error and Jacobian rows that hold a foot site flat on the floor at its spot and heading."""
        jp, jr = self._jac(s)
        want = yaw_mat(math.radians(foot.yaw))
        have = self.d.site_xmat[s].reshape(3, 3)
        rot = 0.5 * sum(np.cross(have[:, i], want[:, i]) for i in range(3))
        return [rot, np.array((*foot.pos, 0.0)) - self.d.site_xpos[s]], [jr, jp]

    def solve(self, pelvis: NDArray[np.float64], yaw: float, key: Key) -> NDArray[np.float64]:
        """Joint angles in ``JOINTS`` order for one keyframe.

        Every solve starts from the guard: from straight legs it gets stuck at the knee limit.
        """
        m, d = self.m, self.d
        d.qpos[:3], d.qpos[3:7] = pelvis, yaw_quat(yaw)
        d.qpos[self.out_adr] = self.guard
        heading = yaw_mat(yaw)
        miss: dict[str, float] = {}
        for _ in range(3000):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            errs, jacs = [], []
            for site, foot in zip(("left_foot", "right_foot"), key.feet, strict=True):
                s = m.site(site).id
                if foot.planted:
                    e, j = self._plant(s, foot)
                    errs += e
                    jacs += j
                else:
                    errs.append(pelvis + heading @ np.array(foot.pos) - d.site_xpos[s])
                    jacs.append(self._jac(s)[0])
                miss[site] = float(np.linalg.norm(errs[-1]))
            torso_r = d.xmat[self.torso].reshape(3, 3)
            for site, palm in zip(("left_palm", "right_palm"), key.hands, strict=True):
                s = m.site(site).id
                errs.append(d.xpos[self.torso] + torso_r @ np.array(palm) - d.site_xpos[s])
                jacs.append(self._jac(s)[0])
                miss[site] = float(np.linalg.norm(errs[-1]))
            e, jac = np.concatenate(errs), np.vstack(jacs)[:, self.cols]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(len(e)), e)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + 0.5 * dq, self.lo, self.hi)
        print(f"{key.name:18s} miss, mm:", " ".join(f"{k}={v * 1000:.0f}" for k, v in miss.items()))
        return d.qpos[self.out_adr].copy()

    def pose(self, pelvis: NDArray[np.float64], yaw: float, joints: NDArray[np.float64]) -> None:
        """Put the robot in a pose."""
        self.d.qpos[:3], self.d.qpos[3:7], self.d.qpos[self.out_adr] = pelvis, yaw_quat(yaw), joints
        mujoco.mj_forward(self.m, self.d)

    def fix_feet(
        self, pelvis: NDArray[np.float64], yaw: float, joints: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], float]:
        """Re-solve the legs of one frame: planted feet flat on their spot, swinging feet above ``CLEARANCE``.

        Splining joints between keyframes moves the feet freely; without this they slid and sank up to 15 cm
        into the floor. Returns the joints and the largest remaining error.
        """
        self.pose(pelvis, yaw, joints)
        worst = 0.0
        for _ in range(500):
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            errs, jacs = [], []
            for side, site in enumerate(("left_foot", "right_foot")):
                s = self.m.site(site).id
                foot = planted_at(side, t)
                if foot is not None:
                    e, j = self._plant(s, foot)
                    errs += e
                    jacs += j
                elif self.d.site_xpos[s][2] < CLEARANCE:
                    errs.append(np.array([CLEARANCE - self.d.site_xpos[s][2]]))
                    jacs.append(self._jac(s)[0][2:3])
            if not errs:
                break
            e, jac = np.concatenate(errs), np.vstack(jacs)[:, self.leg_cols]
            worst = float(np.abs(e).max())
            if worst < IK_TOL:
                break
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(len(e)), e)
            self.d.qpos[self.leg_qadr] = np.clip(self.d.qpos[self.leg_qadr] + 0.5 * dq, self.leg_lo, self.leg_hi)
        return self.d.qpos[self.out_adr].copy(), worst


CLEARANCE = 0.03  # lowest a swinging foot's sole may go, m
IK_TOL = 1e-4  # converged when every error is below this, m or rad


def planted_at(side: int, t: float) -> Foot | None:
    """Foot ``side`` (0 left, 1 right) on the floor at time ``t``, or None while it swings.

    Planted between two keyframes that both plant it, sliding linearly between their spots and headings
    (the pivot on the left foot before takeoff).
    """
    for a, b in itertools.pairwise(KEYS):
        if not a.t <= t <= b.t:
            continue
        fa, fb = a.feet[side], b.feet[side]
        if fa.planted and fb.planted:
            w = (t - a.t) / (b.t - a.t)
            pos = tuple((1 - w) * pa + w * pb for pa, pb in zip(fa.pos, fb.pos, strict=True))
            return Foot(pos, True, (1 - w) * fa.yaw + w * fb.yaw)
        if fa.planted and math.isclose(t, a.t, abs_tol=1e-6):
            return fa
        if fb.planted and math.isclose(t, b.t, abs_tol=1e-6):
            return fb
        return None
    return None


def build(solver: Solver) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Keyframe poses and the motion CSV rows.

    Keyframe poses are [K, 3 + 1 + joints]: pelvis xyz, yaw, joints.
    CSV rows are base pos, base quat xyzw, joints in ``JOINTS`` order, at ``FPS``.
    """
    times = np.array([k.t for k in KEYS])
    zs = [k.z if k.z is not None else pelvis_z(k.t) for k in KEYS]
    yaws = np.radians([k.yaw for k in KEYS])
    joints = np.stack([solver.solve(np.array((*k.xy, z)), yaw, k) for k, z, yaw in zip(KEYS, zs, yaws, strict=True)])
    keys = np.column_stack([[k.xy for k in KEYS], zs, yaws, joints])

    t = np.arange(0.0, KEYS[-1].t + 1e-9, 1.0 / FPS)
    xy = CubicSpline(times, [k.xy for k in KEYS], bc_type="clamped")(t)
    yaw = CubicSpline(times, yaws, bc_type="clamped")(t)
    q = CubicSpline(times, joints, bc_type="clamped")(t)
    ground_before = times <= TAKEOFF
    ground_after = times >= LANDING
    z_before = CubicSpline(times[ground_before], np.array(zs)[ground_before], bc_type=((1, 0.0), (1, V0)))
    z_after = CubicSpline(times[ground_after], np.array(zs)[ground_after], bc_type=((1, -V0), (1, 0.0)))
    z = np.where(t <= TAKEOFF, z_before(t), np.where(t >= LANDING, z_after(t), [pelvis_z(x) for x in t]))
    fixed = [solver.fix_feet(np.array((*xy[i], z[i])), yaw[i], q[i], t[i]) for i in range(len(t))]
    q = np.stack([f[0] for f in fixed])
    print(f"feet fixed per frame, worst remaining error {max(f[1] for f in fixed) * 1000:.1f} mm")
    quat_xyzw = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1)
    return keys, np.column_stack([xy, z, quat_xyzw, q])


def storyboard(solver: Solver, keys: NDArray[np.float64], path: str) -> None:
    """Render every keyframe from a camera fixed in the world, so the turn shows, into one labelled image."""
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (0.05, 0.0, 0.7), 3.2, 150, -12
    tiles = []
    with mujoco.Renderer(solver.m, 360, 300) as renderer:
        for key, row in zip(KEYS, keys, strict=True):
            solver.pose(row[:3], row[3], row[4:])
            renderer.update_scene(solver.d, cam)
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text((8, 6), f"{key.t:.2f} s  {key.yaw:+.0f} deg\n{key.name}", fill="white")
            tiles.append(tile)
    cols = 4
    sheet = Image.new("RGB", (cols * 300, -(-len(tiles) // cols) * 360))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 300, (i // cols) * 360))
    sheet.save(path)


if __name__ == "__main__":
    out, *board = sys.argv[1:] or ["/tmp/helicopter_kick.csv"]
    solver = Solver()
    keys, rows = build(solver)
    np.savetxt(out, rows, delimiter=",", fmt="%.6f")
    print(f"{len(rows)} frames, {len(rows) / FPS:.2f} s -> {out}")
    for path in board:
        storyboard(solver, keys, path)
        print(f"storyboard -> {path}")
