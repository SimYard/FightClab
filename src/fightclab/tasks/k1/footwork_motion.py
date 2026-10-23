"""The footwork clips as one continuous motion for the tracking task, which takes a single motion file.

    uv run python -m fightclab.tasks.k1.footwork_motion OUT.csv CHUNK.npz...

Each chunk (a 60 s piece of a tutorial: CHUNK.npz from ``motion_npz`` next to its grounded CSV, CHUNK.csv) gives its
guarded runs (``amp.guarded_runs``), merged across gaps under GAP and kept from MIN_SEGMENT on. Every segment is
turned and moved to start where the last one ended, and a BLEND-long interpolation joins them. Planted feet are
then held still and the pelvis steadied (``motion.lock_feet``): the reconstruction slid the feet, and the
coach rocked from side to side. The arms, torso and head hold one guard (GUARD_POSE): the reference is for footwork
only. Then ``motion_npz t800 OUT.csv <out.npz>`` replays it for the tracking task.
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from fightclab.motion import ROBOTS, contacts, floating, joint_speed, lock_feet, skating
from fightclab.robots import t800
from fightclab.tasks.k1.amp import guarded_runs, in_guard

FPS, NPZ_FPS = 30, 50
# Held through the footwork: the guard the K-1 coach passed (high, gloves at the temples, elbows in) and the clips'
# median torso turn. The coach's own arm and torso movements, and the reconstruction's jitter there, were noise for a
# footwork reference, and the tracker copied them.
GUARD_POSE = {
    "J12_TORSO_YAW": 0.195, "J13_SHOULDER_PITCH_L": -1.170, "J14_SHOULDER_ROLL_L": -0.250,
    "J15_SHOULDER_YAW_L": -0.360, "J16_ELBOW_PITCH_L": -2.150, "J17_ELBOW_YAW_L": -0.260,
    "J18_SHOULDER_PITCH_R": -1.290, "J19_SHOULDER_ROLL_R": 0.250, "J20_SHOULDER_YAW_R": 0.330,
    "J21_ELBOW_PITCH_R": -2.040, "J22_ELBOW_YAW_R": -0.340, "J23_HEAD_PITCH": 0.0, "J24_HEAD_YAW": 0.0,
}  # fmt: skip
GAP = 1.0  # s: runs closer than this are one segment, the short lapse included
MIN_SEGMENT = 2.0  # s
BLEND = 0.5  # s


def segments(npz: Path, names: list[str]) -> list[slice]:
    """CSV row ranges of a chunk's guarded runs, merged across short gaps unless a hand leaves the guard in them.

    A punch is shorter than GAP: merged in, the coach's jabs came back into the footwork.
    """
    clip = dict(np.load(npz))
    guard = in_guard(clip, names)
    runs = [(r.start, r.stop) for r in guarded_runs(clip, names)]
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < GAP * NPZ_FPS and guard[merged[-1][1] : a].all():
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [slice(a * FPS // NPZ_FPS, b * FPS // NPZ_FPS) for a, b in merged if b - a >= MIN_SEGMENT * NPZ_FPS]


def heading(quat_xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    """Yaw of the pelvis' x axis."""
    return Rotation.from_quat(quat_xyzw).as_euler("ZYX")[..., 0]


def place(rows: NDArray[np.float64], end: NDArray[np.float64]) -> NDArray[np.float64]:
    """``rows`` turned about z and moved on the floor so that they start at ``end``'s position and heading."""
    turn = Rotation.from_euler("z", heading(end[3:7]) - heading(rows[0, 3:7]))
    out = rows.copy()
    out[:, :2] = turn.apply(rows[:, :3] - rows[0, :3])[:, :2] + end[:2]
    out[:, 3:7] = (turn * Rotation.from_quat(rows[:, 3:7])).as_quat()
    return out


def blend(a: NDArray[np.float64], b: NDArray[np.float64], seconds: float = BLEND) -> NDArray[np.float64]:
    """``seconds`` of rows from pose ``a`` to pose ``b``, both ends excluded."""
    n = round(seconds * FPS)
    w = np.arange(1, n + 1)[:, None] / (n + 1)
    out = (1 - w) * a + w * b
    out[:, 3:7] = Slerp([0, 1], Rotation.from_quat([a[3:7], b[3:7]]))(w[:, 0]).as_quat()
    return out


def hold_guard(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """``rows`` with the upper body held in GUARD_POSE."""
    out, joints = rows.copy(), ROBOTS["t800"].joints
    for name, angle in GUARD_POSE.items():
        out[:, 7 + joints.index(name)] = angle
    return out


def checks(rows: NDArray[np.float64]) -> tuple[list[str], list[str], list[str]]:
    """The K-1 coach's checks of a footwork reference: names, bars and this motion's values."""
    robot = ROBOTS["t800"]
    sole = contacts(rows, robot)[1]
    return (
        ["feet skating, of grounded frames", "both feet up, of frames", "frames with a joint over 20 rad/s",
         "frames with a foot over 20 cm up"],
        ["under 5%", "none", "none", "none"],
        [f"{skating(rows, robot):.0%}", f"{floating(rows, robot):.0%}", f"{(joint_speed(rows) > 20).sum()}",  # noqa: PLR2004
         f"{(sole > 0.2).any(0).sum()}"],  # noqa: PLR2004
    )  # fmt: skip


if __name__ == "__main__":
    out, chunks = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    m = t800.get_spec().compile()
    names = [m.body(b).name for b in range(1, m.nbody)]
    motion: list[NDArray[np.float64]] = []
    for npz in chunks:
        rows = np.loadtxt(npz.with_suffix(".csv"), delimiter=",")
        for s in segments(npz, names):
            seg = rows[s] if not motion else place(rows[s], motion[-1][-1])
            if motion:
                motion.append(blend(motion[-1][-1], seg[0]))
            motion.append(seg)
    rows = np.concatenate(motion)
    locked = hold_guard(lock_feet(rows, ROBOTS["t800"], steady=True))
    np.savetxt(out, locked, delimiter=",", fmt="%.6f")
    pieces, seconds = (len(motion) + 1) // 2, len(rows) / FPS
    print(f"{pieces} segments, {seconds:.0f} s ({(pieces - 1) * BLEND:.0f} s of blends) -> {out}")
    print("The K-1 coach's checks, before -> after holding the feet:")
    for name, bar, before, after in zip(*checks(rows), checks(locked)[2], strict=True):
        print(f"  {name}: {before} -> {after} (bar: {bar})")
