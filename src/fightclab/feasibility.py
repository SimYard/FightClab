"""Can G1 with Unitree's motor curves perform a motion? Inverse dynamics frame by frame.

For each frame of a motion CSV (base pos, base quat xyzw, 29 joints in Unitree order): the joint torques the
motion needs (mj_inverse with contacts off; the floating base's residual is the force the floor would have
to supply) against the torque available at that joint speed on Unitree's torque-speed curves
(fightclab.robots.g1). A ratio above 1 means the motor cannot do it.

Run: uv run python -m fightclab.feasibility <motion.csv> [fps]
"""

import re
import sys

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from numpy.typing import NDArray

from fightclab.robots.g1 import G1_MOTORS


def _model() -> tuple[mujoco.MjModel, list[str], NDArray[np.float64]]:
    """G1 with contacts off and our motors' armature; joint names; [J, 3] effort limit, saturation, speed limit."""
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    names = [m.joint(j).name for j in range(1, m.njnt)]
    limits = np.zeros((len(names), 3))
    for i, n in enumerate(names):
        cfg = next(c for c in G1_MOTORS if any(re.fullmatch(p, n) for p in c.target_names_expr))
        limits[i] = cfg.effort_limit, cfg.saturation_effort, cfg.velocity_limit
        m.dof_armature[6 + i] = cfg.armature
    return m, names, limits


def torque_ratios(rows: NDArray[np.float64], fps: float) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Per frame and joint, needed torque over available torque; per frame, the floor force needed in body weights."""
    m, names, limits = _model()
    d = mujoco.MjData(m)
    qpos = np.concatenate([rows[:, :3], rows[:, [6, 3, 4, 5]], rows[:, 7:]], axis=1)  # xyzw -> wxyz
    qvel = np.zeros((len(qpos), m.nv))
    for t in range(len(qpos)):
        a, b = max(t - 1, 0), min(t + 1, len(qpos) - 1)
        mujoco.mj_differentiatePos(m, qvel[t], (b - a) / fps, qpos[a], qpos[b])
    qacc = np.gradient(qvel, 1 / fps, axis=0)
    weight = m.body_mass.sum() * 9.81
    ratio, floor = np.zeros((len(qpos), len(names))), np.zeros(len(qpos))
    for t in range(len(qpos)):
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[t], qvel[t], qacc[t]
        mujoco.mj_inverse(m, d)
        tau, v = d.qfrc_inverse[6:], qvel[t, 6:]
        curve = np.clip(limits[:, 1] * (1 - np.abs(v) / limits[:, 2]), 0, None)
        available = np.where(tau * v > 0, np.minimum(limits[:, 0], curve), limits[:, 0])  # braking: full limit
        ratio[t] = np.abs(tau) / np.maximum(available, 1e-6)
        floor[t] = np.linalg.norm(d.qfrc_inverse[:3]) / weight
    return ratio, floor, names


def report(rows: NDArray[np.float64], fps: float, top: int = 6) -> float:
    """Print the worst joints and return the worst ratio."""
    ratio, floor, names = torque_ratios(rows, fps)
    for j in np.argsort(-ratio.max(axis=0))[:top]:
        t = int(np.argmax(ratio[:, j]))
        print(f"  {names[j]:28s} {ratio[t, j] * 100:5.0f}% of the motor at t={t / fps:.2f}s")
    over = (ratio > 1).any(axis=1)
    print(f"  frames over the motors: {over.sum()} of {len(rows)}; floor force up to {floor.max():.1f} body weights")
    return float(ratio.max())


if __name__ == "__main__":
    path, *fps = sys.argv[1:]
    report(np.loadtxt(path, delimiter=","), float(fps[0]) if fps else 30.0)
