"""Smoke checks without a GPU: every task's configs build, every robot compiles, the motor curve holds.

uv run python tests/test_smoke.py      (the pre-push hook)
"""

import torch
from mjlab.actuator.dc_actuator import dc_motor_clip
from mjlab.tasks.registry import list_tasks, load_env_cfg

import fightclab.tasks  # noqa: F401  (registering a task builds its configs)
from fightclab.robots.unitree import N7520_22, unitree_motor


def test_robots_compile() -> None:
    """Each task's robot model compiles.

    A robot built on derived data that isn't here (a fresh clone, another machine) is named and skipped.
    """
    tasks = [task for task in list_tasks() if task.startswith("Fightclab-")]
    assert tasks, "no FightClab tasks registered"
    for spec_fn in {load_env_cfg(task).scene.entities["robot"].spec_fn for task in tasks}:
        try:
            spec = spec_fn()
        except FileNotFoundError as missing:
            print(f"skipped a robot built on derived data: {missing.filename}")
            continue
        spec.compile()


def test_unitree_curve() -> None:
    """Mjlab's DC curve passes through Unitree's points: y1 at rest and at the knee x1, zero at no-load x2."""
    m = unitree_motor(("knee",), N7520_22, armature=0.0)
    vel = torch.tensor([0.0, N7520_22.x1, (N7520_22.x1 + N7520_22.x2) / 2, N7520_22.x2])
    ones = torch.ones_like(vel)
    limit = dc_motor_clip(ones * 1e6, ones * m.saturation_effort, ones * m.velocity_limit, ones * m.effort_limit, vel)
    assert torch.allclose(limit, torch.tensor([111.0, 111.0, 55.5, 0.0]), atol=0.01), limit


if __name__ == "__main__":
    test_unitree_curve()
    test_robots_compile()
    print("smoke: ok")
