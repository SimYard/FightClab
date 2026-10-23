"""Stress a tracking policy: its own motion under weaker motors, another floor and pushes; falls and error per case.

    uv run python tools/stress.py TASK CHECKPOINT [--envs 256] [--seconds 20]

Each case runs ``--envs`` copies for ``--seconds`` from random points of the task's motion (the play config, so no
training randomization beyond its own). A fall is any episode end that is not a time-out. Motor strength is the
damage model's lever (hits lower the torque limits of the joints they land on), so where a skill breaks tells what a
given loss of torque costs; friction and pushes are what a ring and an opponent add.
"""

import argparse
from dataclasses import asdict, replace

import mjlab.tasks  # noqa: F401  (registers the tasks)
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def motors(cfg: ManagerBasedRlEnvCfg, strength: float) -> None:
    """Every motor's torque limits scaled by ``strength``."""
    robot = cfg.scene.entities["robot"]
    weak = tuple(
        replace(a, effort_limit=a.effort_limit * strength, saturation_effort=a.saturation_effort * strength)
        for a in robot.articulation.actuators
    )
    cfg.scene.entities["robot"] = replace(robot, articulation=replace(robot.articulation, actuators=weak))


def friction(cfg: ManagerBasedRlEnvCfg, mu: float) -> None:
    """The feet's friction fixed at ``mu`` (the play config draws it from 0.3-1.2)."""
    cfg.events["foot_friction"].params["ranges"] = (mu, mu)


def pushes(cfg: ManagerBasedRlEnvCfg, speed: float) -> None:
    """A shove every 1-3 s: the pelvis' velocity changed by up to ``speed`` m/s sideways and forward."""
    cfg.events["push"] = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-speed, speed), "y": (-speed, speed)}},
    )


CASES = {
    "nominal": lambda cfg: None,
    "motors 80%": lambda cfg: motors(cfg, 0.8),
    "motors 60%": lambda cfg: motors(cfg, 0.6),
    "motors 40%": lambda cfg: motors(cfg, 0.4),
    "friction 0.3": lambda cfg: friction(cfg, 0.3),
    "friction 1.2": lambda cfg: friction(cfg, 1.2),
    "pushes 0.3 m/s": lambda cfg: pushes(cfg, 0.3),
    "pushes 0.6 m/s": lambda cfg: pushes(cfg, 0.6),
}


def run(task: str, checkpoint: str, case: str, envs: int, seconds: float) -> tuple[float, float, float]:
    """Falls per env-minute, mean body position error (m) and the share of envs that never fell, for one case."""
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = envs
    cfg.commands["motion"].sampling_mode = "uniform"
    CASES[case](cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env)
    runner = load_runner_cls(task)(venv, asdict(load_rl_cfg(task)), device="cuda:0")
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    motion = env.command_manager.get_term("motion")
    obs = venv.get_observations()
    falls = torch.zeros(envs, device="cuda:0")
    error, steps = 0.0, round(seconds / env.step_dt)
    for _ in range(steps):
        with torch.no_grad():
            obs, _, dones, extras = venv.step(policy(obs))
        falls += dones.float() * (1.0 - extras["time_outs"].float())
        error += float(motion.metrics["error_body_pos"].mean())
    env.close()
    return float(falls.sum()) / (envs * seconds / 60), error / steps, float((falls == 0).float().mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("checkpoint")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    print(f"{'case':16s} {'falls/min':>9s} {'error cm':>8s} {'no fall':>8s}")
    for case in CASES:
        rate, err, clean = run(args.task, args.checkpoint, case, args.envs, args.seconds)
        print(f"{case:16s} {rate:9.2f} {100 * err:8.1f} {100 * clean:7.0f}%")
