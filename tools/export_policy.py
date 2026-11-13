"""Export a trained policy's actor to TorchScript: the actor's observations in, actions out.

    uv run python tools/export_policy.py TASK CHECKPOINT OUT.pt

The duel task's opponent runs one (``k1/duel.py``).
"""

import sys
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import fightclab.tasks  # noqa: F401  (registers the tasks)

task, checkpoint, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
env_cfg = load_env_cfg(task, play=True)
env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0"), clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
runner.load(checkpoint)
runner.export_policy_to_jit(str(out.parent), out.name)
print(f"{checkpoint} -> {out}")
