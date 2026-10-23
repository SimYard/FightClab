"""How high can G1 jump with Unitree's motor curves? A measurement task, not a skill.

``Fightclab-MaxJump-G1``: from the boxing guard, jump once as high as possible and stay up. The reward is
the pelvis height gained while both feet are off the floor, paid as a new best is reached, so standing on
the toes earns nothing and one high jump beats many small ones. A fall ends the episode with a penalty.
``Metrics/jump/pelvis_rise_max`` and ``_mean`` are the answer; reference motions should not ask for more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

from fightclab.tasks.helicopter_kick import mdp as kick_mdp
from fightclab.tasks.helicopter_kick.env_cfg import FEET, g1_helicopter_kick_env_cfg
from fightclab.tasks.helicopter_kick.rl_cfg import g1_helicopter_kick_ppo_runner_cfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.rl import RslRlOnPolicyRunnerCfg


class AirborneHeightGain:
    """Pelvis height gained in the air over the episode's best so far, in m/s; logs the jump height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        """Track the start and best pelvis height per env."""
        self.robot: Entity = env.scene["robot"]
        self.start = torch.zeros(env.num_envs, device=env.device)
        self.best = torch.zeros(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Measure from the first step after the reset: here the body poses are still the old episode's."""
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
        """Gain over the best height while airborne."""
        z = self.robot.data.root_link_pos_w[:, 2]
        self.start = torch.where(self.fresh, z, self.start)
        self.best = torch.where(self.fresh, z, self.best)
        self.fresh[:] = False
        airborne = kick_mdp.in_air(env, sensor_name)
        gain = torch.where(airborne, (z - self.best).clamp(min=0.0), torch.zeros_like(z))
        self.best = torch.where(airborne, torch.maximum(self.best, z), self.best)
        rise = self.best - self.start
        env.extras["log"]["Metrics/jump/pelvis_rise_mean"] = rise.mean()
        env.extras["log"]["Metrics/jump/pelvis_rise_max"] = rise.max()
        return gain / env.step_dt


def g1_max_jump_env_cfg() -> ManagerBasedRlEnvCfg:
    """The helicopter kick's G1 scene and observations, with jump-height rewards and 2 s episodes."""
    cfg = g1_helicopter_kick_env_cfg()
    cfg.rewards = {
        "height": RewardTermCfg(func=AirborneHeightGain, weight=10.0, params={"sensor_name": FEET}),
        "upright": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-1.0),
        # Rewards are scaled by dt (0.02 s): -4 per fall, worth 40 cm of jump.
        "fell": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
        "joint_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    }
    cfg.episode_length_s = 2.0
    return cfg


def g1_max_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """The helicopter kick's PPO, a shorter run."""
    cfg = g1_helicopter_kick_ppo_runner_cfg()
    cfg.experiment_name = "g1_max_jump"
    cfg.max_iterations = 1_500
    return cfg


register_mjlab_task(
    task_id="Fightclab-MaxJump-G1",
    env_cfg=g1_max_jump_env_cfg(),
    play_env_cfg=g1_max_jump_env_cfg(),
    rl_cfg=g1_max_jump_ppo_runner_cfg(),
)
