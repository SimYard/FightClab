"""Helicopter kick by motion tracking: mjlab's BeyondMimic-style G1 tracking on our G1, with adaptive tolerances."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

from fightclab.robots.g1 import get_g1_cfg
from fightclab.sim import Rk4MujocoCfg
from fightclab.tasks.helicopter_kick import mdp as kick_mdp
from fightclab.tasks.helicopter_kick.env_cfg import FEET, feet_ground_contact
from fightclab.tasks.helicopter_kick.video_motion import phases

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

# The kick from a Ginger Ninja tutorial clip, made by video_motion.py (see its docstring). The keyframe
# version from reference.py is helicopter_kick.npz; it looks stiffer than the person.
MOTION = Path(__file__).with_name("helicopter_kick_video.npz")

# mjlab's tracking rewards, all exp(-error / std**2), that get adaptive tolerances.
TRACKING_REWARDS = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)


class AdaptiveExp:
    """KungfuBot's adaptive tracking factor (PBHC, arXiv 2506.12851) around an mjlab tracking reward.

    The tolerance sigma takes the place of std**2. It starts ``loose`` times wider than mjlab's std and only
    shrinks, to the moving average of the error: sigma <- min(sigma, EMA(error)). With mjlab's fixed std the
    takeoff and flight errors were so large that their reward was almost zero and the policy never got
    past takeoff; a wide tolerance first rewards rough tracking there, and tightens as the policy improves.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        """Start sigma ``loose`` times wider than mjlab's std."""
        self.sigma = torch.tensor((cfg.params["std"] * cfg.params["loose"]) ** 2, device=env.device)
        # The EMA starts at sigma, not at the first error: right after a reset the robot sits exactly on the
        # reference, the error is ~0, and sigma, which never grows back, would collapse on the first step.
        self.ema = self.sigma.clone()
        self.log_key = f"Metrics/adaptive_sigma/{cfg.params['func'].__name__}"

    def __call__(
        self, env: ManagerBasedRlEnv, func: Any, std: float, loose: float, alpha: float, **kwargs: Any
    ) -> torch.Tensor:
        """Reward with the current sigma; the EMA moves by ``alpha`` per env step."""
        del std, loose
        error = -torch.log(func(env, std=1.0, **kwargs).clamp(min=1e-30))  # mjlab's error, recovered
        self.ema = (1 - alpha) * self.ema + alpha * error.mean()
        self.sigma = torch.minimum(self.sigma, self.ema).clamp(min=1e-6)
        env.extras["log"][self.log_key] = self.sigma
        return torch.exp(-error / self.sigma)


def flight(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
    """1 while both feet are off the floor during the reference's flight.

    Tracking pays little for pushing harder until the robot is actually up, and the anchor limit ends the
    episode before that: policies followed the crouch and the spin-up, then stayed down. A max-jump policy
    paid for being in the air learned a high jump on the same motors.
    """
    motion = env.command_manager.get_term(command_name)
    assert isinstance(motion, MotionCommand)
    t = motion.time_steps * env.step_dt
    takeoff, landing = phases()
    return ((t > takeoff) & (t < landing) & kick_mdp._airborne(env, sensor_name)).float()


def g1_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Mjlab's G1 tracking task on the video kick, with adaptive tracking tolerances.

    Our G1: Unitree motor curves and stiff contacts.
    """
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    cfg.scene.entities = {"robot": get_g1_cfg()}
    cfg.sim.mujoco = Rk4MujocoCfg(**vars(cfg.sim.mujoco))
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.motion_file = str(MOTION)
    for name in TRACKING_REWARDS:
        term = cfg.rewards[name]
        term.params = {**term.params, "func": term.func, "loose": 3.0, "alpha": 1e-3}
        term.func = AdaptiveExp
    # The reference kicks head-high, past what G1's motors can follow (see video_motion.py). With mjlab's
    # 25 cm limit on hands and feet an episode ended at every takeoff; 50 cm lets the kick fall short.
    # The torso keeps a tight 20 cm: at 40 cm the policy learned to skip the 37 cm jump and pivot on the
    # standing leg, which is not the kick. 20 cm forces a jump of at least ~17 cm.
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.5
    cfg.terminations["anchor_pos"].params["threshold"] = 0.2
    cfg.scene.sensors = (*cfg.scene.sensors, feet_ground_contact())
    # Up to 5 per step, as much as all the tracking terms together.
    cfg.rewards["flight"] = RewardTermCfg(
        func=flight, weight=5.0, params={"command_name": "motion", "sensor_name": FEET}
    )
    return cfg


def g1_helicopter_kick_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Mjlab's G1 tracking PPO, shorter run for one 2 s clip."""
    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.experiment_name = "g1_helicopter_kick_tracking"
    cfg.max_iterations = 5_000
    return cfg
