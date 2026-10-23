"""Terms for mjlab's motion tracking task (BeyondMimic-style), shared by FightClab's tracking tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg


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


def feet_height(env: ManagerBasedRlEnv, command_name: str, body_names: tuple[str, ...], std: float) -> torch.Tensor:
    """exp(-(reference height - foot height)^2 / std^2), averaged over the feet.

    mjlab's body tracking averages its error over 14 bodies, so a kicking foot 10-15 cm short of the reference costs
    almost nothing: policies that landed every kick kicked 8-31 cm low, and more training lowered R1's kick further.
    """
    motion = env.command_manager.get_term(command_name)
    assert isinstance(motion, MotionCommand)
    feet = [motion.cfg.body_names.index(name) for name in body_names]
    error = motion.body_pos_relative_w[:, feet, 2] - motion.robot_body_pos_w[:, feet, 2]
    return torch.exp(-(error**2) / std**2).mean(dim=-1)


def feet_contact_slip(env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared floor speed of the feet touching the floor, summed.

    Tracking pays for where a foot is, not how it got there: T800's footwork slid its following foot up to 9 cm per
    step along the floor before lifting it (the reference: 0.2 cm). A heel rising on the ball barely moves the foot.
    """
    feet = env.scene[asset_cfg.name].data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    touching = env.scene[sensor_name].data.found > 0
    return (feet.square().sum(dim=-1) * touching).sum(dim=-1)


def bad_anchor_heading(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    """True when the anchor body faces more than ``threshold`` radians off the reference's heading.

    mjlab's anchor terminations check its height and tilt, not where it faces: Asimov landed every kick by turning
    70 degrees and back instead of 360.
    """
    motion = env.command_manager.get_term(command_name)
    assert isinstance(motion, MotionCommand)

    def heading(q: torch.Tensor) -> torch.Tensor:  # of the body's x axis, projected on the floor; q is (w, x, y, z)
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))

    error = heading(motion.anchor_quat_w) - heading(motion.robot_anchor_quat_w)
    return torch.atan2(torch.sin(error), torch.cos(error)).abs() > threshold
