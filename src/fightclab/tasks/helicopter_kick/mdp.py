"""Helicopter kick terms.

The kick, per Wikipedia's "360 Split Hook Kick": jump, turn about 180 degrees, throw both legs out
into a split in the air, finish the 360 and land. The rewards follow those phases:
guard -> spin -> split with a fast heel during the first jump -> land standing tall, facing the same way,
back in the guard; paid by the turns made in the air.
Air rewards stop after the first jump, otherwise the policy hops and spins like a top all episode.
Landing pays only for turns made in the air, otherwise a hop plus a pivot on the floor earns it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

TURN = 2 * math.pi


class KickProgress(CommandTerm):
    """Per-episode kick state.

    ``best`` is the largest angle turned so far in ``cfg.direction``, from integrated world yaw rate.
    ``air_best`` is the same, counting only rotation while in the air during the first jump.
    ``kick_best`` is the best kick in the first jump: 1 with the kicking leg straight out at pelvis height.
    ``flights`` counts takeoffs: moments both feet leave the ground.
    """

    cfg: KickProgressCfg

    def __init__(self, cfg: KickProgressCfg, env: ManagerBasedRlEnv) -> None:
        """Allocate the per-env accumulators."""
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.kick_site = [self.robot.site_names.index(cfg.kick_site)]
        self.yaw = torch.zeros(self.num_envs, device=self.device)
        self.best = torch.zeros(self.num_envs, device=self.device)
        self.air_yaw = torch.zeros(self.num_envs, device=self.device)
        self.air_best = torch.zeros(self.num_envs, device=self.device)
        self.kick_best = torch.zeros(self.num_envs, device=self.device)
        self.flights = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.was_airborne = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.metrics["turns"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["air_turns"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_lift"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["flights"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def turns(self) -> torch.Tensor:
        """Turns made so far. Twisting back does not take them away."""
        return self.best / TURN

    @property
    def air_turns(self) -> torch.Tensor:
        """Turns made in the air during the first jump."""
        return self.air_best / TURN

    @property
    def command(self) -> torch.Tensor:
        """Episode phase in [0, 1], turns made, air turns, kick lift and takeoffs so far (capped at 2)."""
        phase = self._env.episode_length_buf / self._env.max_episode_length
        flights = self.flights.clamp(max=2).float()
        return torch.stack([phase, self.turns, self.air_turns, self.kick_best, flights], dim=-1)

    def _update_metrics(self) -> None:
        self.metrics["turns"][:] = self.turns
        self.metrics["air_turns"][:] = self.air_turns
        self.metrics["kick_lift"][:] = self.kick_best
        self.metrics["flights"][:] = self.flights.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.yaw[env_ids] = 0.0
        self.best[env_ids] = 0.0
        self.air_yaw[env_ids] = 0.0
        self.air_best[env_ids] = 0.0
        self.kick_best[env_ids] = 0.0
        self.flights[env_ids] = 0
        # The robot spawns up to 2 cm above the floor. Count a takeoff only after it has touched down.
        self.was_airborne[env_ids] = True

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            step = self.cfg.direction * self.robot.data.root_link_ang_vel_w[:, 2] * self._env.step_dt
            self.yaw += step
            self.best = torch.maximum(self.best, self.yaw)
            airborne = in_air(self._env, self.cfg.sensor_name)
            self.flights += (airborne & ~self.was_airborne).long()
            self.was_airborne = airborne
            first_flight = airborne & (self.flights == 1)
            self.air_yaw += step * first_flight
            self.air_best = torch.maximum(self.air_best, self.air_yaw)
            # Height alone is met by tucking the knee; a kick also reaches out a leg's length from the pelvis.
            lift = _leg_lift(self.robot, self.kick_site, self.cfg.leg_length)[:, 0]
            foot_xy = self.robot.data.site_pos_w[:, self.kick_site[0], :2]
            reach = torch.norm(foot_xy - self.robot.data.root_link_pos_w[:, :2], dim=-1) / self.cfg.leg_length
            self.kick_best = torch.maximum(self.kick_best, lift * reach.clamp(max=1.0) * first_flight)


@dataclass(kw_only=True)
class KickProgressCfg(CommandTermCfg):
    """Config for KickProgress. Resampled only on reset."""

    sensor_name: str
    direction: float = -1.0
    """-1 turns right (clockwise seen from above), +1 turns left. Kick with the leg on the same side."""
    kick_site: str = "right_foot"
    leg_length: float = 0.7
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env: ManagerBasedRlEnv) -> KickProgress:
        """Build the term."""
        return KickProgress(self, env)


def _progress(env: ManagerBasedRlEnv) -> KickProgress:
    term = env.command_manager.get_term("kick")
    assert isinstance(term, KickProgress)
    return term


def _feet_down(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found > 0


def in_air(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """True where both feet are off the floor (the feet's contact sensor ``sensor_name``, see ``env_cfg.FEET``)."""
    return (~_feet_down(env, sensor_name)).all(dim=-1)


def _leg_lift(robot: Entity, site_ids: list[int] | slice, leg_length: float) -> torch.Tensor:
    """Per foot site [B, n]: 0 with the leg hanging below the pelvis, 1 with the foot at pelvis height."""
    feet_z = robot.data.site_pos_w[:, site_ids, 2]
    return ((feet_z - robot.data.root_link_pos_w[:, 2:3] + leg_length) / leg_length).clamp(0.0, 1.0)


def _first_flight(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """1 while in the air during the first jump of the episode."""
    return (in_air(env, sensor_name) & (_progress(env).flights <= 1)).float()


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Pelvis height above the floor."""
    robot: Entity = env.scene["robot"]
    return robot.data.root_link_pos_w[:, 2:3]


def spin(env: ManagerBasedRlEnv, sensor_name: str, ground_scale: float = 0.2, turns: float = 1.0) -> torch.Tensor:
    """New angle gained toward the full turn in the kick direction, in rad/s. Twisting back and forth earns nothing.

    Turning on the ground pays ``ground_scale`` of turning in the air.
    """
    robot: Entity = env.scene["robot"]
    progress = _progress(env)
    # The same step KickProgress applies right after the rewards are computed.
    yaw = progress.yaw + progress.cfg.direction * robot.data.root_link_ang_vel_w[:, 2] * env.step_dt
    best = torch.maximum(progress.best, yaw).clamp(max=turns * TURN)
    gain = (best - progress.best).clamp(min=0.0) / env.step_dt
    airborne = in_air(env, sensor_name).float()
    return gain * (ground_scale + (1.0 - ground_scale) * airborne)


def split(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    leg_length: float = 0.7,
    spread: float = 1.4,
) -> torch.Tensor:
    """Split in the first jump: both feet raised toward pelvis height and far apart.

    Lift is 0 with legs hanging and 1 with legs horizontal. The spread factor rules out tucking
    the knees, which also raises the feet.
    """
    robot: Entity = env.scene[asset_cfg.name]
    feet = robot.data.site_pos_w[:, asset_cfg.site_ids]  # [B, 2, 3]
    lift = _leg_lift(robot, asset_cfg.site_ids, leg_length).min(dim=-1).values
    apart = (torch.norm(feet[:, 0, :2] - feet[:, 1, :2], dim=-1) / spread).clamp(0.0, 1.0)
    return _first_flight(env, sensor_name) * lift * apart


def kick(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    leg_length: float = 0.7,
    max_speed: float = 10.0,
) -> torch.Tensor:
    """Strike in the first jump: horizontal speed of the raised kicking foot relative to the pelvis, in m/s.

    ``asset_cfg`` sites are the kicking foot (or feet; the faster one counts). Relative to the pelvis, so
    flying forward with a raised leg does not count as a kick.
    """
    robot: Entity = env.scene[asset_cfg.name]
    rel_vel = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2] - robot.data.root_link_lin_vel_w[:, None, :2]
    speed = torch.norm(rel_vel, dim=-1).clamp(max=max_speed)
    lift = _leg_lift(robot, asset_cfg.site_ids, leg_length)
    return _first_flight(env, sensor_name) * (lift * speed).max(dim=-1).values


def _after_kick(env: ManagerBasedRlEnv, sensor_name: str, air_turns: float) -> torch.Tensor:
    """Both feet down after the first jump, scaled by what the jump did.

    Turns made in the air count up to ``air_turns``, and the kicking foot must have got to pelvis height.
    Without the kick factor, a hop-turn into the guard pays as much as a real kick and the leg stays down.
    """
    progress = _progress(env)
    both_down = _feet_down(env, sensor_name).all(dim=-1).float()
    jumped = (progress.flights >= 1).float()
    return jumped * both_down * (progress.air_turns / air_turns).clamp(max=1.0) * progress.kick_best


def landed(
    env: ManagerBasedRlEnv, sensor_name: str, air_turns: float = 0.5, low: float = 0.4, tall: float = 0.7
) -> torch.Tensor:
    """Standing tall and upright after the kick.

    Pelvis height scales the reward from 0 at ``low`` to 1 at ``tall``, so a deep lunge earns little.
    """
    robot: Entity = env.scene["robot"]
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(min=0.0) ** 2
    height = ((robot.data.root_link_pos_w[:, 2] - low) / (tall - low)).clamp(0.0, 1.0)
    return _after_kick(env, sensor_name, air_turns) * upright * height


def facing(env: ManagerBasedRlEnv, sensor_name: str, air_turns: float = 0.5) -> torch.Tensor:
    """After the kick, facing where it faced before: 1 at exactly 360 degrees, 0 at no turn at all.

    Linear over the whole turn: narrower kernels gave no signal short of 180 degrees, and the policy
    never finished the turn.
    """
    error = (_progress(env).yaw - TURN).abs()
    return _after_kick(env, sensor_name, air_turns) * (1.0 - error / TURN).clamp(min=0.0)


def guard(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    hand_targets: tuple[tuple[float, float, float], ...],
    feet_targets: tuple[tuple[float, float], ...],
    *,
    std: float = 0.15,
    air_turns: float = 0.5,
) -> torch.Tensor:
    """Boxing stance after the kick: fists up by the chin, feet staggered.

    ``asset_cfg`` sites must be left palm, right palm, left foot, right foot, in that order, and its one
    body the torso. Hands are compared in the torso frame, feet in the pelvis heading frame (xy only).
    Hands and feet each give half of the reward, through a Gaussian kernel of their summed squared error.
    """
    robot: Entity = env.scene[asset_cfg.name]
    sites = robot.data.site_pos_w[:, asset_cfg.site_ids]  # [B, 4, 3]
    torso = asset_cfg.body_ids[0]
    torso_quat = robot.data.body_link_quat_w[:, torso, None].expand(-1, 2, -1)
    hands = quat_apply_inverse(torso_quat, sites[:, :2] - robot.data.body_link_pos_w[:, torso, None])
    heading = yaw_quat(robot.data.root_link_quat_w)[:, None].expand(-1, 2, -1)
    feet = quat_apply_inverse(heading, sites[:, 2:] - robot.data.root_link_pos_w[:, None])[..., :2]
    hand_err = ((hands - torch.tensor(hand_targets, device=env.device)) ** 2).sum(dim=(-2, -1))
    feet_err = ((feet - torch.tensor(feet_targets, device=env.device)) ** 2).sum(dim=(-2, -1))
    pose = 0.5 * torch.exp(-hand_err / std**2) + 0.5 * torch.exp(-feet_err / std**2)
    return _after_kick(env, sensor_name, air_turns) * pose
