"""Helicopter kick by motion tracking: mjlab's BeyondMimic-style G1 tracking on our robots, with adaptive tolerances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg, bad_motion_body_pos_z_only

from fightclab.robots import asimov, h2, pm01, r1, t1, t800
from fightclab.robots.g1 import get_g1_cfg
from fightclab.sim import Rk4MujocoCfg
from fightclab.tasks.helicopter_kick import mdp as kick_mdp
from fightclab.tasks.helicopter_kick.env_cfg import FEET, feet_ground_contact
from fightclab.tasks.helicopter_kick.video_motion import phases
from fightclab.tracking import AdaptiveExp, bad_anchor_heading, feet_height

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

# Our other robots, for G1's task: the entity, its action scales, and its names for the G1 bodies mjlab's task uses.
ROBOT_CFGS: dict[str, tuple[Callable[[], EntityCfg], dict[str, float], dict[str, str]]] = {
    # R1's bodies are named like G1's, but its hands end at the wrist roll link.
    "r1": (
        r1.get_r1_cfg,
        r1.R1_ACTION_SCALE,
        {f"{s}_wrist_yaw_link": f"{s}_wrist_roll_link" for s in ("left", "right")},
    ),
    "h2": (h2.get_h2_cfg, h2.H2_ACTION_SCALE, {}),
    "pm01": (pm01.get_pm01_cfg, pm01.PM01_ACTION_SCALE, pm01.G1_NAMES),
    "t800": (t800.get_t800_cfg, t800.T800_ACTION_SCALE, t800.G1_NAMES),
    "t1": (t1.get_t1_cfg, t1.T1_ACTION_SCALE, t1.G1_NAMES),
    "asimov": (asimov.get_asimov_cfg, asimov.ASIMOV_ACTION_SCALE, asimov.G1_NAMES),
}

# mjlab's tracking rewards, all exp(-error / std**2), that get adaptive tolerances.
TRACKING_REWARDS = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)


def motion_file(robot: str) -> Path:
    """The kick from a Ginger Ninja tutorial clip on ``robot``, made by video_motion.py and motion_npz.py."""
    return Path(__file__).with_name(f"helicopter_kick_video_{robot}.npz")


def flight(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str, window: tuple[float, float] | None = None
) -> torch.Tensor:
    """1 while both feet are off the floor during the reference's flight.

    Tracking pays little for pushing harder until the robot is actually up, and the anchor limit ends the
    episode before that: policies followed the crouch and the spin-up, then stayed down. A max-jump policy
    paid for being in the air learned a high jump on the same motors.
    """
    motion = env.command_manager.get_term(command_name)
    assert isinstance(motion, MotionCommand)
    t = motion.time_steps * env.step_dt
    takeoff, landing = window or phases()  # the robot's reference flight, G1's by default
    return ((t > takeoff) & (t < landing) & kick_mdp.in_air(env, sensor_name)).float()


def g1_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Mjlab's G1 tracking task on the video kick, with adaptive tracking tolerances.

    Our G1: Unitree motor curves and stiff contacts.
    """
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    cfg.scene.entities = {"robot": get_g1_cfg()}
    cfg.sim.mujoco = Rk4MujocoCfg(**vars(cfg.sim.mujoco))
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.motion_file = str(motion_file("g1"))
    if not play:
        # mjlab's adaptive sampling has one bin per second (three here) and weighs them by failures. Once only
        # the takeoff failed, ~98% of starts went to 0.83-1.67 s, the run-up into the crouch went unpractised,
        # and kicks played from the start fell from 82% complete (iteration 2500) to 24% (2750).
        motion.sampling_mode = "uniform"
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
    # The feet keep mjlab's 25 cm, and their height gets its own reward (see feet_height).
    feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    cfg.terminations["feet_pos"] = TerminationTermCfg(
        func=bad_motion_body_pos_z_only, params={"command_name": "motion", "threshold": 0.25, "body_names": feet}
    )
    # Facing within 60 degrees of the reference: the spin is the kick.
    cfg.terminations["anchor_heading"] = TerminationTermCfg(
        func=bad_anchor_heading, params={"command_name": "motion", "threshold": 1.05}
    )
    cfg.rewards["feet_height"] = RewardTermCfg(
        func=feet_height, weight=2.0, params={"command_name": "motion", "body_names": feet, "std": 0.1}
    )
    cfg.scene.sensors = (*cfg.scene.sensors, feet_ground_contact())
    # Up to 5 per step, as much as all the tracking terms together.
    cfg.rewards["flight"] = RewardTermCfg(
        func=flight, weight=5.0, params={"command_name": "motion", "sensor_name": FEET}
    )
    return cfg


def helicopter_kick_tracking_env_cfg(robot: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """G1's task on another of our robots (``ROBOT_CFGS``), on its own motion and flight (``video_motion.JUMPS``).

    Foot geoms must be named like G1's (``left_foot1_collision``, ...); our robot modules do that.
    """
    entity, scale, names = ROBOT_CFGS[robot]

    def rename(body: str) -> str:
        return names.get(body, body)

    cfg = g1_helicopter_kick_tracking_env_cfg(play=play)
    cfg.scene.entities = {"robot": entity()}
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = scale
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.motion_file = str(motion_file(robot))
    motion.anchor_body_name = rename(motion.anchor_body_name)
    motion.body_names = tuple(map(rename, motion.body_names))
    for limb in ("ee_body_pos", "feet_pos"):
        params = cfg.terminations[limb].params
        params["body_names"] = tuple(map(rename, params["body_names"]))
    feet = cfg.rewards["feet_height"].params
    feet["body_names"] = tuple(map(rename, feet["body_names"]))
    com = cfg.events["base_com"].params["asset_cfg"]
    com.body_names = tuple(map(rename, com.body_names))
    cfg.viewer.body_name = rename(cfg.viewer.body_name or "")
    whole = ContactMatch(mode="subtree", pattern=rename("pelvis"), entity="robot")
    cfg.scene.sensors = (
        *(replace(s, primary=whole, secondary=whole) for s in cfg.scene.sensors if s.name == "self_collision"),
        feet_ground_contact(rename("left_ankle_roll_link"), rename("right_ankle_roll_link")),
    )
    cfg.rewards["flight"].params["window"] = phases(robot)
    return cfg


def g1_helicopter_kick_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Mjlab's G1 tracking PPO, shorter run for one 2 s clip."""
    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.experiment_name = "g1_helicopter_kick_tracking"
    cfg.max_iterations = 5_000
    return cfg


def ppo_runner_cfg(robot: str) -> RslRlOnPolicyRunnerCfg:
    """G1's PPO for another robot, as long as G1's run that reached 256/256 kicks."""
    cfg = g1_helicopter_kick_tracking_ppo_runner_cfg()
    cfg.experiment_name = f"{robot}_helicopter_kick_tracking"
    cfg.max_iterations = 3_500
    return cfg
