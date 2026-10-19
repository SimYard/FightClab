"""Helicopter kick by motion tracking: mjlab's BeyondMimic-style G1 tracking on our G1, with adaptive tolerances."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg, bad_motion_body_pos_z_only

from fightclab.robots.asimov import ASIMOV_ACTION_SCALE, get_asimov_cfg
from fightclab.robots.asimov import G1_NAMES as ASIMOV_NAMES
from fightclab.robots.g1 import get_g1_cfg
from fightclab.robots.h2 import H2_ACTION_SCALE, get_h2_cfg
from fightclab.robots.pm01 import G1_NAMES as PM01_NAMES
from fightclab.robots.pm01 import PM01_ACTION_SCALE, get_pm01_cfg
from fightclab.robots.r1 import R1_ACTION_SCALE, get_r1_cfg
from fightclab.robots.t1 import G1_NAMES as T1_NAMES
from fightclab.robots.t1 import T1_ACTION_SCALE, get_t1_cfg
from fightclab.robots.t800 import G1_NAMES as T800_NAMES
from fightclab.robots.t800 import T800_ACTION_SCALE, get_t800_cfg
from fightclab.sim import Rk4MujocoCfg
from fightclab.tasks.helicopter_kick import mdp as kick_mdp
from fightclab.tasks.helicopter_kick.env_cfg import FEET, feet_ground_contact
from fightclab.tasks.helicopter_kick.video_motion import ROBOTS, phases

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

# The kick from a Ginger Ninja tutorial clip, made by video_motion.py and motion_npz.py (see their docstrings).
# The keyframe version from reference.py is helicopter_kick.npz; it looks stiffer than the person.
MOTION = Path(__file__).with_name("helicopter_kick_video.npz")
MOTION_R1 = Path(__file__).with_name("helicopter_kick_video_r1.npz")
MOTION_H2 = Path(__file__).with_name("helicopter_kick_video_h2.npz")
MOTION_PM01 = Path(__file__).with_name("helicopter_kick_video_pm01.npz")
MOTION_T800 = Path(__file__).with_name("helicopter_kick_video_t800.npz")
MOTION_T1 = Path(__file__).with_name("helicopter_kick_video_t1.npz")
MOTION_ASIMOV = Path(__file__).with_name("helicopter_kick_video_asimov.npz")

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
    return ((t > takeoff) & (t < landing) & kick_mdp._airborne(env, sensor_name)).float()


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
    cfg.rewards["feet_height"] = RewardTermCfg(
        func=feet_height, weight=2.0, params={"command_name": "motion", "body_names": feet, "std": 0.1}
    )
    cfg.scene.sensors = (*cfg.scene.sensors, feet_ground_contact())
    # Up to 5 per step, as much as all the tracking terms together.
    cfg.rewards["flight"] = RewardTermCfg(
        func=flight, weight=5.0, params={"command_name": "motion", "sensor_name": FEET}
    )
    return cfg


def _on_robot(
    play: bool, entity: EntityCfg, scale: dict[str, float], motion_file: Path, names: dict[str, str] | None = None
) -> ManagerBasedRlEnvCfg:
    """G1's task on another robot: its entity, action scales and motion, and its ``names`` for G1's bodies.

    Foot geoms must be named like G1's (``left_foot1_collision``, ...); our robot modules do that.
    """
    names = names or {}

    def rename(body: str) -> str:
        return names.get(body, body)

    cfg = g1_helicopter_kick_tracking_env_cfg(play=play)
    cfg.scene.entities = {"robot": entity}
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = scale
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.motion_file = str(motion_file)
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
    return cfg


def r1_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The same task on R1. Its bodies are named like G1's, but its hands end at the wrist roll link."""
    hands = {f"{side}_wrist_yaw_link": f"{side}_wrist_roll_link" for side in ("left", "right")}
    return _on_robot(play, get_r1_cfg(), R1_ACTION_SCALE, MOTION_R1, hands)


def h2_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The same task on H2, whose bodies are named like G1's."""
    return _on_robot(play, get_h2_cfg(), H2_ACTION_SCALE, MOTION_H2)


def pm01_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The same task on EngineAI's PM01."""
    return _on_robot(play, get_pm01_cfg(), PM01_ACTION_SCALE, MOTION_PM01, PM01_NAMES)


def t800_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The same task on EngineAI's T800."""
    return _on_robot(play, get_t800_cfg(), T800_ACTION_SCALE, MOTION_T800, T800_NAMES)


def t1_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The same task on Booster's T1, whose reference flies shorter than G1's (see video_motion.ROBOTS)."""
    cfg = _on_robot(play, get_t1_cfg(), T1_ACTION_SCALE, MOTION_T1, T1_NAMES)
    cfg.rewards["flight"].params["window"] = phases(ROBOTS["t1"])
    return cfg


def asimov_helicopter_kick_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The same task on Asimov v1, whose reference flies shorter than G1's (see video_motion.ROBOTS)."""
    cfg = _on_robot(play, get_asimov_cfg(), ASIMOV_ACTION_SCALE, MOTION_ASIMOV, ASIMOV_NAMES)
    cfg.rewards["flight"].params["window"] = phases(ROBOTS["asimov"])
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
