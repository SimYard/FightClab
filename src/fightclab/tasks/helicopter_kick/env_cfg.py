"""G1 helicopter kick environment."""

import math

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.tasks.velocity.mdp import foot_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from fightclab.robots.g1 import GUARD_FEET, GUARD_HANDS, get_g1_cfg
from fightclab.sim import Rk4MujocoCfg
from fightclab.tasks.helicopter_kick import mdp

FEET = "feet_ground_contact"


def feet_ground_contact() -> ContactSensorCfg:
    """Each foot's contact with the floor."""
    return ContactSensorCfg(
        name=FEET,
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )


def g1_helicopter_kick_env_cfg() -> ManagerBasedRlEnvCfg:
    """One kick per 3 s episode: from the boxing guard, jump, spin 360 with a split in the air, land in the guard."""
    feet_ground = feet_ground_contact()

    # Sim only for now, so the actor sees the same clean state as the critic.
    obs_terms = {
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "base_height": ObservationTermCfg(func=mdp.base_height),
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "kick": ObservationTermCfg(func=envs_mdp.generated_commands, params={"command_name": "kick"}),
        "feet_contact": ObservationTermCfg(func=foot_contact, params={"sensor_name": FEET}),
    }

    feet_sites = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    rewards = {
        "spin": RewardTermCfg(func=mdp.spin, weight=1.0, params={"sensor_name": FEET, "ground_scale": 0.05}),
        "split": RewardTermCfg(func=mdp.split, weight=3.0, params={"sensor_name": FEET, "asset_cfg": feet_sites}),
        # Right leg, matching the right turn set by KickProgressCfg.direction.
        "kick": RewardTermCfg(
            func=mdp.kick,
            weight=1.0,
            params={"sensor_name": FEET, "asset_cfg": SceneEntityCfg("robot", site_names=("right_foot",))},
        ),
        "landed": RewardTermCfg(func=mdp.landed, weight=2.0, params={"sensor_name": FEET}),
        "guard": RewardTermCfg(
            func=mdp.guard,
            weight=2.0,
            params={
                "sensor_name": FEET,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    site_names=("left_palm", "right_palm", "left_foot", "right_foot"),
                    body_names=("torso_link",),
                    preserve_order=True,
                ),
                "hand_targets": GUARD_HANDS,
                "feet_targets": GUARD_FEET,
            },
        ),
        # A full 360: face the way it faced before the kick.
        "facing": RewardTermCfg(func=mdp.facing, weight=2.0, params={"sensor_name": FEET}),
        # Rewards are scaled by dt (0.02 s), so this is -10 per fall: more than the air rewards pay.
        "fell": RewardTermCfg(func=envs_mdp.is_terminated, weight=-500.0),
        "joint_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_cfg()},
            sensors=(feet_ground,),
            extent=2.0,
        ),
        observations={
            "actor": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True, enable_corruption=False),
            "critic": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True, enable_corruption=False),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True
            )
        },
        commands={"kick": mdp.KickProgressCfg(sensor_name=FEET)},
        events={
            "reset_base": EventTermCfg(
                func=envs_mdp.reset_root_state_uniform,
                mode="reset",
                params={"pose_range": {"z": (0.0, 0.02), "yaw": (-math.pi, math.pi)}, "velocity_range": {}},
            ),
            "reset_joints": EventTermCfg(
                func=envs_mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.05, 0.05),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                },
            ),
        },
        rewards=rewards,
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
            "too_low": TerminationTermCfg(func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.4}),
            # Hard spins can blow up the physics; reset that env instead of crashing the run.
            "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            njmax=300,
            contact_sensor_maxmatch=64,
            mujoco=Rk4MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=3.0,
    )
