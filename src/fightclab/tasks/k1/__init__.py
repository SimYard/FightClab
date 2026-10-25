"""K-1 kickboxing skills tracked from tutorial videos.

Tasks:
- ``Fightclab-K1-Footwork-T800``: T800 moves on velocity commands in a K-1 stance (``footwork.py``), styled by AMP.
  The K-1 coach failed its footwork on every command (it leaned instead of stepping, squared up and hopped going
  back, lunged and skidded sideways), so the footwork is now tracked instead:
- ``Fightclab-K1-Footwork-Tracking-T800``: T800 tracks generated footwork and jabs (``steps.py``, ``strikes.py``).
- ``Fightclab-K1-Makiwara-T800``: the same with a makiwara in front of every jab, to hit, not push (``makiwara.py``).
- ``Fightclab-K1-Tracking-T800``: T800 tracks ``MOTION_T800``, the K-1 clip made by GVHMR, ``tools/gvhmr_to_csv.py``,
  ``video_motion ... --ground`` and ``motion_npz ... <out.npz>``. The kick task's setup without its flight reward:
  these motions stay on the floor.
"""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.mdp import motion_relative_body_orientation_error_exp
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from fightclab.tasks.helicopter_kick.env_cfg import FEET
from fightclab.tasks.helicopter_kick.tracking import helicopter_kick_tracking_env_cfg, ppo_runner_cfg
from fightclab.tasks.k1 import makiwara
from fightclab.tasks.k1.footwork import still_head_action_scale, t800_footwork_env_cfg, t800_footwork_ppo_runner_cfg
from fightclab.tracking import (
    BodiesTravel,
    bodies_pos,
    body_ang_vel_error_xy,
    feet_contact_slip,
    feet_height,
    joint_vel_error_l2,
)

MOTION_T800 = Path(__file__).parent / "k1_t800.npz"
# Generated K-1 step-drags and jabs on random commands (``steps ... random:600``, then ``motion_npz``), on the stance
# of the footwork clips joined by ``footwork_motion``: the clips themselves brought lunges, rocking and hanging feet.
# Not in the repo (56 MB): ``python -m fightclab.tasks.k1.steps ~/fw_clips/steps_t800.csv random:600``, then
# ``motion_npz t800`` it to this path.
MOTION_FOOTWORK_T800 = Path.home() / "fw_clips" / "steps_t800.npz"


def t800_k1_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """T800 on the K-1 clip."""
    cfg = helicopter_kick_tracking_env_cfg("t800", play)
    cfg.commands["motion"].motion_file = str(MOTION_T800)
    del cfg.rewards["flight"]
    return cfg


def t800_k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """The kick's PPO, logged as K-1."""
    cfg = ppo_runner_cfg("t800")
    cfg.experiment_name = "t800_k1_tracking"
    return cfg


def t800_k1_footwork_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """T800 on generated footwork and jabs, tracked frame by frame (the start of the skill library, see docs/duel.md).

    No low kicks: low-kick defense needs an opponent who throws them, not pushes from nowhere (docs/duel.md).
    """
    cfg = t800_k1_tracking_env_cfg(play)
    cfg.commands["motion"].motion_file = str(MOTION_FOOTWORK_T800)
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = still_head_action_scale()  # the reference's head never moves; the policy turned it up to 75 deg
    # Footwork without rocking: the tracked policy still swayed its pelvis and torso about 1 deg at 1-2 Hz and moved
    # its arms 0.12 rad/s, against a steadier reference and a still guard.
    # (Against the reference's own roll and pitch rates: a jab leans.)
    for body in ("LINK_BASE", "LINK_WAIST_YAW"):
        cfg.rewards[f"rock_{body}"] = RewardTermCfg(
            func=body_ang_vel_error_xy, weight=-1.0, params={"command_name": "motion", "body_names": (body,)}
        )
    # The kick's tracking widths let footwork lag and drag: 20 cm behind the reference still earned 64% of the root
    # position reward, and a sole on the floor instead of 4 cm up 85% of the feet's height reward. The robot made
    # 0.12-0.22 m/s of the reference's 0.25 and slid its following foot in 16% of grounded frames (the reference: 3%).
    # The hips turn with the fist: at mjlab's 0.5 the trained jab turned the pelvis 6-11 deg of 13, late (the coach).
    cfg.rewards["motion_global_root_ori"].weight = 1.5
    # Floors under the adaptive tolerances (``AdaptiveExp``): footwork, 84% of the frames, shrank them to its own
    # precision (torso 1.6 deg, bodies 2.6 cm), and the jabs' larger errors earned nothing (the RL review).
    for name, floor in (("motion_global_root_ori", 0.15), ("motion_body_pos", 0.05), ("motion_body_ori", 0.1)):
        cfg.rewards[name].params["floor"] = floor
    root = cfg.rewards["motion_global_root_pos"]
    root.weight, root.params["std"] = 2.0, 0.1
    cfg.rewards["feet_height"].params["std"] = 0.03
    # Arms and torso move as the reference does: still in footwork, fast in a jab.
    cfg.rewards["upper_body_vel"] = RewardTermCfg(
        func=joint_vel_error_l2,
        weight=-0.05,
        params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=(r"J(1[2-9]|2[0-4])_.*",))},
    )
    # The jab's fist, and the rear glove at the face (``bodies_pos``). At 0.1 m a fist 4 cm off the line, the shoulder
    # 3 deg and the elbow 5-8 deg too wide, still earned 0.86-0.94 (the RL review).
    cfg.rewards["fists"] = RewardTermCfg(
        func=bodies_pos,
        weight=3.0,
        params={"command_name": "motion", "body_names": ("LINK_WRIST_END_L", "LINK_WRIST_END_R"), "std": 0.05},
    )
    # The catch's rear arm from afar: 10 cm off with the lead fist on target, the fists' term paid 13%, and the trained
    # glove stayed out by the cheek with the elbow wide. The elbow too, or a flat forearm with the elbow up and out
    # would earn it (the K-1 coach). As much as the fists.
    cfg.rewards["rear_arm"] = RewardTermCfg(
        func=bodies_pos,
        weight=3.0,
        params={"command_name": "motion", "body_names": ("LINK_WRIST_END_R", "LINK_ELBOW_PITCH_R"), "std": 0.15},
    )
    # The torso over the stance: the jab's arm pushed the rest of the body back, the pelvis 0.008 H back where the
    # reference's goes 0.022 H in (the step jab: 0.012 of 0.063). Body and fist tracking, re-anchored on the robot's
    # torso, pay nothing for it but through the ankles, and the anchor's world position drifts 9 cm by then. The ankles
    # in the torso's frame are exactly that, without the drift (a pelvis velocity term at 0.15 m/s paid for stillness).
    cfg.rewards["stance"] = RewardTermCfg(
        func=bodies_pos,
        weight=2.0,
        params={"command_name": "motion", "body_names": ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"), "std": 0.05},
    )
    # The rear foot on its ball: in the step jab the robot's rear heel dropped flat (8 to -1 deg up, the reference's 10
    # to 19) and slid back while the fist flew, and the body arrived 0.06-0.09 s after it (the K-1 coach). One foot of
    # 14 bodies in mjlab's orientation tracking costs next to nothing.
    cfg.rewards["rear_foot"] = RewardTermCfg(
        func=motion_relative_body_orientation_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.15, "body_names": ("LINK_ANKLE_ROLL_R",)},
    )
    # The pelvis and feet travel as the reference's over 0.3 s (``BodiesTravel``): the step jab's 4.9 cm off earned
    # 0.41 at this width, footwork 0.58-0.72, standing jabs 0.74-0.79 (on the trained robot's rollout).
    cfg.rewards["travel"] = RewardTermCfg(
        func=BodiesTravel,
        weight=2.0,
        params={
            "command_name": "motion",
            "body_names": ("LINK_BASE", "LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"),
            "std": 0.04,
            "steps": 15,
        },
    )
    # The K-1 coach failed its back step: the lead foot slid back 8.8 cm per step flat on the floor before lifting
    # (the reference: 0.2 cm), and a foot dragged like that can't lift to check a low kick. At -1 it cost 3% of the
    # feet's height reward and barely fell in 600 iterations.
    cfg.rewards["feet_slip"] = RewardTermCfg(
        func=feet_contact_slip,
        weight=-5.0,
        params={
            "sensor_name": FEET,
            "asset_cfg": SceneEntityCfg("robot", body_names=cfg.rewards["feet_height"].params["body_names"]),
        },
    )
    # Rubber soles on a dry ring or floor. On mjlab's 0.3-1.2 the robot, which can't tell the floor, never asked its
    # feet for more than 0.3, and its centre of mass stayed 2 cm behind the jab and 5 cm behind the step jab, as far as
    # 0.3 allows (the RL review); the generator keeps every move within 0.5.
    cfg.events["foot_friction"].params["ranges"] = (0.5, 1.2)
    return cfg


def t800_k1_makiwara_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The footwork and jabs with a makiwara in front of every strike (``makiwara.py``)."""
    cfg = t800_k1_footwork_tracking_env_cfg(play)
    cfg.scene.entities[makiwara.ENTITY] = makiwara.makiwara_entity()
    cfg.scene.sensors = (*cfg.scene.sensors, makiwara.makiwara_hits())
    cfg.rewards["fists"].func = makiwara.fists_on_makiwara
    # One 15 N s hit earns what a third of a second of the fists' tracking does.
    cfg.rewards["makiwara"] = RewardTermCfg(func=makiwara.Makiwara, weight=50.0, params={"command_name": "motion"})
    # The body jab's drop: the pelvis is 1 of 14 bodies in the body tracking, the root position's tolerance had grown
    # to 17 cm, and the high fist went untracked on the pad; trained with the makiwara the drop shrank to 0.03 H of
    # 0.045 (the RL review).
    cfg.rewards["pelvis_height"] = RewardTermCfg(
        func=feet_height, weight=2.0, params={"command_name": "motion", "body_names": ("LINK_BASE",), "std": 0.03}
    )
    return cfg


def t800_k1_footwork_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """The kick's PPO, logged as K-1 footwork tracking."""
    cfg = ppo_runner_cfg("t800")
    cfg.experiment_name = "t800_k1_footwork_tracking"
    return cfg


register_mjlab_task(
    task_id="Fightclab-K1-Footwork-T800",
    env_cfg=t800_footwork_env_cfg(),
    play_env_cfg=t800_footwork_env_cfg(play=True),
    rl_cfg=t800_footwork_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Fightclab-K1-Footwork-Tracking-T800",
    env_cfg=t800_k1_footwork_tracking_env_cfg(),
    play_env_cfg=t800_k1_footwork_tracking_env_cfg(play=True),
    rl_cfg=t800_k1_footwork_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Fightclab-K1-Makiwara-T800",
    env_cfg=t800_k1_makiwara_env_cfg(),
    play_env_cfg=t800_k1_makiwara_env_cfg(play=True),
    rl_cfg=t800_k1_footwork_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Fightclab-K1-Tracking-T800",
    env_cfg=t800_k1_tracking_env_cfg(),
    play_env_cfg=t800_k1_tracking_env_cfg(play=True),
    rl_cfg=t800_k1_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
