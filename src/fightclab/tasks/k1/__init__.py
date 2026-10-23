"""K-1 kickboxing skills tracked from tutorial videos.

Tasks:
- ``Fightclab-K1-Footwork-T800``: T800 moves on velocity commands in a K-1 stance (``footwork.py``), styled by AMP.
  The K-1 coach failed its footwork on every command (it leaned instead of stepping, squared up and hopped going
  back, lunged and skidded sideways), so the footwork is now tracked instead:
- ``Fightclab-K1-Footwork-Tracking-T800``: T800 tracks the footwork clips joined into one motion
  (``footwork_motion.py``).
- ``Fightclab-K1-Tracking-T800``: T800 tracks ``MOTION_T800``, the K-1 clip made by GVHMR, ``tools/gvhmr_to_csv.py``,
  ``video_motion ... --ground`` and ``motion_npz ... <out.npz>``. The kick task's setup without its flight reward:
  these motions stay on the floor.
"""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.rewards import joint_vel_l2
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.mdp.rewards import body_angular_velocity_penalty
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from fightclab.tasks.helicopter_kick.env_cfg import FEET
from fightclab.tasks.helicopter_kick.tracking import helicopter_kick_tracking_env_cfg, ppo_runner_cfg
from fightclab.tasks.k1.footwork import still_head_action_scale, t800_footwork_env_cfg, t800_footwork_ppo_runner_cfg
from fightclab.tracking import feet_contact_slip

MOTION_T800 = Path(__file__).parent / "k1_t800.npz"
# Generated K-1 step-drags on random commands (``steps ... random:600``, then ``motion_npz``), on the stance of the
# footwork clips joined by ``footwork_motion``: the clips themselves brought lunges, rocking and hanging feet.
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
    """T800 on the footwork clips, tracked frame by frame.

    No low kicks yet: the footwork comes first, then they are a separate fine-tuning stage (``add_low_kicks``).
    """
    cfg = t800_k1_tracking_env_cfg(play)
    cfg.commands["motion"].motion_file = str(MOTION_FOOTWORK_T800)
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = still_head_action_scale()  # the reference's head never moves; the policy turned it up to 75 deg
    # Footwork without rocking: the tracked policy still swayed its pelvis and torso about 1 deg at 1-2 Hz and moved
    # its arms 0.12 rad/s, against a steadier reference and a still guard.
    for body in ("LINK_BASE", "LINK_WAIST_YAW"):
        cfg.rewards[f"rock_{body}"] = RewardTermCfg(
            func=body_angular_velocity_penalty,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(body,))},
        )
    # The kick's tracking widths let footwork lag and drag: 20 cm behind the reference still earned 64% of the root
    # position reward, and a sole on the floor instead of 4 cm up 85% of the feet's height reward. The robot made
    # 0.12-0.22 m/s of the reference's 0.25 and slid its following foot in 16% of grounded frames (the reference: 3%).
    root = cfg.rewards["motion_global_root_pos"]
    root.weight, root.params["std"] = 2.0, 0.1
    cfg.rewards["feet_height"].params["std"] = 0.03
    cfg.rewards["upper_body_vel"] = RewardTermCfg(
        func=joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"J(1[2-9]|2[0-4])_.*",))},
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
    task_id="Fightclab-K1-Tracking-T800",
    env_cfg=t800_k1_tracking_env_cfg(),
    play_env_cfg=t800_k1_tracking_env_cfg(play=True),
    rl_cfg=t800_k1_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
