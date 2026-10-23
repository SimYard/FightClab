"""Van Damme's helicopter kick (360 split kick).

Tasks:
- ``Fightclab-HelicopterKick-G1``: pure RL from rewards, no reference. It learns a jump-turn, not a kick.
- ``Fightclab-HelicopterKick-Tracking-<robot>``: tracks the kick from video (``video_motion.py``) on G1, R1, H2,
  PM01, T800, T1 and ASIMOV.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from fightclab.tasks.helicopter_kick.env_cfg import g1_helicopter_kick_env_cfg
from fightclab.tasks.helicopter_kick.rl_cfg import g1_helicopter_kick_ppo_runner_cfg
from fightclab.tasks.helicopter_kick.tracking import (
    ROBOT_CFGS,
    g1_helicopter_kick_tracking_env_cfg,
    g1_helicopter_kick_tracking_ppo_runner_cfg,
    helicopter_kick_tracking_env_cfg,
    ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Fightclab-HelicopterKick-G1",
    env_cfg=g1_helicopter_kick_env_cfg(),
    play_env_cfg=g1_helicopter_kick_env_cfg(),
    rl_cfg=g1_helicopter_kick_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Fightclab-HelicopterKick-Tracking-G1",
    env_cfg=g1_helicopter_kick_tracking_env_cfg(),
    play_env_cfg=g1_helicopter_kick_tracking_env_cfg(play=True),
    rl_cfg=g1_helicopter_kick_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

for robot in ROBOT_CFGS:
    register_mjlab_task(
        task_id=f"Fightclab-HelicopterKick-Tracking-{robot.upper()}",
        env_cfg=helicopter_kick_tracking_env_cfg(robot),
        play_env_cfg=helicopter_kick_tracking_env_cfg(robot, play=True),
        rl_cfg=ppo_runner_cfg(robot),
        runner_cls=MotionTrackingOnPolicyRunner,
    )
