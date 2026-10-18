"""Van Damme's helicopter kick (360 split kick) for G1.

Two tasks:
- ``Fightclab-HelicopterKick-G1``: pure RL from rewards, no reference. It learns a jump-turn, not a kick.
- ``Fightclab-HelicopterKick-Tracking-G1``: tracks the keyframe reference from ``reference.py``.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from fightclab.tasks.helicopter_kick.env_cfg import g1_helicopter_kick_env_cfg
from fightclab.tasks.helicopter_kick.rl_cfg import g1_helicopter_kick_ppo_runner_cfg
from fightclab.tasks.helicopter_kick.tracking import (
    g1_helicopter_kick_tracking_env_cfg,
    g1_helicopter_kick_tracking_ppo_runner_cfg,
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
