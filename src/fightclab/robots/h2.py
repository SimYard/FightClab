"""Unitree H2 for FightClab: Unitree's MJCF with Unitree motor curves picked by torque limit.

The model is ``unitree_robots/h2/h2_mujoco.xml`` from unitree_mujoco (see ``unitree.MODELS``). H2 is 1.8 m and
71 kg, with 31 motors: 6 per leg, 3 in the waist, 7 per arm and 2 in the head. Its curves are not published, so each
joint gets the smallest published Unitree motor whose peak covers the model's torque limit, capped at that limit.
The legs' 360 N*m is beyond every published motor; they get M107-24's speed curve (H1-2's biggest, 240 N*m) at 1.5x.
"""

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from fightclab.robots import unitree
from fightclab.robots.g1 import STIFF_COLLISION
from fightclab.robots.unitree import M107_24, motor_for

H2_XML = unitree.MODELS / "h2/h2_mujoco.xml"


def get_spec() -> mujoco.MjSpec:
    """H2 for mjlab, see ``unitree.load``. Its feet are the ankle pitch links (the roll axis comes first)."""
    return unitree.load(H2_XML, "ankle_pitch_link")


# Torque limits from the model's motors, N*m.
H2_MOTORS = (
    motor_for((".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"), 360, M107_24, 1.5),
    motor_for((".*_ankle_pitch_joint",), 67),
    motor_for((".*_ankle_roll_joint",), 19),
    motor_for(("waist_yaw_joint",), 120),
    motor_for(("waist_roll_joint", "waist_pitch_joint"), 180),
    motor_for((".*_shoulder_pitch_joint",), 130),
    motor_for((".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"), 60),
    motor_for((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 10),
    motor_for(("head_.*_joint",), 50),
)

# mjlab's G1 rule: an action of 1 moves the target by a quarter of the peak torque's worth of stiffness.
H2_ACTION_SCALE = {
    name: 0.25 * motor.effort_limit / motor.stiffness for motor in H2_MOTORS for name in motor.target_names_expr
}

KNEES_BENT = EntityCfg.InitialStateCfg(
    pos=(0, 0, 1.0),  # standing straight, the pelvis is at 1.04 m
    joint_pos={".*_hip_pitch_joint": -0.3, ".*_knee_joint": 0.6, ".*_ankle_pitch_joint": -0.3},
    joint_vel={".*": 0.0},
)


def get_h2_cfg() -> EntityCfg:
    """H2 with all collisions on (self-collisions too), stiff contacts as on our G1, Unitree motor curves."""
    return EntityCfg(
        init_state=KNEES_BENT,
        collisions=(STIFF_COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=H2_MOTORS, soft_joint_pos_limit_factor=0.9),
    )
