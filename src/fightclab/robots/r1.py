"""Unitree R1 for FightClab: Unitree's MJCF without its placeholder joints, with Unitree motor curves.

The model is unitree_mujoco's ``unitree_robots/r1/R1_C++.xml``, in ``assets/robots/r1``.
R1 has 24 motors: 6 per leg, waist roll and yaw, 5 per arm. Its per-joint torque limits (88, 139, 50 and 25 N*m) are
those of G1's motors, so G1's Unitree curves are reused by torque class; R1's own curves are not published.
"""

import mujoco
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from fightclab.robots import ASSETS, painted, unitree
from fightclab.robots.g1 import N5020_16, N7520_14, N7520_22, STIFF_COLLISION, unitree_motor

R1_XML = ASSETS / "r1/R1_C++.xml"
# Unitree parks these at z = 20 m, unattached, so the SDK sees 29 joints; R1 has no waist pitch or wrist pitch/yaw.
PLACEHOLDERS = ("waist_pitch", "left_wrist_pitch", "left_wrist_yaw", "right_wrist_pitch", "right_wrist_yaw")


def get_spec() -> mujoco.MjSpec:
    """R1 for mjlab, see ``unitree.load``, painted (see ``painted``)."""
    spec = unitree.load(R1_XML, "ankle_roll_link", PLACEHOLDERS)
    painted.apply(spec, "r1")
    return spec


R1_MOTORS = (
    unitree_motor(
        (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),  # 88 N*m
        N7520_14,
        armature=g1.ARMATURE_7520_14,
        stiffness=g1.STIFFNESS_7520_14,
        damping=g1.DAMPING_7520_14,
    ),
    unitree_motor(
        (".*_knee_joint",),  # 139 N*m
        N7520_22,
        armature=g1.ARMATURE_7520_22,
        stiffness=g1.STIFFNESS_7520_22,
        damping=g1.DAMPING_7520_22,
    ),
    unitree_motor(
        (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"),  # 25 N*m
        N5020_16,
        armature=g1.ARMATURE_5020,
        stiffness=g1.STIFFNESS_5020,
        damping=g1.DAMPING_5020,
    ),
    unitree_motor(
        (".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_roll_joint"),  # 50 N*m: two 5020s, as on G1
        N5020_16,
        armature=g1.ARMATURE_5020,
        stiffness=g1.STIFFNESS_5020,
        damping=g1.DAMPING_5020,
        count=2,
        cap=50.0,
    ),
)

# mjlab's G1 rule: an action of 1 moves the target by a quarter of the peak torque's worth of stiffness.
R1_ACTION_SCALE = {
    name: 0.25 * motor.effort_limit / motor.stiffness for motor in R1_MOTORS for name in motor.target_names_expr
}

KNEES_BENT = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.70),
    joint_pos={".*_hip_pitch_joint": -0.3, ".*_knee_joint": 0.6, ".*_ankle_pitch_joint": -0.3},
    joint_vel={".*": 0.0},
)


def get_r1_cfg() -> EntityCfg:
    """R1 with all collisions on (self-collisions too), stiff contacts as on our G1, Unitree's motor curves."""
    return EntityCfg(
        init_state=KNEES_BENT,
        collisions=(STIFF_COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=R1_MOTORS, soft_joint_pos_limit_factor=0.9),
    )
