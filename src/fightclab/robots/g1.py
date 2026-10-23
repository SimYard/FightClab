"""Unitree G1 for FightClab: mjlab's G1 with stiffer contacts and Unitree's motor torque-speed curves (``unitree``).

Joint-to-motor mapping and the URDF limits are from ``unitree_ros`` ``g1_29dof_rev_1_0.urdf``.
"""

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityCfg

from fightclab.robots.common import entity_cfg
from fightclab.robots.unitree import N5020_16, N7520_14, N7520_22, W4010_25, unitree_motor

# Joint order of mjlab's csv_to_npz (Unitree's SDK order).
G1_JOINTS = (
    *(f"left_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    *(f"right_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *(f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")),
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    *(f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")),
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Orthodox boxing guard: the start pose, and what the helicopter kick's guard reward asks to return to.
# Left and right palm targets in the torso frame (x forward, z up). G1's head center is at (0, 0, 0.43).
GUARD_HANDS = ((0.30, 0.10, 0.36), (0.20, -0.08, 0.40))
# Left and right foot targets in the pelvis heading frame, xy. Standing feet are at (0.04, +-0.12).
GUARD_FEET = ((0.17, 0.13), (-0.10, -0.13))
GUARD_PELVIS_HEIGHT = 0.72

# Solved by damped least-squares IK for the targets above: soles flat on the floor, waist at zero.
GUARD_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, GUARD_PELVIS_HEIGHT),
    joint_pos={
        "left_hip_pitch_joint": -0.635,
        "left_hip_roll_joint": 0.016,
        "left_hip_yaw_joint": -0.012,
        "left_knee_joint": 0.869,
        "left_ankle_pitch_joint": -0.234,
        "left_ankle_roll_joint": -0.019,
        "right_hip_pitch_joint": -0.156,
        "right_hip_roll_joint": -0.019,
        "right_hip_yaw_joint": 0.008,
        "right_knee_joint": 0.851,
        "right_ankle_pitch_joint": -0.695,
        "right_ankle_roll_joint": 0.020,
        "left_shoulder_pitch_joint": -0.862,
        "left_shoulder_roll_joint": 0.155,
        "left_shoulder_yaw_joint": -0.003,
        "left_elbow_joint": -0.352,
        "left_wrist_roll_joint": 0.057,
        "left_wrist_pitch_joint": 0.313,
        "left_wrist_yaw_joint": 0.010,
        "right_shoulder_pitch_joint": -0.892,
        "right_shoulder_roll_joint": 0.132,
        "right_shoulder_yaw_joint": 0.029,
        "right_elbow_joint": -0.613,
        "right_wrist_roll_joint": -0.126,
        "right_wrist_pitch_joint": 0.056,
        "right_wrist_yaw_joint": -0.014,
    },
    joint_vel={".*": 0.0},
)

G1_MOTORS = (
    unitree_motor(
        (".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"), N7520_14, armature=g1.ARMATURE_7520_14
    ),
    unitree_motor((".*_hip_roll_joint", ".*_knee_joint"), N7520_22, armature=g1.ARMATURE_7520_22),
    unitree_motor(
        (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"), N5020_16, armature=g1.ARMATURE_5020
    ),
    unitree_motor((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), W4010_25, armature=g1.ARMATURE_4010),
    # Ankles and waist roll/pitch: two 5020s through a 4-bar linkage, nominal 1:1 as mjlab assumes,
    # capped at the URDF joint limit of 35 N*m.
    unitree_motor(
        (".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_pitch_joint", "waist_roll_joint"),
        N5020_16,
        armature=g1.ARMATURE_5020,
        count=2,
        cap=35.0,
    ),
)


def get_g1_cfg() -> EntityCfg:
    """G1 in the boxing guard, all collisions on (self-collisions too), stiff contacts, Unitree's motor curves."""
    return entity_cfg(g1.get_spec, G1_MOTORS, GUARD_KEYFRAME)
