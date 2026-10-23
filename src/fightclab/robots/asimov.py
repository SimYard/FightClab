"""Asimov v1 (Menlo) for FightClab: Asimov's MJCF with Menlo's actuator table.

The model is Asimov's ``sim-model/xmls/asimov_1.xml``, in ``assets/robots/asimov`` under its CERN-OHL-S licence.
Asimov v1 is 1.2 m and 32 kg, with 23 motors in the model: 6 per leg, waist yaw, 5 per arm.
Torque and speed limits are Menlo's per-joint table (docs.menlo.ai/asimov/1/hardware/joint-design-and-actuation):
peak torque, modeled as full torque up to the velocity limit. The URDF's efforts are close to the rated torques,
a third of the peaks. The ankle's two linkage actuators (A 145.4, B 57.6 N*m peak) are mapped to pitch and roll as the
URDF maps their rated torques. Reflected inertia is the MJCF's. Joint ranges are the MJCF's except where Menlo's table
allows more (``RANGES``). Its joints are narrow for this kick: hip flexion 120 deg, hip roll and yaw +-45 deg, knee
86 deg, ankles +-20 and +-6 deg.
"""

import mujoco
from mjlab.entity import EntityCfg

from fightclab.robots import ASSETS, painted, unitree
from fightclab.robots.common import action_scale, entity_cfg
from fightclab.robots.unitree import Motor, unitree_motor

ASIMOV_XML = ASSETS / "asimov/xmls/asimov_1.xml"
G1_NAMES = {"pelvis": "pelvis_link", "torso_link": "waist_yaw_link"}  # its other bodies are named like G1's

# Joint: Menlo's peak torque (N*m) and velocity limit (rad/s), MJCF armature (kg*m^2).
LIMITS = {
    "hip_pitch": (120, 12.57, 0.095625),  # EC-A6416-P2-25
    "hip_roll": (90, 3.98, 0.11),  # EC-A5013-H17-100
    "hip_yaw": (60, 5.45, 0.038),  # EC-A3814-H14-107
    "knee": (75, 12.25, 0.0339552),  # EC-A4315-P2-36
    "ankle_pitch": (145.4, 9.32, 0.0565056),  # EC-A4310-P2-36, linkage A
    "ankle_roll": (57.6, 9.32, 0.0565056),  # EC-A4310-P2-36, linkage B
    "waist_yaw": (120, 12.57, 0.095625),
    "shoulder_pitch": (90, 3.98, 0.11),
    "shoulder_roll": (75, 12.25, 0.0339552),
    "shoulder_yaw": (60, 5.45, 0.038),
    "elbow": (36, 9.32, 0.0282528),
    "wrist_yaw": (36, 9.32, 0.0282528),
}


# Menlo's table allows the hips 120 deg of flexion and 57 of extension, the MJCF +-90; the kicking leg was held at
# 90. The elbows bend to 140 deg, not 120. In radians, as in the table; tools/gvhmr_to_csv.py repeats them.
RANGES = {
    "left_hip_pitch_joint": (-2.09, 1.00),
    "right_hip_pitch_joint": (-1.00, 2.09),
    "left_elbow_joint": (0.0, 2.44),
    "right_elbow_joint": (-2.44, 0.0),
}


def get_spec() -> mujoco.MjSpec:
    """Asimov for mjlab, see ``unitree.load``, with Menlo's joint ranges, painted (see ``painted``)."""
    spec = unitree.load(ASIMOV_XML, "ankle_roll_link", imu="imu_in_pelvis")
    for joint, limits in RANGES.items():
        spec.joint(joint).range = limits
    painted.apply(spec, "asimov")
    return spec


ASIMOV_MOTORS = tuple(
    unitree_motor(
        (f".*{joint}_joint",),
        Motor(y1=effort, x1=speed, x2=speed * 1.01, fs=0.0, fd=0.0),  # full torque up to the top speed
        armature=armature,
    )
    for joint, (effort, speed, armature) in LIMITS.items()
)
ASIMOV_ACTION_SCALE = action_scale(ASIMOV_MOTORS)

KNEES_BENT = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.61),  # standing straight, the pelvis is at 0.63 m
    joint_pos={
        "left_hip_pitch_joint": -0.3,
        "left_knee_joint": 0.6,
        "left_ankle_pitch_joint": -0.3,
        "right_hip_pitch_joint": 0.3,
        "right_knee_joint": -0.6,
        "right_ankle_pitch_joint": 0.3,
    },
    joint_vel={".*": 0.0},
)  # the right leg's axes point the other way


def get_asimov_cfg() -> EntityCfg:
    """Asimov, see ``entity_cfg``."""
    return entity_cfg(get_spec, ASIMOV_MOTORS, KNEES_BENT)
