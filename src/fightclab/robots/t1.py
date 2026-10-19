"""Booster T1 for FightClab: Booster's MJCF (from GMR) with torque-limited motors.

The model is ``assets/booster_t1/T1_serial.xml`` from GMR, in ``assets/robots/t1``. T1 is 1.2 m and 35 kg, with 23
motors: 6 per leg, waist yaw, 4 per arm, 2 in the head. Its root is the trunk; the legs hang from a waist yaw joint.

Peak torques: Booster's integrated joints come in 7, 36, 60, 90 and 130 N*m (peak; RoboCup HSL 2026 team specs), and
130 is quoted for the knee. Which joint gets which is not published. Booster's models give about half those numbers
(knee 60, hip pitch 45, hip roll and yaw 30, arms 18), rated torques by the look of it, so each joint gets the size
twice its model value: knee 130, hip pitch 90, hip roll, yaw and waist 60, ankles and arms 36, head 7. That mapping is
an inference, not a datasheet. The model's reflected inertia is a placeholder, so each motor gets the speed curve and
reflected inertia of the Unitree motor nearest in torque (``unitree.nearest``).
"""

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from fightclab.robots import ASSETS
from fightclab.robots.g1 import STIFF_COLLISION
from fightclab.robots.unitree import nearest

T1_XML = ASSETS / "t1/T1_serial.xml"
# T1's names for the G1 bodies mjlab's tracking task uses. Its waist link stands in for G1's pelvis.
G1_NAMES = {
    "pelvis": "Waist",
    "torso_link": "Trunk",
    **{
        f"{side}_{g1}": t1.format(S=side[0].upper(), Side=side.capitalize(), side=side)
        for side in ("left", "right")
        for g1, t1 in (
            ("hip_roll_link", "Hip_Roll_{Side}"),
            ("knee_link", "Shank_{Side}"),
            ("ankle_roll_link", "{side}_foot_link"),
            ("shoulder_roll_link", "A{S}2"),
            ("elbow_link", "A{S}3"),
            ("wrist_yaw_link", "{side}_hand_link"),
        )
    },
}


def get_spec() -> mujoco.MjSpec:
    """T1 without its floor, motors and sensors; collision geoms named like G1's, an IMU on the trunk."""
    spec = mujoco.MjSpec.from_file(str(T1_XML))
    spec.meshdir = str(T1_XML.parent / spec.meshdir)
    for element in [*spec.worldbody.geoms, *spec.actuators, *spec.sensors]:  # mjlab adds terrain and motors
        spec.delete(element)
    for body in spec.bodies:
        geoms = [g for g in body.geoms if g.contype or g.conaffinity]
        prefix = body.name.replace("_foot_link", "_foot")
        for i, geom in enumerate(geoms, 1):
            geom.name = f"{prefix}{i}_collision"
    spec.body("Trunk").add_site(name="imu")
    site = mujoco.mjtObj.mjOBJ_SITE
    spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=site, objname="imu")
    spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=site, objname="imu")
    return spec


T1_MOTORS = (
    nearest((".*_Knee_Pitch",), 130),  # model 60
    nearest((".*_Hip_Pitch",), 90),  # model 45
    nearest((".*_Hip_Roll", ".*_Hip_Yaw", "Waist"), 60),  # model 30
    nearest((".*_Ankle_Pitch", ".*_Ankle_Roll", ".*_Shoulder_.*", ".*_Elbow_.*"), 36),  # model 15-20
    nearest(("AAHead_yaw", "Head_pitch"), 7),  # model 7
)

# mjlab's G1 rule: an action of 1 moves the target by a quarter of the peak torque's worth of stiffness.
T1_ACTION_SCALE = {
    name: 0.25 * motor.effort_limit / motor.stiffness for motor in T1_MOTORS for name in motor.target_names_expr
}

KNEES_BENT = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.68),
    joint_pos={".*_Hip_Pitch": -0.3, ".*_Knee_Pitch": 0.6, ".*_Ankle_Pitch": -0.3},
    joint_vel={".*": 0.0},
)


def get_t1_cfg() -> EntityCfg:
    """T1 with all collisions on (self-collisions too), stiff contacts as on our G1."""
    return EntityCfg(
        init_state=KNEES_BENT,
        collisions=(STIFF_COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=T1_MOTORS, soft_joint_pos_limit_factor=0.9),
    )
