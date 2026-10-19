"""EngineAI T800 for FightClab: EngineAI's MJCF with torque-limited motors, see ``engineai``.

The model is ``assets/engineai_t800/xml/serial_t800.xml`` from EngineAI's GMR fork (github.com/engineai-robotics/GMR),
kept outside the repo in ~/robot_models/engineai_gmr. T800 is 1.7 m and 85 kg, with 25 motors: 6 per leg, torso yaw,
5 per arm, 2 in the head. Torque limits and reflected inertia are the model's; speed curves are the nearest Unitree
motor's: M107-24 (240 N*m) for 222-415 N*m, M107-15 (150 N*m) for 160, N7520-14 (71 N*m) for 52.
"""

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from fightclab.robots import engineai
from fightclab.robots.g1 import STIFF_COLLISION

T800_XML = Path.home() / "robot_models/engineai_gmr/assets/engineai_t800/xml/serial_t800.xml"
G1_NAMES = engineai.g1_names(torso="LINK_WAIST_YAW", hand="WRIST_END")


def get_spec() -> mujoco.MjSpec:
    """T800 for mjlab, see ``engineai.load``; the model comes textured."""
    return engineai.load(T800_XML)


T800_MOTORS = (
    engineai.motor(("J0[03]_.*", "J0[69]_.*"), 415, 0.243),  # hip pitch, knee
    engineai.motor(("J0[17]_.*",), 370, 0.141),  # hip roll
    engineai.motor(("J0[28]_.*", "J12_.*"), 222, 0.045),  # hip yaw, torso yaw
    engineai.motor(("J0[45]_.*", "J1[01]_.*", "J1[3-6]_.*", "J(18|19|20|21)_.*"), 160, 0.035),  # ankles, arms
    engineai.motor(("J1[7]_.*", "J2[2-4]_.*"), 52, 0.007),  # elbow yaw, head
)

# mjlab's G1 rule: an action of 1 moves the target by a quarter of the peak torque's worth of stiffness.
T800_ACTION_SCALE = {
    name: 0.25 * motor.effort_limit / motor.stiffness for motor in T800_MOTORS for name in motor.target_names_expr
}

KNEES_BENT = EntityCfg.InitialStateCfg(
    pos=(0, 0, 1.0),
    joint_pos={"J0[06]_HIP_PITCH_.*": -0.3, "J0[39]_KNEE_PITCH_.*": 0.6, "J(04|10)_ANKLE_PITCH_.*": -0.3},
    joint_vel={".*": 0.0},
)


def get_t800_cfg() -> EntityCfg:
    """T800 with all collisions on (self-collisions too), stiff contacts as on our G1."""
    return EntityCfg(
        init_state=KNEES_BENT,
        collisions=(STIFF_COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=T800_MOTORS, soft_joint_pos_limit_factor=0.9),
    )
