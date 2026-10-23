"""EngineAI T800 for FightClab: EngineAI's MJCF with torque-limited motors, see ``engineai``.

The model is ``assets/engineai_t800/xml/serial_t800.xml`` from EngineAI's GMR fork (github.com/engineai-robotics/GMR),
in ``assets/robots/t800``. T800 is 1.7 m and 85 kg, with 25 motors: 6 per leg, torso yaw,
5 per arm, 2 in the head. Torque limits and reflected inertia are the model's; speed curves are the nearest Unitree
motor's: M107-24 (240 N*m) for 222-415 N*m, M107-15 (150 N*m) for 160, N7520-14 (71 N*m) for 52.
"""

import mujoco
from mjlab.entity import EntityCfg

from fightclab.robots import ASSETS, engineai
from fightclab.robots.common import action_scale, entity_cfg
from fightclab.robots.unitree import nearest

T800_XML = ASSETS / "t800/xml/serial_t800.xml"
G1_NAMES = engineai.g1_names(torso="LINK_WAIST_YAW", hand="WRIST_END")


def get_spec() -> mujoco.MjSpec:
    """T800 for mjlab, see ``engineai.load``; the model comes textured."""
    return engineai.load(T800_XML)


T800_MOTORS = (
    nearest(("J0[03]_.*", "J0[69]_.*"), 415, 0.243),  # hip pitch, knee
    nearest(("J0[17]_.*",), 370, 0.141),  # hip roll
    nearest(("J0[28]_.*", "J12_.*"), 222, 0.045),  # hip yaw, torso yaw
    nearest(("J0[45]_.*", "J1[01]_.*", "J1[3-6]_.*", "J(18|19|20|21)_.*"), 160, 0.035),  # ankles, arms
    nearest(("J1[7]_.*", "J2[2-4]_.*"), 52, 0.007),  # elbow yaw, head
)
T800_ACTION_SCALE = action_scale(T800_MOTORS)

KNEES_BENT = EntityCfg.InitialStateCfg(
    pos=(0, 0, 1.0),
    joint_pos={"J0[06]_HIP_PITCH_.*": -0.3, "J0[39]_KNEE_PITCH_.*": 0.6, "J(04|10)_ANKLE_PITCH_.*": -0.3},
    joint_vel={".*": 0.0},
)


def get_t800_cfg() -> EntityCfg:
    """T800, see ``entity_cfg``."""
    return entity_cfg(get_spec, T800_MOTORS, KNEES_BENT)
