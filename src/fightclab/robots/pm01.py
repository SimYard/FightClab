"""EngineAI PM01 for FightClab: EngineAI's MJCF with torque-limited motors, see ``engineai``.

The model is ``assets/engineai_pm01/xml/serial_pm_v2.xml`` from GMR, kept outside the repo. PM01 is 1.4 m and 41 kg,
with 24 motors: 6 per leg, waist yaw, 5 per arm, head yaw. Torque limits are 164 and 61 N*m; the speed curves are
M107-15's (150 N*m) and N7520-14's (71 N*m).
"""

from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from fightclab.robots import engineai
from fightclab.robots.g1 import STIFF_COLLISION

PM01_XML = Path.home() / "GMR/assets/engineai_pm01/xml/serial_pm_v2.xml"
G1_NAMES = engineai.g1_names(torso="LINK_TORSO_YAW", hand="ELBOW_END")

# EngineAI's textured meshes of the same robot, from github.com/engineai-robotics/engineai_robotics_description.
PM01_TEXTURED = Path.home() / "robot_models/engineai_robotics_description/pm01_edu"


def get_spec() -> mujoco.MjSpec:
    """PM01 for mjlab, see ``engineai.load``."""
    spec = engineai.load(PM01_XML, PM01_TEXTURED)
    # The meshes leave the chest badge's hole open; a light disc fills it (center and axis measured by raycasting).
    center, axis = np.array([0.0986, 0.0, 0.1822]), np.array([0.988, 0.0, 0.155])
    spec.body("LINK_TORSO_YAW").add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.029, 0, 0),
        fromto=(*(center - 0.006 * axis), *(center - 0.002 * axis)),
        rgba=(0.88, 0.88, 0.88, 1),
        contype=0,
        conaffinity=0,
        group=2,
        density=0,
    )
    return spec


PM01_MOTORS = (
    engineai.motor(("J0[013]_.*", "J0[679]_.*"), 164, 0.045),  # hip pitch and roll, knee
    engineai.motor(("J0[2458]_.*", "J1[0-9]_.*", "J2[0-3]_.*"), 61, 0.039),  # the rest
)

# mjlab's G1 rule: an action of 1 moves the target by a quarter of the peak torque's worth of stiffness.
PM01_ACTION_SCALE = {
    name: 0.25 * motor.effort_limit / motor.stiffness for motor in PM01_MOTORS for name in motor.target_names_expr
}

KNEES_BENT = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.80),
    joint_pos={"J0[06]_HIP_PITCH_.*": -0.3, "J0[39]_KNEE_PITCH_.*": 0.6, "J(04|10)_ANKLE_PITCH_.*": -0.3},
    joint_vel={".*": 0.0},
)


def get_pm01_cfg() -> EntityCfg:
    """PM01 with all collisions on (self-collisions too), stiff contacts as on our G1."""
    return EntityCfg(
        init_state=KNEES_BENT,
        collisions=(STIFF_COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=PM01_MOTORS, soft_joint_pos_limit_factor=0.9),
    )
