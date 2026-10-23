"""What every robot's mjlab config shares: stiff contacts, mjlab's G1 action scale rule, the entity."""

from collections.abc import Callable
from dataclasses import replace

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# mjlab's default contact time constant (0.02 s) lets the feet sink up to ~6 cm into the floor on
# hard landings. 0.01 s is the stiffest MuJoCo allows at our 5 ms timestep (2x the timestep).
STIFF_COLLISION = replace(g1.FULL_COLLISION, solref=(0.01, 1.0))


def action_scale(motors: tuple[DcMotorActuatorCfg, ...]) -> dict[str, float]:
    """Mjlab's G1 rule: an action of 1 moves the target by a quarter of the peak torque's worth of stiffness."""
    return {name: 0.25 * motor.effort_limit / motor.stiffness for motor in motors for name in motor.target_names_expr}


def entity_cfg(
    spec_fn: Callable[[], mujoco.MjSpec], motors: tuple[DcMotorActuatorCfg, ...], init_state: EntityCfg.InitialStateCfg
) -> EntityCfg:
    """A robot with all collisions on (self-collisions too), stiff contacts as on our G1, and ``motors``."""
    return EntityCfg(
        init_state=init_state,
        collisions=(STIFF_COLLISION,),
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(actuators=motors, soft_joint_pos_limit_factor=0.9),
    )
