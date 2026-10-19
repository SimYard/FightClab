"""Unitree's MJCF models (unitree_mujoco) prepared for mjlab, and Unitree motors picked by torque limit."""

from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1

from fightclab.robots.g1 import N5020_16, N7520_14, N7520_22, Motor, unitree_motor

# Unitree's bigger published motors (unitree_rl_lab, assets/robots/unitree_actuators.py), from H1-2.
M107_15 = Motor(y1=150.0, x1=14.0, x2=25.6, fs=0.0, fd=0.0)
M107_24 = Motor(y1=240.0, x1=8.8, x2=16.0, fs=0.0, fd=0.0)
# Reflected inertia per motor, kg*m^2, from the same file.
ARMATURE = {
    N5020_16: 0.003609725,
    N7520_14: 0.01017752,
    N7520_22: 0.025101925,
    M107_15: 0.063259741,
    M107_24: 0.160478022,
}


def load(xml: Path, foot: str, placeholders: tuple[str, ...] = (), imu: str = "imu") -> mujoco.MjSpec:
    """A G1-like MJCF for mjlab: without floor, placeholder bodies, motors and sensors; collision geoms named like G1's.

    ``foot`` is the last body of each leg without its side prefix; its geoms become ``left_foot1_collision``, ...
    Adds the IMU sensors mjlab's tracking observations read, named as in G1's model, on the ``imu`` site.
    """
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.meshdir = str(xml.parent / spec.meshdir)
    for body in placeholders:
        spec.delete(spec.body(body))
    for element in [*spec.worldbody.geoms, *spec.actuators, *spec.sensors]:  # mjlab adds terrain and motors
        spec.delete(element)
    for body in spec.bodies:
        geoms = [g for g in body.geoms if g.contype or g.conaffinity]
        prefix = body.name.replace(foot, "foot")
        for i, geom in enumerate(geoms, 1):
            geom.name = f"{prefix}{i}_collision"
            geom.group = 3  # hidden, as in mjlab's G1: in group 0 R1's torso box showed through its chest
    site = mujoco.mjtObj.mjOBJ_SITE
    spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=site, objname=imu)
    spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=site, objname=imu)
    return spec


def nearest(names: tuple[str, ...], limit: float, armature: float | None = None) -> DcMotorActuatorCfg:
    """For other makers' robots: the model's torque limit, a Unitree motor's speed curve.

    The Unitree motor is the one nearest in peak torque; its reflected inertia is used unless the model gives one.
    Gains follow mjlab's G1 rule, as in ``motor_for``.
    """
    speed_of = min(ARMATURE, key=lambda m: abs(m.y1 - limit))
    armature = armature or ARMATURE[speed_of]
    return unitree_motor(
        names,
        Motor(y1=limit, x1=speed_of.x1, x2=speed_of.x2, fs=0.0, fd=0.0),
        armature=armature,
        stiffness=armature * g1.NATURAL_FREQ**2,
        damping=2 * g1.DAMPING_RATIO * armature * g1.NATURAL_FREQ,
    )


def motor_for(names: tuple[str, ...], limit: float, motor: Motor | None = None, count: float = 1) -> DcMotorActuatorCfg:
    """The smallest published Unitree motor with a peak of at least ``limit`` N*m, capped at ``limit``.

    mjlab's G1 gains: stiffness from the reflected inertia at a 10 Hz natural frequency, damping ratio 2.
    """
    motor = motor or min((m for m in ARMATURE if m.y1 * count >= limit), key=lambda m: m.y1)
    armature = ARMATURE[motor]
    return unitree_motor(
        names,
        motor,
        armature=armature,
        stiffness=armature * g1.NATURAL_FREQ**2,
        damping=2 * g1.DAMPING_RATIO * armature * g1.NATURAL_FREQ,
        count=count,
        cap=limit,
    )
