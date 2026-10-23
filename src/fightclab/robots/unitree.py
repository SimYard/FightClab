"""Unitree's motors, and Unitree's MJCF models (unitree_mujoco) prepared for mjlab.

Motor data comes from Unitree's own RL repo, unitree_rl_lab, ``assets/robots/unitree_actuators.py``.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1


@dataclass(frozen=True)
class Motor:
    """A Unitree motor's measured torque-speed curve and friction."""

    y1: float
    """Peak torque with torque and speed in the same direction, N*m."""
    x1: float
    """Top speed at full torque, the knee of the curve, rad/s."""
    x2: float
    """No-load speed, rad/s."""
    fs: float
    """Static friction, N*m."""
    fd: float
    """Viscous friction, N*m*s/rad."""


N7520_14 = Motor(y1=71.0, x1=22.63, x2=35.52, fs=1.6, fd=0.16)
N7520_22 = Motor(y1=111.0, x1=14.5, x2=22.7, fs=2.4, fd=0.24)
N5020_16 = Motor(y1=24.8, x1=30.86, x2=40.13, fs=0.6, fd=0.06)
W4010_25 = Motor(y1=4.8, x1=15.3, x2=24.76, fs=0.6, fd=0.06)
# The bigger ones, from H1-2.
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


def unitree_motor(
    names: tuple[str, ...], motor: Motor, *, armature: float, count: float = 1, cap: float | None = None
) -> DcMotorActuatorCfg:
    """Unitree's torque-speed curve: ``y1`` up to ``x1`` rad/s, then linear down to zero at ``x2``.

    mjlab's DC motor curve is ``min(effort_limit, saturation * (1 - |v| / velocity_limit))``, which matches
    that shape exactly when the line goes through (x1, y1) and (x2, 0).
    Gains follow mjlab's G1 rule: stiffness from the reflected inertia ``armature`` at a 10 Hz natural frequency,
    damping ratio 2. ``count`` motors drive the joint together through a linkage (torque, inertia, gains and
    friction scale with it); ``cap`` is the joint's URDF effort limit.

    ponytail: braking (torque against the motion) is also capped at y1. Unitree measures a higher braking
    peak (Y2), but mjlab's DC model has one limit for both directions; a custom actuator would add Y2.
    """
    peak = count * motor.y1
    stiffness = armature * g1.NATURAL_FREQ**2
    damping = 2 * g1.DAMPING_RATIO * armature * g1.NATURAL_FREQ
    return DcMotorActuatorCfg(
        target_names_expr=names,
        saturation_effort=peak * motor.x2 / (motor.x2 - motor.x1),
        velocity_limit=motor.x2,
        effort_limit=peak if cap is None else min(peak, cap),
        frictionloss=count * motor.fs,
        viscous_damping=count * motor.fd,
        armature=count * armature,
        stiffness=count * stiffness,
        damping=count * damping,
    )


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
    """
    speed_of = min(ARMATURE, key=lambda m: abs(m.y1 - limit))
    return unitree_motor(
        names,
        Motor(y1=limit, x1=speed_of.x1, x2=speed_of.x2, fs=0.0, fd=0.0),
        armature=armature or ARMATURE[speed_of],
    )


def motor_for(names: tuple[str, ...], limit: float, motor: Motor | None = None, count: float = 1) -> DcMotorActuatorCfg:
    """The smallest published Unitree motor with a peak of at least ``limit`` N*m, capped at ``limit``."""
    motor = motor or min((m for m in ARMATURE if m.y1 * count >= limit), key=lambda m: m.y1)
    return unitree_motor(names, motor, armature=ARMATURE[motor], count=count, cap=limit)
