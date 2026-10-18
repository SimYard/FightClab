"""Unitree G1 for FightClab: mjlab's G1 with stiffer contacts and Unitree's motor torque-speed curves.

Motor data comes from Unitree's own RL repo, unitree_rl_lab, ``assets/robots/unitree_actuators.py``.
Joint-to-motor mapping and the URDF limits are from ``unitree_ros`` ``g1_29dof_rev_1_0.urdf``.
"""

from dataclasses import dataclass, replace

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# mjlab's default contact time constant (0.02 s) lets the feet sink up to ~6 cm into the floor on
# hard landings. 0.01 s is the stiffest MuJoCo allows at our 5 ms timestep (2x the timestep).
STIFF_COLLISION = replace(g1.FULL_COLLISION, solref=(0.01, 1.0))

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


def unitree_motor(
    names: tuple[str, ...],
    motor: Motor,
    *,
    armature: float,
    stiffness: float,
    damping: float,
    count: int = 1,
    cap: float | None = None,
) -> DcMotorActuatorCfg:
    """Unitree's torque-speed curve: ``y1`` up to ``x1`` rad/s, then linear down to zero at ``x2``.

    mjlab's DC motor curve is ``min(effort_limit, saturation * (1 - |v| / velocity_limit))``, which matches
    that shape exactly when the line goes through (x1, y1) and (x2, 0).
    ``count`` motors drive the joint together through a linkage; ``cap`` is the joint's URDF effort limit.

    ponytail: braking (torque against the motion) is also capped at y1. Unitree measures a higher braking
    peak (Y2), but mjlab's DC model has one limit for both directions; a custom actuator would add Y2.
    """
    peak = count * motor.y1
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


G1_MOTORS = (
    unitree_motor(
        (".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
        N7520_14,
        armature=g1.ARMATURE_7520_14,
        stiffness=g1.STIFFNESS_7520_14,
        damping=g1.DAMPING_7520_14,
    ),
    unitree_motor(
        (".*_hip_roll_joint", ".*_knee_joint"),
        N7520_22,
        armature=g1.ARMATURE_7520_22,
        stiffness=g1.STIFFNESS_7520_22,
        damping=g1.DAMPING_7520_22,
    ),
    unitree_motor(
        (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"),
        N5020_16,
        armature=g1.ARMATURE_5020,
        stiffness=g1.STIFFNESS_5020,
        damping=g1.DAMPING_5020,
    ),
    unitree_motor(
        (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
        W4010_25,
        armature=g1.ARMATURE_4010,
        stiffness=g1.STIFFNESS_4010,
        damping=g1.DAMPING_4010,
    ),
    # Ankles and waist roll/pitch: two 5020s through a 4-bar linkage, nominal 1:1 as mjlab assumes,
    # capped at the URDF joint limit of 35 N*m.
    unitree_motor(
        (".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_pitch_joint", "waist_roll_joint"),
        N5020_16,
        armature=g1.ARMATURE_5020,
        stiffness=g1.STIFFNESS_5020,
        damping=g1.DAMPING_5020,
        count=2,
        cap=35.0,
    ),
)


def get_g1_cfg() -> EntityCfg:
    """G1 in the boxing guard, all collisions on (self-collisions too), stiff contacts, Unitree's motor curves."""
    cfg = get_g1_robot_cfg()
    cfg.init_state = GUARD_KEYFRAME
    cfg.collisions = (STIFF_COLLISION,)
    cfg.articulation = EntityArticulationInfoCfg(actuators=G1_MOTORS, soft_joint_pos_limit_factor=0.9)
    return cfg


if __name__ == "__main__":
    # The DC curve must pass through Unitree's points: y1 at rest and at the knee x1, zero at no-load x2.
    import torch
    from mjlab.actuator.dc_actuator import dc_motor_clip

    m = unitree_motor(("knee",), N7520_22, armature=0.0, stiffness=0.0, damping=0.0)
    vel = torch.tensor([0.0, N7520_22.x1, (N7520_22.x1 + N7520_22.x2) / 2, N7520_22.x2])
    ones = torch.ones_like(vel)
    limit = dc_motor_clip(ones * 1e6, ones * m.saturation_effort, ones * m.velocity_limit, ones * m.effort_limit, vel)
    assert torch.allclose(limit, torch.tensor([111.0, 111.0, 55.5, 0.0]), atol=0.01), limit
    print("knee torque at", vel.tolist(), "rad/s:", [round(x, 1) for x in limit.tolist()], "N*m")
