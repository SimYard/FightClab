"""EngineAI's MJCF models (from their GMR fork, BSD-3) prepared for mjlab, with torque-limited motors.

EngineAI publishes torque limits and reflected inertia in its models but no torque-speed curves, so each motor gets
the speed curve of the Unitree motor nearest in peak torque.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg

from fightclab.robots import unitree


def load(xml: Path, textured: Path | None = None) -> mujoco.MjSpec:
    """An EngineAI MJCF without its motors and sensors; collision geoms named like G1's, IMU sensors as in G1's.

    ``textured`` is a folder of EngineAI's textured meshes for the same robot (``meshes/*.obj`` with UVs,
    ``texture/*.png``), drawn instead of the model's one-color ones.
    """
    spec = mujoco.MjSpec.from_file(str(xml))
    for geom in spec.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = spec.mesh(geom.meshname)
        name = Path(mesh.file).stem
        if textured and (png := textured / "texture" / f"{name}.png").exists():
            mesh.file = str(textured / "meshes" / f"{name}.obj")
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
            spec.add_material(name=name).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
            geom.material = name
        # The texture's own colors, not tinted by the model's grey; untextured parts are black in EngineAI's CAD.
        geom.rgba = (1, 1, 1, 1) if geom.material else (0.03, 0.03, 0.03, 1)
    for element in [*spec.actuators, *spec.sensors]:  # mjlab adds its own motors
        spec.delete(element)
    for body in spec.bodies:
        geoms = [g for g in body.geoms if g.contype or g.conaffinity]
        prefix = {"LINK_FOOT_L": "left_foot", "LINK_FOOT_R": "right_foot"}.get(body.name, body.name)
        for i, geom in enumerate(geoms, 1):
            geom.name = f"{prefix}{i}_collision"
    site = mujoco.mjtObj.mjOBJ_SITE
    spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=site, objname="imu")
    spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=site, objname="imu")
    return spec


def motor(names: tuple[str, ...], limit: float, armature: float) -> DcMotorActuatorCfg:
    """The model's torque limit and reflected inertia, the speed curve of the Unitree motor nearest in torque."""
    return unitree.nearest(names, limit, armature)


def g1_names(torso: str, hand: str) -> dict[str, str]:
    """EngineAI's names for the G1 bodies mjlab's tracking task uses."""
    pairs = (
        ("hip_roll_link", "HIP_ROLL"),
        ("knee_link", "KNEE_PITCH"),
        ("ankle_roll_link", "ANKLE_ROLL"),
        ("shoulder_roll_link", "SHOULDER_ROLL"),
        ("elbow_link", "ELBOW_PITCH"),
        ("wrist_yaw_link", hand),
    )
    limbs = {f"{side}_{g}": f"LINK_{e}_{side[0].upper()}" for side in ("left", "right") for g, e in pairs}
    return {"pelvis": "LINK_BASE", "torso_link": torso, **limbs}
