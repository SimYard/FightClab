"""EngineAI's MJCF models (from GMR and EngineAI's GMR fork) prepared for mjlab, with torque-limited motors.

EngineAI publishes torque limits and reflected inertia in its models but no torque-speed curves, so each motor gets
the speed curve of the Unitree motor nearest in peak torque (``unitree.nearest``).
"""

from pathlib import Path

import mujoco
import numpy as np


def load(xml: Path, textured: Path | None = None) -> mujoco.MjSpec:
    """An EngineAI MJCF without its motors and sensors; collision geoms named like G1's, IMU sensors as in G1's.

    ``textured`` is a folder of EngineAI's textured meshes for the same robot (``meshes/*.msh`` with UVs,
    ``texture/*.png`` for most), drawn instead of the model's one-color ones, which then need not be there.
    """
    spec = mujoco.MjSpec.from_file(str(xml))
    for geom in spec.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = spec.mesh(geom.meshname)
        name = Path(mesh.file).stem
        if textured:
            mesh.file = str(textured / "meshes" / f"{name}.msh")
        elif mesh.file.endswith(".obj"):  # stored as binary MSH in assets/robots, see tools/obj_to_msh.py
            mesh.file = str(Path(mesh.file).with_suffix(".msh"))
        if textured and (png := textured / "texture" / f"{name}.png").exists():
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
            spec.add_material(name=name).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
            geom.material = name
        # The texture's own colors, not tinted by the model's grey; untextured parts are black in EngineAI's CAD.
        geom.rgba = (1, 1, 1, 1) if geom.material else (0.03, 0.03, 0.03, 1)
    for mesh in spec.meshes:
        two_sided(mesh, xml.parent / mesh.file)
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


def two_sided(mesh: mujoco.MjsMesh, msh: Path) -> None:
    """``mesh`` drawn from both sides: its MSH file's faces, and each again wound the other way with flipped normals.

    EngineAI's CAD parts are open shells, T800's with hundreds to thousands of open edges per part. MuJoCo draws front
    faces only, so the backs of those shells showed as holes, through the heels and shins.
    """
    nv, nn, nt, nf = np.fromfile(msh, "<i4", 4)
    assert nv == nn == nt, f"{msh}: one normal and texture coordinate per vertex, as tools/obj_to_msh.py writes"
    data = np.fromfile(msh, "<f4", 8 * nv, offset=16)
    v, n, uv = data[: 3 * nv].reshape(-1, 3), data[3 * nv : 6 * nv].reshape(-1, 3), data[6 * nv :].reshape(-1, 2)
    faces = np.fromfile(msh, "<i4", 3 * nf, offset=16 + 32 * nv).reshape(-1, 3)
    mesh.file = ""
    mesh.uservert = np.r_[v, v].ravel()
    mesh.usernormal = np.r_[n, -n].ravel()
    mesh.usertexcoord = np.r_[uv, uv].ravel()
    mesh.userface = np.r_[faces, faces[:, ::-1] + nv].ravel()


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
