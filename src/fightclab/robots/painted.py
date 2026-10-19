"""Meshes painted by ``tools/paint_meshes.py``, drawn in place of the model's grey ones.

Per mesh, its parts: one per color, and textured decals.
"""

import json
from pathlib import Path

import mujoco

from fightclab.robots import ASSETS


def apply(spec: mujoco.MjSpec, robot: str) -> None:
    """Swap the visual mesh geoms that have painted parts for one geom per part; no-op if the robot is unpainted."""
    folder = ASSETS / robot / "painted"
    if not (folder / "parts.json").exists():
        return
    parts = json.loads((folder / "parts.json").read_text())
    textures: set[str] = set()
    for body in spec.bodies:
        for geom in list(body.geoms):
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH or geom.contype or geom.conaffinity:
                continue
            name = Path(spec.mesh(geom.meshname).file).stem
            if name not in parts:
                continue
            for i, (file, rgba, *texture) in enumerate(parts[name]):
                # Shell inertia: a part is an open surface, and the body's inertia is the model's own anyway.
                mesh = spec.add_mesh(name=f"{geom.meshname}_{i}", file=str(folder / file))
                mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                part = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    pos=geom.pos,
                    quat=geom.quat,
                    rgba=rgba,
                    contype=0,
                    conaffinity=0,
                    group=geom.group,
                    density=0,
                )
                if texture:  # one texture and material per file: parts of one color share theirs
                    tex = f"painted_{Path(texture[0]).stem}"
                    if tex not in textures:
                        textures.add(tex)
                        spec.add_texture(name=tex, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(folder / texture[0]))
                        spec.add_material(name=tex).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
                    part.material = tex
            spec.delete(geom)
