"""Convert Wavefront OBJ meshes to MuJoCo's binary MSH, next to them, and delete the OBJ files.

    python tools/obj_to_msh.py assets/robots/pm01/textured/meshes/*.obj

MSH keeps the vertices, texture coordinates and normals, as one vertex per distinct corner (MSH has no separate
texture or normal indices), in binary: the OBJ text was 8 million lines of the repo's diffs.
"""

import struct
import sys
from pathlib import Path

import numpy as np


def write_msh(path: Path, corners: np.ndarray, normals: np.ndarray, uv: np.ndarray) -> None:
    """Triangles given per corner, (n, 3, 3) positions and normals and (n, 3, 2) OBJ texture coordinates, as MSH.

    MuJoCo flips an OBJ's v coordinate on loading but takes an MSH's as it is, so v is flipped here.
    """
    uv = np.stack([uv[..., 0], 1 - uv[..., 1]], axis=-1)
    rows = np.concatenate([corners, normals, uv], axis=2).reshape(-1, 8).astype("<f4")
    unique, index = np.unique(rows, axis=0, return_inverse=True)
    with path.open("wb") as f:
        f.write(struct.pack("<4i", len(unique), len(unique), len(unique), len(index) // 3))
        for block in (unique[:, :3], unique[:, 3:6], unique[:, 6:]):
            f.write(np.ascontiguousarray(block).tobytes())
        f.write(index.astype("<i4").tobytes())


def obj_to_msh(obj: Path) -> None:
    """Convert one OBJ whose faces index positions, texture coordinates and normals (``f v/vt/vn ...``)."""
    lists: dict[str, list[list[float]]] = {"v": [], "vt": [], "vn": []}
    faces = []
    for line in obj.read_text().splitlines():
        tag, *rest = line.split() or [""]
        if tag in lists:
            lists[tag].append([float(x) for x in rest[: 2 if tag == "vt" else 3]])
        elif tag == "f":
            ring = [[int(i) - 1 if i else -1 for i in corner.split("/")] for corner in rest]  # -1: none given
            faces += [(ring[0], ring[k], ring[k + 1]) for k in range(1, len(ring) - 1)]  # fans for polygons
    index = np.array(faces)  # (n, 3 corners, v/vt/vn)
    lists["vt"].append([0.0, 0.0])  # index -1: corners without texture coordinates sample the texture's corner
    v, vt, vn = (np.array(lists[k]) for k in ("v", "vt", "vn"))
    write_msh(obj.with_suffix(".msh"), v[index[..., 0]], vn[index[..., 2]], vt[index[..., 1]])
    obj.unlink()


if __name__ == "__main__":
    for name in sys.argv[1:]:
        obj_to_msh(Path(name))
