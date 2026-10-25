"""Fit capsules and boxes to T800's upper-body visual meshes, for robot-to-robot contact (``k1/duel.py``).

    CUDA_VISIBLE_DEVICES= uv run python tools/fit_primitives_t800.py OUT.json

MuJoCo Warp collides capsule pairs and capsule-box pairs analytically; meshes go through the general convex pipeline,
and 896 convex pieces cut by CoACD made a duel step 1.4 s at 4096 envs. Fitting primitives to a mesh is also closer
per collider than convex decomposition (Knodt and Gao, CGF 2026, arXiv 2602.07369). Each body's SHAPES are fitted by
Adam on points of its visual surface: none may stand out of the primitives (cover), and the primitives' outer surface
should lie on the visual one (tight). Writes OUT.json: per body, its primitives in the body frame (capsule: fromto and
radius; box: pos, quat wxyz, half sizes). Derived data, not in git.
"""

import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial import cKDTree

import fightclab.tasks  # noqa: F401  (before fightclab.motion)
from fightclab.motion import ROBOTS, model

# What a jab meets and what it hits with: the shape and number of primitives per body. The head's surface lies within
# 1.1 cm of 6 capsules' at p95 (3: 1.8 cm), a forearm's within 1.3 cm of 3 (2: 1.7 cm).
SHAPES = {
    "LINK_HEAD_YAW": ("capsule", 6),
    "LINK_WAIST_YAW": ("box", 3),
    "LINK_BASE": ("box", 2),
    "LINK_SHOULDER_YAW_[LR]": ("capsule", 1),
    "LINK_ELBOW_PITCH_[LR]": ("capsule", 1),
    "LINK_ELBOW_YAW_[LR]": ("capsule", 3),
}
SURFACE = 5_000  # points of the visual surface per step, and of the primitives' surface for the stats
FIT_POINTS = 60_000  # points of the visual surface the steps draw from
DENSE = 300_000  # points of the visual surface for the nearest-surface lookup (0.5 mm on a head)
KMEANS_ROUNDS = 20  # rounds of k-means for the starting clusters
ON_SHAPE = 500  # points per primitive's surface per step
STEPS, LEARNING_RATE = 800, 5e-3
COVER_WEIGHT = 20.0  # a point of the robot outside its primitives is worse than a primitive standing off it
INSIDE = -1e-4  # m: a point of a primitive's surface deeper than this in another is not on the outer surface


def surface(v: np.ndarray, f: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points on the triangles v[f], by area."""
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(tri), n, p=area / area.sum())
    a, b = rng.random((2, n, 1))
    a, b = np.where(a + b > 1, 1 - a, a), np.where(a + b > 1, 1 - b, b)
    return tri[pick, 0] + a * (tri[pick, 1] - tri[pick, 0]) + b * (tri[pick, 2] - tri[pick, 0])


def on_segment(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The points of segments a-b nearest to points ``p`` (broadcast)."""
    ab = b - a
    t = (((p - a) * ab).sum(-1, keepdim=True) / (ab * ab).sum(-1, keepdim=True).clamp(min=1e-9)).clamp(0, 1)
    return a + t * ab


def rotation(six: torch.Tensor) -> torch.Tensor:
    """(k, 6) to (k, 3, 3) rotation matrices, columns from Gram-Schmidt."""
    x = torch.nn.functional.normalize(six[:, :3], dim=-1)
    y = torch.nn.functional.normalize(six[:, 3:] - (x * six[:, 3:]).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


class Capsules(torch.nn.Module):
    """``k`` capsules started along the principal axes of point clusters."""

    def __init__(self, clusters: list[np.ndarray]) -> None:
        """Endpoints at the 10th and 90th percentiles of each cluster's main axis, radius its median spread."""
        super().__init__()
        ends, radii = [], []
        for c in clusters:
            centre = c.mean(0)
            axis = np.linalg.svd(c - centre, full_matrices=False)[2][0]
            t = (c - centre) @ axis
            ends.append(np.r_[centre + np.percentile(t, 10) * axis, centre + np.percentile(t, 90) * axis])
            radii.append(np.median(np.linalg.norm(np.cross(c - centre, axis), axis=1)))
        self.ends = torch.nn.Parameter(torch.tensor(np.array(ends)))
        self.log_radius = torch.nn.Parameter(torch.tensor(np.log(radii)))

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """(n, k) signed distances of points ``p`` (n, 3)."""
        a, b, r = self.ends[:, :3], self.ends[:, 3:], self.log_radius.exp()
        return torch.linalg.norm(p[:, None] - on_segment(p[:, None], a, b), dim=-1) - r

    def on_surface(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """``n`` points on each capsule's surface (k * n, 3), differentiable in the parameters."""
        a, b, r = self.ends[:, :3], self.ends[:, 3:], self.log_radius.exp()
        t = torch.rand(len(a), n, 1, generator=gen, dtype=a.dtype)
        d = torch.nn.functional.normalize(torch.randn(len(a), n, 3, generator=gen, dtype=a.dtype), dim=-1)
        x = a[:, None] + t * (b - a)[:, None] + r[:, None, None] * d
        c = on_segment(x, a[:, None], b[:, None])  # pushed out to the surface from the axis
        return (c + r[:, None, None] * torch.nn.functional.normalize(x - c, dim=-1)).reshape(-1, 3)

    def export(self) -> list[dict[str, object]]:
        """The capsules for MuJoCo: fromto and radius."""
        e, r = self.ends.detach().numpy(), self.log_radius.exp().detach().numpy()
        return [{"type": "capsule", "fromto": e[i].tolist(), "radius": float(r[i])} for i in range(len(e))]


class Boxes(torch.nn.Module):
    """``k`` oriented boxes started on the principal axes of point clusters."""

    def __init__(self, clusters: list[np.ndarray]) -> None:
        """Centred on each cluster, half sizes its 5th to 95th percentile spans along its principal axes."""
        super().__init__()
        centres, six, halves = [], [], []
        for c in clusters:
            centre = c.mean(0)
            axes = np.linalg.svd(c - centre, full_matrices=False)[2]
            local = (c - centre) @ axes.T
            centres.append(centre)
            six.append(np.r_[axes[0], axes[1]])
            halves.append((np.percentile(local, 95, axis=0) - np.percentile(local, 5, axis=0)) / 2)
        self.centre = torch.nn.Parameter(torch.tensor(np.array(centres)))
        self.six = torch.nn.Parameter(torch.tensor(np.array(six)))
        self.log_half = torch.nn.Parameter(torch.tensor(np.log(np.array(halves))))

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """(n, k) signed distances of points ``p`` (n, 3)."""
        q = torch.einsum("kji,nkj->nki", rotation(self.six), p[:, None] - self.centre)
        d = q.abs() - self.log_half.exp()
        return torch.linalg.norm(d.clamp(min=0), dim=-1) + d.max(-1).values.clamp(max=0)

    def on_surface(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """``n`` points on each box's faces (k * n, 3), area-weighted, differentiable in the parameters."""
        h, rot = self.log_half.exp(), rotation(self.six)
        k = len(h)
        u = torch.rand(k, n, 3, generator=gen, dtype=h.dtype) * 2 - 1
        area = torch.stack((h[:, 1] * h[:, 2], h[:, 0] * h[:, 2], h[:, 0] * h[:, 1]), -1).detach()
        axis = torch.multinomial(area, n, replacement=True, generator=gen)  # (k, n): the face's normal axis
        side = torch.randint(0, 2, (k, n), generator=gen).to(h.dtype) * 2 - 1
        u = torch.where(torch.nn.functional.one_hot(axis, 3).bool(), side[..., None], u)
        return (self.centre[:, None] + torch.einsum("kij,knj->kni", rot, u * h[:, None])).reshape(-1, 3)

    def export(self) -> list[dict[str, object]]:
        """The boxes for MuJoCo: pos, quat (w, x, y, z), half sizes."""
        out = []
        for c, r, h in zip(
            self.centre.detach().numpy(),
            rotation(self.six).detach().numpy(),
            self.log_half.exp().detach().numpy(),
            strict=True,
        ):
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r.flatten())
            out.append({"type": "box", "pos": c.tolist(), "quat": quat.tolist(), "size": h.tolist()})
        return out


def fit(points: np.ndarray, dense: cKDTree, shape: str, k: int, seed: int) -> tuple[Capsules | Boxes, dict[str, float]]:
    """Primitives of ``shape`` fitted to the surface ``points``; their cover and tightness in cm."""
    rng = np.random.default_rng(seed)
    centres = points[rng.choice(len(points), k, replace=False)]
    for _ in range(KMEANS_ROUNDS):  # k-means for the starting clusters
        label = np.argmin(np.linalg.norm(points[:, None] - centres, axis=-1), axis=1)
        centres = np.array([points[label == j].mean(0) for j in range(k)])
    prims = (Capsules if shape == "capsule" else Boxes)([points[label == j] for j in range(k)])
    opt = torch.optim.Adam(prims.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed)
    p = torch.tensor(points)
    for _ in range(STEPS):
        batch = p[torch.randint(0, len(p), (SURFACE,), generator=gen)]
        cover = prims.sdf(batch).min(-1).values.clamp(min=0).square().mean()
        x = prims.on_surface(ON_SHAPE, gen)
        outer = prims.sdf(x.detach()).min(-1).values > INSIDE  # not inside another primitive
        near = torch.tensor(dense.data[dense.query(x.detach().numpy())[1]])
        tight = (x[outer] - near[outer]).square().sum(-1).mean()
        loss = COVER_WEIGHT * cover + tight
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = prims.sdf(p).min(-1).values.clamp(min=0).numpy()
        x = prims.on_surface(SURFACE, gen)
        x = x[prims.sdf(x).min(-1).values > INSIDE].numpy()
        off = dense.query(x)[0]
    stats = {
        "outside p95": np.percentile(out, 95),
        "outside max": out.max(),
        "off median": np.median(off),
        "off p95": np.percentile(off, 95),
        "off max": off.max(),
    }
    return prims, {k: round(100 * float(v), 2) for k, v in stats.items()}


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)  # the machine is shared
    out_path = Path(sys.argv[1])
    result: dict[str, list[dict[str, object]]] = {}
    m, d, adr = model(ROBOTS["t800"])
    rng = np.random.default_rng(0)
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        shape = next((s for pattern, s in SHAPES.items() if re.fullmatch(pattern, body)), None)
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or shape is None:
            continue
        i = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[i] : m.mesh_faceadr[i] + m.mesh_facenum[i]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        v = v @ rot.reshape(3, 3).T + m.geom_pos[g]  # into the body's frame
        prims, stats = fit(surface(v, f, FIT_POINTS, rng), cKDTree(surface(v, f, DENSE, rng)), *shape, seed=0)
        assert body not in result, f"{body}: a second visual mesh would replace the first's primitives"
        result[body] = prims.export()
        print(f"{body}: {shape[1]} {shape[0]}s, cm: {stats}", flush=True)
    out_path.write_text(json.dumps(result, indent=1))
