"""Paint a robot's one-grey STL meshes in the real robot's colors and close the holes they leave open.

    python tools/paint_meshes.py r1

Each face gets, in order of precedence: its mesh's color; its connected part's (CAD exports keep discs and plates as
separate parts, so these get exact edges); a region's (a test on face centers and normals). Vents (slots sunk into a
surface) are found by their walls and filled. Decals (shapes, text, patterns) are drawn into textures projected along
an axis onto the faces that face it, at 0.125 mm per pixel. Flat open holes are closed with a fan of faces. Rules are
written for the left side and centered meshes: right-side meshes (``right_*``) are painted mirrored. Writes, per mesh,
a MuJoCo MSH mesh per color and per decal group, with its PNG texture, to assets/robots/<robot>/painted, and
parts.json, which ``fightclab.robots.painted`` reads.
"""

import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import trimesh
from matplotlib import font_manager
from matplotlib.path import Path as Polygon
from obj_to_msh import write_msh  # tools/, on the path when this script runs
from PIL import Image, ImageDraw, ImageFont

from fightclab.robots import ASSETS

Rgba = tuple[float, float, float, float]
Box = tuple[tuple[float, float, float], tuple[float, float, float]]
Shape = tuple[Any, ...]

WHITE: Rgba = (0.93, 0.93, 0.94, 1.0)
LIGHT_GREY: Rgba = (0.78, 0.79, 0.81, 1.0)
NAVY: Rgba = (0.1, 0.15, 0.36, 1.0)
BLACK: Rgba = (0.04, 0.04, 0.05, 1.0)
DARK: Rgba = (0.12, 0.12, 0.13, 1.0)
GRAPHITE: Rgba = (0.24, 0.24, 0.26, 1.0)
METAL: Rgba = (0.35, 0.35, 0.37, 1.0)
CARBON: Rgba = (0.17, 0.17, 0.18, 1.0)
RED: Rgba = (0.8, 0.08, 0.1, 1.0)
ORANGE: Rgba = (0.95, 0.4, 0.06, 1.0)
SILVER: Rgba = (0.72, 0.72, 0.74, 1.0)

FLAT = 0.015  # m: a hole's rim at most this thick is flat enough to close with a fan
FACING = 0.2  # a decal reaches faces whose normal is at most ~78 degrees off its axis
WALL = 0.35  # a vent's walls are faces at most this aligned with the surface's normal
VENT = (0.004, 0.02, 0.005)  # m: a vent's length is between the first two, its width under the third
CREASE = 25.0  # degrees: edges sharper than this bound a surface
PX = 8000  # decal texture pixels per m
PLANE = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}  # plane coords of decals along each axis


def box(center: tuple[float, float, float], half: float = 0.004) -> Box:
    """A small box around a part's center, to pick that part."""
    return (tuple(c - half for c in center), tuple(c + half for c in center))  # type: ignore[return-value]


def mirrored(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """A left-half outline (from y = 0 round to y = 0) closed with its mirror image across y = 0 (first coord)."""
    return points + [(-a, b) for a, b in reversed(points[1:-1])]


def flip_shape(shape: Shape) -> Shape:
    """``shape`` reflected across the first plane coord's zero (y = 0 on +x decals)."""
    kind, color, *args = shape
    if kind in ("poly", "outline"):
        return (kind, color, [(-a, b) for a, b in args[0]], *args[1:])
    return (kind, color, (-args[0][0], args[0][1]), *args[1:])


def bars(
    start: tuple[float, float], step: tuple[float, float], n: int, *, w: float, h: float, skew: float
) -> list[Shape]:
    """``n`` red parallelogram stripes ``w`` by ``h``, sheared by ``skew``, from ``start`` in steps of ``step``."""
    out: list[Shape] = []
    for k in range(n):
        a, b = start[0] + k * step[0], start[1] + k * step[1]
        out.append(("poly", RED, [(a, b), (a + w, b + skew), (a + w, b + skew - h), (a, b - h)]))
    return out


def in_outline(points: np.ndarray, outline: list[tuple[float, float]]) -> np.ndarray:
    """Which 2D ``points`` lie inside ``outline``."""
    return Polygon(outline).contains_points(points)


# Unitree R1, from its product photos. White shells; navy shoulders, waist, hip and outer knee joints, shoes, ear discs,
# the pelvis' sides and the inverted A on the forehead; black visor, soles, vents and inner knee joints; graphite
# hands; carbon plates above the ankles; red chevron on the chest, "01" and stripes on the thighs. Coordinates are read
# off orthographic views of the meshes; the forehead's A follows the creases of its panel.
R1_GOGGLES = mirrored(
    [
        (0.0, -0.0120),
        (-0.025, -0.0127),
        (-0.045, -0.0136),
        (-0.058, -0.0130),
        (-0.058, -0.040),
        (-0.0473, -0.0392),
        (-0.0427, -0.0410),
        (-0.0287, -0.0425),
        (-0.0157, -0.0435),
        (-0.0122, -0.0427),
        (-0.0080, -0.0326),
        (-0.0055, -0.031),
        (0.0, -0.030),
    ]
)
R1_GOGGLES_DEPTH = 0.07  # m: the goggles stand in front of x = 70 mm, the temples behind
R1_SIDE = 0.045  # m: the goggles' ends and the pelvis' sides lie past |y| = 45 mm
R1_GOGGLES_SIDE = [  # (x, z): the frame's end, seen from the side
    (0.0817, -0.011),
    (0.0915, -0.0135),
    (0.0905, -0.02),
    (0.085, -0.0325),
    (0.0808, -0.0445),
    (0.079, -0.0462),
    (0.0735, -0.0395),
    (0.074, -0.03),
    (0.0788, -0.0192),
]
R1_PELVIS_TOP = -0.03  # the navy band's top, between the upper vents' top and the hip joints
R1_PELVIS_BAND: Shape = (  # its inner edge runs between the upper and the lower vents
    "poly",
    NAVY,
    [
        (-0.05, R1_PELVIS_TOP),
        (-0.0475, -0.05),
        (-0.0455, -0.064),
        (-0.041, -0.08),
        (-0.037, -0.095),
        (-0.036, -0.14),
        (-0.095, -0.14),
        (-0.095, R1_PELVIS_TOP),
    ],
)
R1_PELVIS_SIDE: list[Shape] = [
    ("poly", NAVY, [(-0.05, R1_PELVIS_TOP), (0.1, R1_PELVIS_TOP), (0.1, -0.13), (-0.05, -0.13)])
]
R1_PELVIS_UNDER: list[Shape] = [  # (x, y): the band seen from below, past |y| = R1_SIDE
    ("poly", NAVY, [(-0.05, 0.045), (0.1, 0.045), (0.1, 0.1), (-0.05, 0.1)]),
    ("poly", NAVY, [(-0.05, -0.045), (0.1, -0.045), (0.1, -0.1), (-0.05, -0.1)]),
]
R1_FOREHEAD: list[Shape] = [
    (
        "poly",
        NAVY,
        [(-0.0395, 0.0385), (-0.0178, -0.0074), (0.018, -0.0074), (0.0392, 0.0385), (0.0392, 0.06), (-0.0395, 0.06)],
    ),
    ("poly", WHITE, [(-0.0117, 0.0175), (0.0117, 0.0175), (0.0, 0.0045)]),
]
R1_EAR: list[Shape] = [("circle", NAVY, (0.005, -0.020), 0.0235)]
R1_BLADE: Shape = (
    "poly",
    NAVY,
    [
        (0.079, 0.240),
        (0.064, 0.205),
        (0.051, 0.170),
        (0.041, 0.135),
        (0.034, 0.105),
        (0.030, 0.075),
        (0.038, 0.100),
        (0.050, 0.130),
        (0.064, 0.160),
        (0.081, 0.192),
        (0.101, 0.226),
    ],
)
R1_CHEST: list[Shape] = [
    ("outline", RED, [(-0.0215, 0.2165), (0.0215, 0.2165), (0.006, 0.1875), (-0.006, 0.1875)], 0.0028),
    R1_BLADE,
    flip_shape(R1_BLADE),
]
ASIMOV_EAR = 0.04  # m: the head's surfaces centered past |y| = 40 mm are its ears
ASIMOV_CAP: list[Shape] = [("poly", ORANGE, [(-0.1, -0.005), (0.1, -0.005), (0.1, 0.1), (-0.1, 0.1)])]  # z > -5 mm
ROBOTS: dict[str, dict[str, Any]] = {
    "r1": {
        "meshes": "r1/meshes",
        "colors": (
            ("head_pitch_link", LIGHT_GREY),
            ("*_shoulder_pitch_link", NAVY),
            ("waist_roll_link", NAVY),
            ("*_hip_pitch_link", NAVY),
            ("*_wrist_roll_link", GRAPHITE),
            ("*_ankle_roll_link", NAVY),
            ("*_ankle_*", METAL),
            ("imu_in_pelvis_link", GRAPHITE),
            ("*", WHITE),
        ),
        "parts": (  # (mesh, box holding the part's center, color)
            ("waist_yaw_link", box((0.059, 0.0, 0.201), 0.006), DARK),  # inside the chevron
            ("*_hip_yaw_link", box((-0.023, -0.041, -0.159)), BLACK),  # inner knee disc
            ("*_knee_link", box((0.0, 0.017, 0.0)), NAVY),  # outer knee disc
            ("*_knee_link", box((0.003, -0.021, -0.14)), BLACK),  # side strut
            ("*_knee_link", box((0.022, -0.021, -0.237)), CARBON),  # plate above the ankle
            ("*_ankle_roll_link", box((0.04, 0.0, -0.041)), BLACK),  # sole
        ),
        "surfaces": (  # (mesh, test on the surface's center c, color): the goggles' lens, frame and the frame's ends
            ("head_yaw_link", lambda c: (c[:, 0] > R1_GOGGLES_DEPTH) & in_outline(c[:, 1:], R1_GOGGLES), BLACK),
            (
                "head_yaw_link",
                lambda c: (
                    (c[:, 0] > R1_GOGGLES_DEPTH) & (abs(c[:, 1]) > R1_SIDE) & in_outline(c[:, ::2], R1_GOGGLES_SIDE)
                ),
                BLACK,
            ),
        ),
        "regions": (  # (mesh, test on face centers p and normals n, color)
            # The pelvis' sides that neither its front nor its side decals reach.
            (
                "pelvis_link",
                lambda p, n: (
                    (abs(p[:, 1]) > R1_SIDE)
                    & (p[:, 2] < R1_PELVIS_TOP)
                    & (n[:, 0] <= FACING)
                    & (abs(n[:, 1]) <= FACING)
                    & (n[:, 2] >= -FACING)
                ),
                NAVY,
            ),
        ),
        "vents": (  # (mesh, surface normal, box, decal axis): drawn into that mesh and axis' first decal
            ("pelvis_link", (1, 0, 0), ((0.04, -0.062, -0.07), (0.1, 0.062, -0.03)), "+x"),
            ("pelvis_link", (0.5, 0.866, 0), ((0.03, 0.04, -0.1), (0.1, 0.075, -0.05)), "+x"),
            ("pelvis_link", (0.5, -0.866, 0), ((0.03, -0.075, -0.1), (0.1, -0.04, -0.05)), "+x"),
        ),
        "decals": (  # (mesh, axis, shapes, paints over this color only, skipped parts)
            ("head_yaw_link", "+x", R1_FOREHEAD, WHITE, (box((0.009, 0.0, 0.033)),)),  # not on the dome
            ("pelvis_link", "+x", [R1_PELVIS_BAND, flip_shape(R1_PELVIS_BAND)], WHITE, ()),
            ("pelvis_link", "+y", R1_PELVIS_SIDE, WHITE, ()),
            ("pelvis_link", "-y", R1_PELVIS_SIDE, WHITE, ()),
            ("pelvis_link", "-z", R1_PELVIS_UNDER, WHITE, (((-1, -1, R1_PELVIS_TOP), (1, 1, 1)),)),  # band only
            ("head_yaw_link", "+y", R1_EAR, WHITE, ()),
            ("head_yaw_link", "-y", R1_EAR, WHITE, ()),
            ("waist_yaw_link", "+x", R1_CHEST, WHITE, ()),
            ("left_hip_roll_link", "+x", [("text", RED, (0.024, -0.033), 0.018, "01")], WHITE, ()),
            (
                "left_hip_yaw_link",
                "+x",
                bars((0.008, -0.03), (0.002, -0.0105), 5, w=0.013, h=0.0045, skew=-0.003),
                WHITE,
                (),
            ),
            (
                "right_hip_yaw_link",
                "+x",
                bars((-0.028, -0.03), (0.0, -0.0105), 5, w=0.011, h=0.0045, skew=0.003)
                + bars((0.022, -0.03), (0.0, -0.0105), 5, w=0.011, h=0.0045, skew=-0.003),
                WHITE,
                (),
            ),
            ("*_knee_link", "+x", [("poly", NAVY, [(-0.026, -0.068), (-0.016, -0.068), (-0.021, -0.077)])], WHITE, ()),
            (
                "*_knee_link",
                "+x",
                [("checker", (0.11, 0.11, 0.12, 1.0), (-0.045, -0.28, 0.003, -0.195), 0.0015)],
                CARBON,
                (),
            ),
        ),
        "caps": (
            ("waist_yaw_link", ((-0.2, -0.08, 0.15), (-0.03, 0.08, 0.235)), WHITE),  # cut-outs in the back plate
            ("head_yaw_link", ((0.05, -0.06, -0.06), (0.2, 0.06, 0.0)), BLACK),  # visor slit
        ),
    },
    # Menlo Asimov 1, from Menlo's renders and photos: gloss black; orange helmet, chest shell, shoulder caps, elbow
    # and knee plates, and bands above the elbows and the ankles; black visor and ears; silver ankle links and rods.
    "asimov": {
        "meshes": "asimov/assets/meshes",
        "colors": (("NECK_PITCH", ORANGE), ("*_ANKLE_A", SILVER), ("*", BLACK)),
        "surfaces": (("NECK_PITCH", lambda c: abs(c[:, 1]) > ASIMOV_EAR, BLACK),),  # ears: discs and rims
        "picks": (  # (mesh, axis, point seen along it, color): the CAD surface seen there
            ("NECK_PITCH", "+x", (0.0, 0.07), BLACK),  # visor
            ("WAIST_YAW", "+x", (0.0, 0.2), ORANGE),  # chest shell
            ("*_SHOULDER_ROLL", "+y", (0.0, -0.131), ORANGE),  # band above the elbow
            ("*_ELBOW", "+y", (-0.005, 0.005), ORANGE),  # plate
            ("*_KNEE", "+y", (-0.005, -0.02), ORANGE),  # plate
            ("*_KNEE", "+y", (-0.003, -0.262), ORANGE),  # band above the ankle
            ("*_KNEE", "+x", (0.0, -0.262), ORANGE),
            ("*_KNEE", "-y", (-0.003, -0.262), ORANGE),
            ("*_KNEE", "-x", (0.0, -0.262), ORANGE),
            ("*_KNEE", "+x", (-0.021, -0.298), SILVER),  # rods
            ("*_KNEE", "+x", (0.021, -0.298), SILVER),
        ),
        # The shoulder caps are the top of the upper arm's shell, which is one surface down to the elbow.
        "decals": tuple(("*_SHOULDER_ROLL", axis, ASIMOV_CAP, BLACK, ()) for axis in ("+x", "-x", "+y", "-y", "+z")),
    },
}


def inside(points: np.ndarray, b: Box) -> np.ndarray:
    """Which of ``points`` lie in box ``b``."""
    return np.all((points >= b[0]) & (points <= b[1]), axis=-1)


def caps(mesh: trimesh.Trimesh, b: Box) -> np.ndarray:
    """Fan triangles (both sides) closing the flat boundary cycles centered in ``b``, as (n, 3, 3) corners."""
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    graph = nx.Graph(edges[counts == 1].tolist())
    triangles = []
    for nodes in nx.connected_components(graph):
        loop = graph.subgraph(nodes)
        points = mesh.vertices[list(nodes)]
        center = points.mean(axis=0)
        flat = np.ptp(points, axis=0).min() < FLAT
        cycle = set(dict(loop.degree()).values()) == {2}
        if not (flat and cycle and inside(center, b)):
            continue
        ring = mesh.vertices[nx.cycle_basis(loop)[0]]
        for p, q in zip(ring, np.roll(ring, -1, axis=0), strict=False):
            triangles += [(center, p, q), (center, q, p)]
    return np.array(triangles).reshape(-1, 3, 3)


def first_hit(tri: np.ndarray, axis: str, point: tuple[float, float]) -> int | None:
    """The triangle a ray from outside, along ``axis`` reversed through plane point ``point``, hits first."""
    sign, ax = (1 if axis[0] == "+" else -1), "xyz".index(axis[1])
    p = tri[:, :, list(PLANE[axis[1]])] - point

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    w = np.stack([cross(p[:, 1], p[:, 2]), cross(p[:, 2], p[:, 0]), cross(p[:, 0], p[:, 1])], axis=1)
    total = w.sum(axis=1)
    hit = np.flatnonzero((total != 0) & np.all(np.sign(w) == np.sign(total)[:, None], axis=1))
    if not len(hit):
        return None
    depth = (w[hit] * tri[hit, :, ax]).sum(axis=1) / total[hit]
    return int(hit[np.argmax(sign * depth)])


def hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Convex hull of 2D ``points``, counterclockwise (Andrew's monotone chain)."""
    pts = sorted(set(map(tuple, np.round(points, 6))))

    def turn(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and turn(out[-2], out[-1], p) <= 0:  # noqa: PLR2004
                out.pop()
            out.append(p)
        return out[:-1]

    return half(pts) + half(pts[::-1])


def vents(mesh: trimesh.Trimesh, normal: np.ndarray, b: Box, axis: str) -> tuple[np.ndarray, list[Shape]]:
    """The walls of slots sunk into the surface in ``b``, and the slots' outlines in ``axis``'s decal plane."""
    normal = normal / np.linalg.norm(normal)
    wall = inside(mesh.triangles_center, b) & (abs(face_normals(mesh.triangles) @ normal) < WALL)
    pairs = mesh.face_adjacency[wall[mesh.face_adjacency].all(axis=1)]
    graph = nx.Graph(pairs.tolist())
    a, c = PLANE[axis[1]]
    walls = np.zeros(len(wall), bool)
    shapes: list[Shape] = []
    for component in nx.connected_components(graph):
        faces = list(component)
        pts = mesh.triangles[faces].reshape(-1, 3)
        flat = pts - pts.mean(axis=0)
        spans = np.ptp(flat @ np.linalg.svd(flat, full_matrices=False)[2].T, axis=0)
        if VENT[0] < spans[0] < VENT[1] and spans[1] < VENT[2]:
            walls[faces] = True
            shapes.append(("poly", BLACK, hull(pts[:, [a, c]])))
    return walls, shapes


def draw(shapes: list[Shape], base: Rgba, lo: np.ndarray, hi: np.ndarray) -> Image.Image:
    """The decal texture over plane box ``lo``..``hi``: ``base`` with ``shapes`` drawn on it, antialiased."""
    scale = 2 * PX
    w, h = int(np.ceil((hi[0] - lo[0]) * scale)), int(np.ceil((hi[1] - lo[1]) * scale))
    img = Image.new("RGB", (max(w, 2), max(h, 2)), rgb(base))
    d = ImageDraw.Draw(img)

    def px(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] - lo[0]) * scale, (hi[1] - p[1]) * scale)

    for kind, color, *args in shapes:
        if kind == "poly":
            d.polygon([px(p) for p in args[0]], fill=rgb(color))
        elif kind == "outline":
            pts = [px(p) for p in args[0]]
            d.line([*pts, pts[0], pts[1]], fill=rgb(color), width=round(args[1] * scale), joint="curve")
        elif kind == "circle":
            (x, y), r = px(args[0]), args[1] * scale
            d.ellipse([x - r, y - r, x + r, y + r], fill=rgb(color))
        elif kind == "text":
            name = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
            font = ImageFont.truetype(name, round(args[1] * scale))
            d.text(px(args[0]), args[2], fill=rgb(color), font=font, anchor="mm")
        elif kind == "checker":
            a0, b0, a1, b1 = args[0]
            cell = args[1]
            for i, a in enumerate(np.arange(a0, a1, cell)):
                for j, bb in enumerate(np.arange(b0, b1, cell)):
                    if (i + j) % 2:
                        d.rectangle([px((a, bb + cell)), px((a + cell, bb))], fill=rgb(color))
    return img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS)


def decal_points(shape: Shape) -> list[tuple[float, float]]:
    """Plane points bounding a decal shape."""
    kind, _, *args = shape
    if kind in ("poly", "outline"):
        return list(args[0])
    if kind in ("text", "circle"):
        (a, b), r = args[0], args[1]
        return [(a - r, b - r), (a + r, b + r)]
    a0, b0, a1, b1 = args[0]
    return [(a0, b0), (a1, b1)]


def rgb(color: Rgba) -> tuple[int, int, int]:
    """``color`` as 8-bit RGB."""
    return (round(255 * color[0]), round(255 * color[1]), round(255 * color[2]))


def face_normals(tri: np.ndarray) -> np.ndarray:
    """Unit normals of triangles by their winding, as MuJoCo takes them (some STL files store others)."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def corner_normals(mesh: trimesh.Trimesh, crease: float = 40.0) -> np.ndarray:
    """Smooth normals per face corner, (n, 3, 3), kept sharp across creases sharper than ``crease`` degrees.

    Computed once on the whole mesh, so the parts it is cut into shade without seams where they meet.
    """
    normals, areas = face_normals(mesh.triangles), mesh.area_faces
    out = np.repeat(normals[:, None, :], 3, axis=1)
    cos = np.cos(np.radians(crease))
    for v, around in enumerate(mesh.vertex_faces):
        faces = around[around >= 0]
        n = normals[faces]
        summed = ((n @ n.T > cos) * areas[faces]) @ n
        summed /= np.linalg.norm(summed, axis=1, keepdims=True)
        for f, s in zip(faces, summed, strict=False):
            out[f, mesh.faces[f] == v] = s
    return out


def paint_mesh(path: Path, cfg: dict[str, Any], out: Path) -> list[tuple[str, Rgba, str]]:
    """Paint one mesh; returns its parts as (MSH file, white, PNG texture)."""
    name = path.stem
    mesh = trimesh.load(path, process=True)
    flip = name.lower().startswith("right_")
    if flip:  # paint the mirror image with the left side's rules
        mesh.vertices[:, 1] *= -1
        mesh.invert()
    tri, centers, normals = mesh.triangles, mesh.triangles_center, face_normals(mesh.triangles)
    kinds = ("parts", "surfaces", "picks", "regions", "vents", "decals", "caps")
    rules = {k: [r for r in cfg.get(k, ()) if fnmatch.fnmatch(name, r[0])] for k in kinds}
    base = next(color for pattern, color in cfg["colors"] if fnmatch.fnmatch(name, pattern))
    colors = np.array([base] * len(tri))
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(tri))
    part_center = np.zeros((len(tri), 3))
    for lab in np.unique(labels):  # a part's center is its bounding box's
        pts = tri[labels == lab].reshape(-1, 3)
        part_center[labels == lab] = (pts.min(axis=0) + pts.max(axis=0)) / 2
    for _, b, color in rules["parts"]:
        colors[inside(part_center, b)] = color
    if rules["surfaces"] or rules["picks"]:
        # Surfaces: patches bounded by the CAD model's creases, so a surface painted whole has its exact edges.
        smooth = mesh.face_adjacency[mesh.face_adjacency_angles < np.radians(CREASE)]
        surface = trimesh.graph.connected_component_labels(smooth, node_count=len(tri))
        area = mesh.area_faces
        weight = np.bincount(surface, area)
        surface_center = np.stack([np.bincount(surface, area * centers[:, i]) / weight for i in range(3)], axis=1)
        for _, test, color in rules["surfaces"]:
            colors[test(surface_center[surface])] = color
        for _, axis, point, color in rules["picks"]:
            if (face := first_hit(tri, axis, point)) is None:
                print(f"  {name}: nothing seen along {axis} at {point}")
            else:
                colors[surface == surface[face]] = color
    for _, test, color in rules["regions"]:
        colors[test(centers, normals)] = color
    decals = [(m, axis, list(shapes), over, skip) for m, axis, shapes, over, skip in rules["decals"]]
    for _, normal, b, axis in rules["vents"]:
        walls, shapes = vents(mesh, np.array(normal, float), b, axis)
        colors[walls] = BLACK
        next(d for d in decals if d[1] == axis)[2].extend(shapes)

    unmirror = np.array([1, -1, 1]) if flip else np.ones(3)  # back to the right side, if mirrored
    vertices = mesh.vertices * unmirror
    corner_n = corner_normals(mesh) * unmirror

    def write(stem: str, faces: np.ndarray, uv: np.ndarray, normals: np.ndarray, png: str) -> None:
        if flip:  # restore the winding
            faces, uv, normals = faces[:, [0, 2, 1]], uv[:, [0, 2, 1]], normals[:, [0, 2, 1]]
        write_msh(out / f"{stem}.msh", vertices[faces], normals, uv)
        parts.append((f"{stem}.msh", (1.0, 1.0, 1.0, 1.0), png))

    parts: list[tuple[str, Rgba, str]] = []
    textured = np.zeros(len(tri), bool)
    for k, (_, axis, shapes, over, skip) in enumerate(decals):
        if not shapes:
            continue
        sign, ax = (1 if axis[0] == "+" else -1), "xyz".index(axis[1])
        a, c = PLANE[axis[1]]
        points = np.array([p for s in shapes for p in decal_points(s)])
        lo2, hi2 = points.min(axis=0) - 0.003, points.max(axis=0) + 0.003
        proj = centers[:, [a, c]]
        pick = (normals[:, ax] * sign > FACING) & np.all((proj >= lo2) & (proj <= hi2), axis=1)
        pick &= np.all(colors == over, axis=1) & ~textured
        for s in skip:  # a test on the triangles, or a box: faces, or parts, centered in it
            pick &= ~(s(tri) if callable(s) else inside(part_center, s) | inside(centers, s))
        if not pick.any():
            continue
        textured |= pick
        corners = tri[pick][:, :, [a, c]]
        lo, hi = corners.reshape(-1, 2).min(axis=0), corners.reshape(-1, 2).max(axis=0)
        draw(shapes, over, lo, hi).save(out / f"{name}.decal{k}.png")
        write(f"{name}.decal{k}", mesh.faces[pick], (corners - lo) / (hi - lo), corner_n[pick], f"{name}.decal{k}.png")
    # The plain parts are textured too, each with a texture of its color: MuJoCo lights textured and untextured
    # geoms differently (specular), which would show the decals' outlines.
    caps_corners = np.concatenate([np.zeros((0, 3, 3)), *(caps(mesh, b) for _, b, _ in rules["caps"])])
    caps_colors = [color for _, b, color in rules["caps"] for _ in range(len(caps(mesh, b)))]
    n = len(vertices)
    vertices = np.concatenate([vertices, caps_corners.reshape(-1, 3) * unmirror])
    faces = np.concatenate([mesh.faces[~textured], n + np.arange(3 * len(caps_corners)).reshape(-1, 3)])
    caps_n = face_normals(caps_corners)
    normals = np.concatenate([corner_n[~textured], np.repeat(caps_n[:, None] * unmirror, 3, axis=1)])
    face_colors = [tuple(c) for c in colors[~textured]] + caps_colors
    for i, color in enumerate(dict.fromkeys(face_colors)):
        png = "color_{:02x}{:02x}{:02x}.png".format(*rgb(color))
        Image.new("RGB", (4, 4), rgb(color)).save(out / png)
        mask = np.array([c == color for c in face_colors])
        write(f"{name}.{i}", faces[mask], np.full((mask.sum(), 3, 2), 0.5), normals[mask], png)
    return parts


def paint(robot: str) -> None:
    """Paint ``robot``'s meshes as configured in ``ROBOTS`` and write them with parts.json."""
    cfg = ROBOTS[robot]
    out = ASSETS / robot / "painted"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        old.unlink()
    parts = {}
    for path in sorted((ASSETS / cfg["meshes"]).glob("*.[Ss][Tt][Ll]")):
        if "collision" not in path.stem:
            parts[path.stem] = paint_mesh(path, cfg, out)
            print(path.stem, [p[0] for p in parts[path.stem]])
    (out / "parts.json").write_text(json.dumps(parts, indent=1))


if __name__ == "__main__":
    paint(sys.argv[1])
