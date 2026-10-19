"""GVHMR result -> robot motion CSV (base pos, base quat xyzw, joints) through GMR, without GMR's viewer.

Runs in GMR's own environment, not FightClab's:
    ~/GMR/.venv/bin/python tools/gvhmr_to_csv.py <g1|r1|h2|pm01|t800|t1|asimov> <hmr4d_results.pt> <out.csv>
Rows are at 30 fps of the clip. G1's joint order is mjlab's csv_to_npz order (Unitree SDK order); the others' is the
order of their model.

GMR has no R1 or H2, so they are registered here: a mocap model built from Unitree's MJCF, and an IK config adapted
from G1's. Their links are named and oriented like G1's, so G1's matching carries over with rescaled proportions.
"""

import json
import pathlib
import sys
import tempfile
from dataclasses import dataclass

import mujoco
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting, params
from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

GMR = pathlib.Path.home() / "GMR"
ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets/robots"  # run in GMR's venv, without fightclab


@dataclass(frozen=True)
class Extra:
    """A robot GMR lacks, described relative to G1."""

    xml: pathlib.Path
    foot: str
    """The last body of each leg, without the side prefix; GMR matches the human foot to a toe body in it."""
    toe: tuple[float, float, float]
    """Toe position in the left foot body: like G1's, 2 cm behind and 1 cm above the front foot contacts."""
    scale: dict[str, float]
    """Lengths over G1's (0.656, 0.233, 0.292 m): hip-ankle (legs), shoulder-wrist (arms), pelvis-shoulder (trunk)."""
    hand: str = "wrist_yaw_link"
    placeholders: tuple[str, ...] = ()
    """Bodies to drop."""
    straight_hips: bool = False
    """Hip roll frames without G1's 10 degree pitch."""
    renames: tuple[tuple[str, str], ...] = ()
    """G1 bodies named otherwise on the robot."""
    arm_rotations: bool = True
    """False where elbow and wrist frames are turned against G1's: match their positions only."""


EXTRAS = {
    # Unitree parks R1's placeholders at z = 20 m so the SDK sees 29 joints; R1 has no waist pitch or wrist pitch/yaw.
    "r1": Extra(
        ASSETS / "r1/R1_C++.xml",
        "ankle_roll_link",
        (0.095, 0.0, -0.035),
        {"legs": 0.91, "arms": 1.10, "trunk": 0.84},  # 0.599, 0.256, 0.246 m
        hand="wrist_roll_link",
        placeholders=("waist_pitch", "left_wrist_pitch", "left_wrist_yaw", "right_wrist_pitch", "right_wrist_yaw"),
    ),
    "h2": Extra(
        ASSETS / "h2/h2_mujoco.xml",
        "ankle_pitch_link",
        (0.13, 0.036, -0.04),
        {"legs": 1.43, "arms": 1.34, "trunk": 1.65},  # 0.938, 0.312, 0.483 m
        straight_hips=True,
    ),
    "asimov": Extra(
        ASSETS / "asimov/xmls/asimov_1.xml",
        "ankle_roll_link",
        (0.102, 0.0, -0.019),
        {"legs": 0.84, "arms": 1.36, "trunk": 1.15},  # 0.554, 0.317, 0.336 m
        straight_hips=True,
        renames=(("pelvis", "pelvis_link"), ("torso_link", "waist_yaw_link")),
        arm_rotations=False,  # its elbow and wrist frames are turned 45 degrees
    ),
}


def mocap(e: Extra, name: str) -> pathlib.Path:
    """The robot for GMR: without placeholder bodies, with toe bodies where G1's mocap model has them."""
    spec = mujoco.MjSpec.from_file(str(e.xml))
    spec.meshdir = str(e.xml.parent / spec.meshdir)
    gone = {spec.body(b).first_joint().name for b in e.placeholders}
    for b in e.placeholders:
        spec.delete(spec.body(b))
    for element in [*spec.actuators, *spec.sensors]:
        if getattr(element, "target", None) in gone or getattr(element, "objname", None) in gone:
            spec.delete(element)
    x, y, z = e.toe
    for side, sign in (("left", 1), ("right", -1)):
        spec.body(f"{side}_{e.foot}").add_body(name=f"{side}_toe_link", pos=[x, sign * y, z])
    out = pathlib.Path(tempfile.gettempdir()) / f"{name}_mocap.xml"
    out.write_text(spec.to_xml())
    return out


def ik_config(e: Extra, name: str) -> pathlib.Path:
    """G1's SMPL-X config with the robot's hand bodies and proportions."""
    cfg = json.loads((GMR / "general_motion_retargeting/ik_configs/smplx_to_g1.json").read_text())
    for table in ("ik_match_table1", "ik_match_table2"):
        for side in ("left", "right"):
            cfg[table][f"{side}_{e.hand}"] = cfg[table].pop(f"{side}_wrist_yaw_link")
            if e.straight_hips:
                cfg[table][f"{side}_hip_roll_link"][4] = cfg[table][f"{side}_knee_link"][4]
            if not e.arm_rotations:  # GMR drops a task with both weights 0, then misses its offsets
                for body in (f"{side}_elbow_link", f"{side}_{e.hand}"):
                    cfg[table][body][2] = 1e-3
        for g1_body, body in e.renames:
            cfg[table][body] = cfg[table].pop(g1_body)
    cfg["robot_root_name"] = dict(e.renames).get("pelvis", "pelvis")
    arms = ("shoulder", "elbow", "wrist")
    for joint in cfg["human_scale_table"]:
        part = "trunk" if joint in ("pelvis", "spine3") else "arms" if any(a in joint for a in arms) else "legs"
        cfg["human_scale_table"][joint] *= e.scale[part]
    out = pathlib.Path(tempfile.gettempdir()) / f"smplx_to_{name}.json"
    out.write_text(json.dumps(cfg))
    return out


robot, pred_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
native = {"g1": "unitree_g1", "pm01": "engineai_pm01", "t800": "engineai_t800", "t1": "booster_t1"}
target = native.get(robot, f"fightclab_{robot}")
if robot == "t800":  # from EngineAI's GMR fork (github.com/engineai-robotics/GMR), not in GMR
    params.ROBOT_XML_DICT[target] = ASSETS / "t800/t800_mocap.xml"
    params.IK_CONFIG_DICT["smplx"][target] = ASSETS / "t800/smplx_to_t800.json"
    params.ROBOT_BASE_DICT[target] = "LINK_BASE"
    params.VIEWER_CAM_DISTANCE_DICT[target] = 2.0
if robot in EXTRAS:
    params.ROBOT_XML_DICT[target] = mocap(EXTRAS[robot], robot)
    params.IK_CONFIG_DICT["smplx"][target] = ik_config(EXTRAS[robot], robot)
    params.ROBOT_BASE_DICT[target] = dict(EXTRAS[robot].renames).get("pelvis", "pelvis")
    params.VIEWER_CAM_DISTANCE_DICT[target] = 2.0
smplx_folder = GMR / "assets" / "body_models"
smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(pred_file, smplx_folder)
frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
retarget = GeneralMotionRetargeting(actual_human_height=height, src_human="smplx", tgt_robot=target)

rows = []
for frame in frames:
    q = retarget.retarget(frame)
    rows.append(np.concatenate([q[:3], q[[4, 5, 6, 3]], q[7:]]))  # wxyz -> xyzw
np.savetxt(out, np.array(rows), delimiter=",", fmt="%.6f")
print(f"human height {height:.2f} m, {len(rows)} frames at {fps} fps -> {out}")
