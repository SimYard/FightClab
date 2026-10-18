"""GVHMR result -> G1 motion CSV (base pos, base quat xyzw, 29 joints) through GMR, without GMR's viewer.

Runs in GMR's own environment, not FightClab's:
    ~/GMR/.venv/bin/python tools/gvhmr_to_g1_csv.py <hmr4d_results.pt> <out.csv>
GMR's G1 joint order is the same as mjlab's csv_to_npz (Unitree SDK order). Rows are at 30 fps of the clip.
"""

import pathlib
import sys

import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

pred_file, out = sys.argv[1], sys.argv[2]
smplx_folder = pathlib.Path.home() / "GMR" / "assets" / "body_models"
smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(pred_file, smplx_folder)
frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
retarget = GeneralMotionRetargeting(actual_human_height=height, src_human="smplx", tgt_robot="unitree_g1")

rows = []
for frame in frames:
    q = retarget.retarget(frame)
    rows.append(np.concatenate([q[:3], q[[4, 5, 6, 3]], q[7:]]))  # wxyz -> xyzw
np.savetxt(out, np.array(rows), delimiter=",", fmt="%.6f")
print(f"human height {height:.2f} m, {len(rows)} frames at {fps} fps -> {out}")
