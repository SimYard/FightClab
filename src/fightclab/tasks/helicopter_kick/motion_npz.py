"""Motion CSV -> the tracking task's npz, for any of our robots.

    MUJOCO_GL=egl WANDB_MODE=disabled uv run python -m fightclab.tasks.helicopter_kick.motion_npz ROBOT IN.csv [OUT.npz]

Replays the 30 fps CSV at 50 fps in the robot's tracking scene with mjlab's csv_to_npz, which only knows G1 itself,
and copies the result over the kick task's motion file, or to ``OUT.npz``.
"""

import shutil
import sys
from pathlib import Path

from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import run_sim
from mjlab.sim import Simulation, SimulationCfg

from fightclab.tasks.helicopter_kick import tracking
from fightclab.tasks.helicopter_kick.video_motion import FPS, ROBOTS

TASKS = {
    "g1": (tracking.g1_helicopter_kick_tracking_env_cfg, tracking.MOTION),
    "r1": (tracking.r1_helicopter_kick_tracking_env_cfg, tracking.MOTION_R1),
    "h2": (tracking.h2_helicopter_kick_tracking_env_cfg, tracking.MOTION_H2),
    "pm01": (tracking.pm01_helicopter_kick_tracking_env_cfg, tracking.MOTION_PM01),
    "t800": (tracking.t800_helicopter_kick_tracking_env_cfg, tracking.MOTION_T800),
    "t1": (tracking.t1_helicopter_kick_tracking_env_cfg, tracking.MOTION_T1),
    "asimov": (tracking.asimov_helicopter_kick_tracking_env_cfg, tracking.MOTION_ASIMOV),
}

if __name__ == "__main__":
    robot, csv = sys.argv[1:3]
    env_cfg, motion = TASKS[robot]
    motion = Path(sys.argv[3]) if sys.argv[3:] else motion
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / 50
    scene = Scene(env_cfg().scene, device="cuda:0")
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device="cuda:0")
    scene.initialize(sim.mj_model, sim.model, sim.data)
    run_sim(
        sim=sim,
        scene=scene,
        joint_names=list(ROBOTS[robot].joints),
        input_file=csv,
        input_fps=FPS,
        output_fps=50,
        output_name=motion.stem,
        render=False,
        line_range=None,
    )
    shutil.copy("/tmp/motion.npz", motion)
    print(motion)
