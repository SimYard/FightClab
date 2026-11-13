#!/usr/bin/env bash
# Train the tracker on footwork, jabs and the rear-hand catch, on any Linux machine with an NVIDIA driver.
#
#   tools/train_catch.sh CHECKPOINT.pt [ITERATIONS=1500]     (ENVS=4096 by default; fewer on a smaller GPU)
#
# Run it from a clone of the repo. It installs uv if needed, syncs the environment, generates the
# training motion (~/fw_clips/steps_t800.*, about 600 s of moves, jabs and catches) and resumes training from
# CHECKPOINT, a policy of Fightclab-K1-Makiwara-T800 or Fightclab-K1-Footwork-Tracking-T800 (policies aren't in git).
# One from before these tasks saw an opponent (140 actor inputs) loads only widened by tools/expand_inputs.py.
set -euo pipefail
ckpt=$(readlink -f "$1")
iterations=${2:-1500}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl WANDB_MODE=disabled
uv sync
mkdir -p ~/fw_clips
uv run python -m fightclab.tasks.k1.steps ~/fw_clips/steps_t800.csv random:600
uv run python -m fightclab.tasks.helicopter_kick.motion_npz t800 ~/fw_clips/steps_t800.csv ~/fw_clips/steps_t800.npz
# rsl_rl resumes from a run directory under the experiment's logs.
start=logs/rsl_rl/t800_k1_footwork_tracking/start
mkdir -p "$start"
cp "$ckpt" "$start/"
uv run train Fightclab-K1-Makiwara-T800 --env.scene.num-envs "${ENVS:-4096}" --agent.max-iterations "$iterations" \
    --agent.logger tensorboard --agent.run-name catch --agent.resume True --agent.load-run start \
    --agent.load-checkpoint "$(basename "$ckpt")"
