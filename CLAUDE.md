# FightClab: notes for Claude

Talk to the user in Russian. They prefer minimal solutions.

## What the project is

An arena for humanoid robot fights in [mjlab](https://github.com/mujocolab/mjlab): fight rules, a physical damage model, and a league of opponents for self-play. Decisions so far:

- **Simulator:** mjlab only, not Isaac Lab.
- **Robots:** Unitree G1 first. Other humanoids come later as configs, with retargeting through GMR.
- **No HP.** A fight is won by:
  - knockout: the opponent can't get up within N seconds;
  - ring-out;
  - points for clean hits.
- **Clean hit.** A striking body (fist, foot, knee, elbow) reaches a target zone (head, torso, legs) with impulse or relative velocity above a threshold. Pushes score zero.
- **Damage is physical.** Hits and falls lower the torque limit of the affected joints or lock them, and motors overheat (I²t with cooling). Optional: a short "stun" (IMU noise or action delay) after a strong head or torso hit.
- **Reward:** win or loss, plus shaping for clean hits, minus damage to yourself.
- **Calibration.** All damage coefficients go in config. Real hardware breakage thresholds are unknown, so they need calibration.
- **Planned stack:**
  1. Mocap: LAFAN1 fight clips, AMASS, and GVHMR for clips from video.
  2. Retargeting with GMR.
  3. A tracking policy in the BeyondMimic style.
  4. A skill latent space in the ASE / PULSE / MaskedMimic style.
  5. A high-level self-play policy that acts over the latents.

  Without a skill prior, self-play degenerates into pushing.

**Next step:** task `Fightclab-Duel-G1`, a ring with two G1s facing each other.
- mjlab discovers external tasks through the entry point group `mjlab.tasks`: add `[project.entry-points."mjlab.tasks"]` to `pyproject.toml` and call `register_mjlab_task` (see `~/mjlab/src/mjlab/tasks/registry.py`).
- rsl_rl is single-agent, so the two-robot and self-play code is ours to write.

## Layout

The layout follows the user's template [AlexanderRex/project_template](https://github.com/AlexanderRex/project_template), with these changes the user chose:
- base image `nvidia/cuda` instead of ROS;
- `uv` + `uv.lock` instead of pip and requirements files;
- no X11 forwarding: rendering is headless via EGL;
- no `defaults.cfg`;
- GPU is set up in the main `docker/docker-compose.yml`, so there is no `run-nvidia.sh`.

`.claude/agents/` holds style coaches (`boxing-coach`, `k1-coach`): give them rendered frames or recorded attempts; they say what is wrong and turn it into measurable checks. One coach per style; add one when a style's techniques are trained. Each coach judges everything by its own style; where they disagree, we choose which checks a robot trains on. `humanoid-rl` reviews training setups instead (configs, rewards, curricula, motion references, logs, metrics) and answers with problems and checks.

`docker/run.sh [--build] [cmd...]` runs `cmd` in the container, or opens a shell if no command is given. The repo is mounted at `/home/dev/fightclab`. The venv lives in the image at `/home/dev/venv`. The container user `dev` has uid 1000, the same as the host user.

## Environment

- **Where the repo lives.** In WSL at `/home/alexthunderrex/FightClab` (Ubuntu-22.04). `C:\Users\alexk\FightClab` is only a Windows symlink to `\\wsl.localhost\Ubuntu-22.04\home\alexthunderrex\FightClab`. Keep the real files in WSL: `.venv` is 6 GB, and access across the Windows/WSL boundary is slow.
- **Running commands from Windows Git Bash:**
  ```bash
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 --exec bash -lc 'export PATH=$HOME/.local/bin:$PATH; cd ~/FightClab && ...'
  ```
  - Use `--exec`. Without it, wsl re-parses the arguments through a shell and quoting breaks.
  - `uv` is in `~/.local/bin`, which is not on PATH in non-interactive shells.
- **Writing files.** Use Write/Edit on the UNC path. Git Bash collapses `\\`, so never write backslash paths through shell heredocs.
- **Hardware:**
  - RTX 4090 Laptop GPU with 16 GB;
  - native Docker 29 inside WSL, not Docker Desktop;
  - `sudo` requires a password, so the user runs anything that needs root.
- **mjlab reference clone:** `~/mjlab`. It has its own `AGENTS.md` and docs under `docs/source`.

## Commands

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024 --agent.max-iterations 3 --agent.logger tensorboard  # smoke test, ~1 min on first run
uvx pre-commit run --files <files>   # hooks: ruff, mypy --strict, hadolint, pretty-format-toml (it sorts pyproject keys)
docker build -f docker/Dockerfile -t fightclab .
```

## Open items

- **GPU in Docker and `docker compose` don't work yet.** The user has to install `nvidia-container-toolkit` and `docker-compose-v2` with sudo, then run `sudo nvidia-ctk runtime configure --runtime=docker`. Verify with `./docker/run.sh nvidia-smi`.
- **Git.** The remote is `github.com/SimYard/FightClab` (public). Nothing has been committed yet. WSL git has no `user.name` or `user.email` and no push credentials. Ask the user before committing or pushing, and ask which author identity to use.
