# FightClab

> The first rule of FightClab: you do not talk about FightClab.

An arena for humanoid robot fights, built on [mjlab](https://github.com/mujocolab/mjlab): fight rules, a physical damage model, and a league of opponents for self-play.

- **No HP.** You win by knockout (the opponent can't get up within N seconds), ring-out, or on points for clean hits.
- **Damage is physical.** Hits and falls weaken the joints they land on, and motors overheat.
- **Reward** = clean hits on the opponent minus damage to yourself.

Status: early. Unitree G1 first, other humanoids later.

## Setup

Linux or WSL2 with an NVIDIA GPU, plus [uv](https://docs.astral.sh/uv/).

```bash
uv sync
# smoke test: 3 iterations of mjlab's G1 walking task
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024 --agent.max-iterations 3 --agent.logger tensorboard
```

## Docker

Requires `docker compose` and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The repo is mounted into the container, so you only need to rebuild when dependencies change.

```bash
./docker/run.sh --build   # build the image and open a shell
./docker/run.sh uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024 --agent.max-iterations 3 --agent.logger tensorboard
```

## Development

```bash
uvx pre-commit install
```
