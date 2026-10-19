# FightClab

> The first rule of FightClab: you do not talk about FightClab.

An arena for humanoid robot fights, built on [mjlab](https://github.com/mujocolab/mjlab): fight rules, a physical damage model, and a league of opponents for self-play.

- **No HP.** You win by knockout (the opponent can't get up within N seconds), ring-out, or on points for clean hits.
- **Damage is physical.** Hits and falls weaken the joints they land on, and motors overheat.
- **Reward** = clean hits on the opponent minus damage to yourself.

Status: early. The robots below have learned a jumping spin kick from a video; fights come next.

## Robots

Models are kept as their makers publish them, each folder with its maker's license file.

| Robot | In this repo | Taken from | License |
|---|---|---|---|
| Unitree G1 | (mjlab's) | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | Apache-2.0 |
| Unitree R1 | `assets/robots/r1` | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) `unitree_robots/r1` @ `1eb6642` | BSD-3-Clause |
| Unitree H2 | `assets/robots/h2` | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) `unitree_robots/h2` @ `1eb6642` | BSD-3-Clause |
| EngineAI PM01 | `assets/robots/pm01` | MJCF: [YanjieZe/GMR](https://github.com/YanjieZe/GMR) `assets/engineai_pm01` @ `bb1bbe4`; textured meshes: [engineai-robotics/engineai_robotics_description](https://github.com/engineai-robotics/engineai_robotics_description) `pm01_edu` @ `1d2d760` | BSD-3-Clause; textured meshes: no license stated |
| EngineAI T800 | `assets/robots/t800` | [engineai-robotics/GMR](https://github.com/engineai-robotics/GMR) `assets/engineai_t800` and `ik_configs/smplx_to_t800.json` @ `71e02c0` | MIT |
| Booster T1 | `assets/robots/t1` | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) `assets/booster_t1` @ `bb1bbe4`, from [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets) | BSD-3-Clause |
| Menlo Asimov v1 | `assets/robots/asimov` | [menloresearch/asimov-1](https://github.com/menloresearch/asimov-1) `sim-model` @ `f71f3fe` | CERN-OHL-S-2.0 |

Left out: PM01's one-grey STL meshes (its textured meshes replace them) and T800's DAE copies. Converted: the OBJ meshes
of PM01 and T800, to MuJoCo's binary MSH by `tools/obj_to_msh.py` (same vertices, texture coordinates and normals; they
render pixel for pixel the same). Added: R1's colors, `assets/robots/r1/painted`, made from Unitree's product photos by
`tools/paint_meshes.py`.

## License

FightClab's code is under the [Apache License 2.0](LICENSE). The robot models in `assets/robots` keep their makers'
licenses, listed above.

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
