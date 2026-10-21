"""K-1 footwork for T800: mjlab's velocity task on flat ground, styled by footwork tutorial videos (AMP).

Hand-made stance rewards (width, crossing, pivots, hops) produced footwork that broke elsewhere, so the style now
comes from clips: GVHMR, ``tools/gvhmr_to_csv.py t800``, ``video_motion ... --ground`` and ``motion_npz ... <out.npz>``
on footwork drills, judged by the discriminator in ``amp.py``. The task reward is only velocity tracking, with
moves commanded toward the opponent, 30 degrees off the pelvis (``OPPONENT_YAW``), plus mjlab's regularizers.
Low kicks hit either shin at random during training, once the footwork is there, so the stance learns to stay up
under them.

T800 starts in the clips' median pose; actions are offsets from it.
"""

import math
from dataclasses import replace
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from fightclab.robots import t800
from fightclab.tasks.k1 import amp

# Joint angles (rad) of the stance; the pelvis stands at STANCE_HEIGHT, pitched STANCE_PITCH forward. The clips'
# median pose, so that resets start where the discriminator's data is: the earlier hand-tuned stance (FightCamp's
# tutorial after the K-1 coach's review) was 2-4 standard deviations off the clips in the lead hip's yaw and the arms.
STANCE = {
    "J00_HIP_PITCH_L": -0.596, "J01_HIP_ROLL_L": 0.253, "J02_HIP_YAW_L": 0.006, "J03_KNEE_PITCH_L": 0.477,
    "J04_ANKLE_PITCH_L": -0.119, "J05_ANKLE_ROLL_L": -0.063, "J06_HIP_PITCH_R": -0.224, "J07_HIP_ROLL_R": -0.143,
    "J08_HIP_YAW_R": -0.069, "J09_KNEE_PITCH_R": 0.436, "J10_ANKLE_PITCH_R": -0.330, "J11_ANKLE_ROLL_R": -0.062,
    "J12_TORSO_YAW": 0.195, "J13_SHOULDER_PITCH_L": -0.564, "J14_SHOULDER_ROLL_L": 0.534,
    "J15_SHOULDER_YAW_L": 0.397, "J16_ELBOW_PITCH_L": -2.286, "J17_ELBOW_YAW_L": 0.323,
    "J18_SHOULDER_PITCH_R": -0.833, "J19_SHOULDER_ROLL_R": -0.164, "J20_SHOULDER_YAW_R": 0.132,
    "J21_ELBOW_PITCH_R": -2.248, "J22_ELBOW_YAW_R": -0.419, "J23_HEAD_PITCH": 0.0, "J24_HEAD_YAW": 0.0,
}  # fmt: skip
STANCE_HEIGHT = 0.996
STANCE_PITCH = math.radians(8.4)
FEET = ("LINK_FOOT_L", "LINK_FOOT_R")
# Orthodox K-1: the pelvis faces about 30 degrees right of the opponent; velocity commands are in this frame.
OPPONENT_YAW = math.radians(30.0)
# The footwork clips: guarded runs (``python -m fightclab.tasks.k1.amp``) of Gabriel Varga's "Footwork Training For
# Kickboxing" (Glory, the K-1 reference) and Coach Mark's "10-Minute Muay Thai Footwork Follow-Along" (more footage,
# but a squarer, upright Muay Thai stance), shared half and half as the K-1 coach advised.
# ponytail: kept outside the repo (~100 MB); ship the features if others train.
CLIPS = Path.home() / "fw_clips" / "runs"
SOURCES = {str(CLIPS / "k1_vg_*.npz"): 0.5, str(CLIPS / "k1_fw_*.npz"): 0.5}
# A low kick on a shin: up to 500 N per axis for 0.05-0.1 s, i.e. up to ~50 N s, a 6 kg shin at ~8 m/s.
# ponytail: guessed from human kicks; calibrate against the arena's own kicks once they land.
LOW_KICK = {"force_range": (-500.0, 500.0), "duration_s": (0.05, 0.1), "cooldown_s": (1.5, 4.0)}
# The kicks grow from nothing to full force between these PPO iterations: footwork first, then staying up under
# kicks. From the start they kept the first AMP policies busy staying up; from iteration 500, when the policy only
# began to move, its style score fell from 0.50 to 0.29 as they grew.
KICKS_FROM, KICKS_FULL = 2000, 3000


def get_spec() -> mujoco.MjSpec:
    """T800 with mjlab's velocity-task extras: sole sites and the root's angular momentum sensor, named as on G1."""
    spec = t800.get_spec()
    for body, site in zip(FEET, ("left_foot", "right_foot"), strict=True):
        spec.body(body).add_site(name=site, pos=(0.02, 0.0, -0.064))
    spec.add_sensor(
        name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM, objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="LINK_BASE",
    )  # fmt: skip
    return spec


class track_smoothed_velocity:
    """Velocity tracking on the pelvis' velocity low-passed over ``tau`` seconds, the command in the opponent's frame.

    Footwork steps and stops: its instantaneous speed swings between 0 and about twice the mean, and tracked as is,
    the clips' own footwork scored no better than standing still. Params: ``command_name``, ``std``, ``tau``,
    ``yaw`` (the opponent's direction left of the pelvis' x) and ``part``: "linear" (vx, vy) or "angular" (wz).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        """Start every robot still."""
        self.v = torch.zeros(env.num_envs, 3, device=env.device)
        self.alpha = env.step_dt / cfg.params["tau"]

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """A reset robot stands still."""
        self.v[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(  # noqa: PLR0917
        self, env: ManagerBasedRlEnv, command_name: str, std: float, tau: float, yaw: float, part: str
    ) -> torch.Tensor:
        """exp(-error^2 / std^2) of the smoothed velocity against the command."""
        del tau  # Used at init.
        data = env.scene["robot"].data
        v, wz = data.root_link_lin_vel_b, data.root_link_ang_vel_b[:, 2]
        now = torch.stack((math.cos(yaw) * v[:, 0] + math.sin(yaw) * v[:, 1],
                           -math.sin(yaw) * v[:, 0] + math.cos(yaw) * v[:, 1], wz), -1)  # fmt: skip
        self.v += self.alpha * (now - self.v)
        command = env.command_manager.get_command(command_name)
        assert command is not None
        error = (command - self.v).square()
        error = error[:, :2].sum(-1) if part == "linear" else error[:, 2]
        return torch.exp(-error / std**2)


def still_head_action_scale() -> dict[str, float]:
    """T800's action scale with the head joints at 0: their targets stay at the default pose, the head held still."""
    scale = dict(t800.T800_ACTION_SCALE)
    scale["J22_.*"] = scale.pop("J2[2-4]_.*")
    return {**scale, "J2[34]_.*": 0.0}


def low_kick_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, start: int, full: int) -> torch.Tensor:
    """Scale the low kicks' force from 0 at env step ``start`` to LOW_KICK's at ``full``; returns the scale."""
    del env_ids  # Unused.
    scale = min(max((env.common_step_counter - start) / (full - start), 0.0), 1.0)
    for side in ("L", "R"):
        params = env.event_manager.get_term_cfg(f"low_kick_{side}").params
        params["force_range"] = tuple(scale * f for f in LOW_KICK["force_range"])
    return torch.tensor(scale)


def t800_footwork_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Mjlab's flat G1 velocity task on T800, styled by the footwork clips."""
    cfg = unitree_g1_flat_env_cfg(play=play)
    robot = t800.get_t800_cfg()
    robot = replace(
        robot,
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, STANCE_HEIGHT),
            rot=(math.cos(STANCE_PITCH / 2), 0.0, math.sin(STANCE_PITCH / 2), 0.0),
            joint_pos=STANCE,
            joint_vel={".*": 0.0},
        ),
    )
    cfg.scene.entities = {"robot": robot}
    sensors = []
    for sensor in cfg.scene.sensors or ():
        renamed = sensor
        if isinstance(sensor, ContactSensorCfg) and sensor.name == "feet_ground_contact":
            feet = replace(sensor.primary, pattern=r"^(LINK_ANKLE_ROLL_L|LINK_ANKLE_ROLL_R)$")
            renamed = replace(sensor, primary=feet)
        elif isinstance(sensor, ContactSensorCfg) and sensor.name == "self_collision":
            pelvis = replace(sensor.primary, pattern="LINK_BASE")
            renamed = replace(sensor, primary=pelvis, secondary=pelvis)
        sensors.append(renamed)
    cfg.scene.sensors = tuple(sensors)
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    # The head stays still: the clips have no head motion (GMR doesn't retarget it), and the policy's idle head turns
    # were what the discriminator told it apart by.
    action.scale = still_head_action_scale()
    cfg.viewer.body_name = "LINK_WAIST_YAW"
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 4)
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_WAIST_YAW",)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("LINK_WAIST_YAW",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("LINK_WAIST_YAW",)

    # Commands the clips can show: their pelvis' speed averaged over 3 s stays under ~0.3 m/s and their turns under
    # ~0.3 rad/s (95th percentiles). Faster commands made the style reward fight the task.
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (-0.3, 0.3)
    cfg.curriculum.pop("command_vel", None)  # it overwrites these ranges with walking speeds (vx up to 3 m/s)
    # Mjlab's tracking widths suit walking at up to 3 m/s; at footwork speeds standing still earned half the reward.
    # With std 0.15, standing earns 0.02 against a 0.3 m/s command.
    for name, part in (("track_linear_velocity", "linear"), ("track_angular_velocity", "angular")):
        track = cfg.rewards[name]
        track.func = track_smoothed_velocity
        track.params = {"command_name": "twist", "std": 0.15, "tau": 1.0, "yaw": OPPONENT_YAW, "part": part}
    # The clips' guard folds the elbows to their stops (GMR pins them there in 53-91% of frames), past the soft
    # limits: the limit penalty leaves them alone.
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(?!.*ELBOW_PITCH).*",))
    # The kicks below are the disturbance. Mjlab's pushes (+-0.5 m/s every 1-3 s from the first iteration, ~35 N s on
    # T800) were bigger than a full kick, and a one-step velocity jump no clip contains.
    cfg.events.pop("push_robot", None)

    # The clips set the style: no posture or swing shaping.
    for name in ("pose", "foot_clearance", "foot_swing_height", "air_time"):
        del cfg.rewards[name]
    cfg.rewards["style"] = RewardTermCfg(func=amp.style, weight=4.0, params={"sources": SOURCES, "train": not play})
    if not play:
        cfg.events["reset_to_clips"] = EventTermCfg(
            func=amp.reset_to_clips, mode="reset", params={"sources": SOURCES, "share": 0.5}
        )
        add_low_kicks(cfg, t800_footwork_ppo_runner_cfg().num_steps_per_env)
    return cfg


def add_low_kicks(cfg: ManagerBasedRlEnvCfg, steps_per_iteration: int) -> None:
    """Low kicks on T800's shins, growing from KICKS_FROM to KICKS_FULL iterations into a run (resumed runs too)."""
    for side in ("L", "R"):
        cfg.events[f"low_kick_{side}"] = EventTermCfg(
            func=apply_body_impulse,
            mode="step",
            params={
                **LOW_KICK, "torque_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=(f"LINK_KNEE_PITCH_{side}",)),
            },
        )  # fmt: skip
    cfg.curriculum["low_kicks"] = CurriculumTermCfg(
        func=low_kick_curriculum,
        params={"start": KICKS_FROM * steps_per_iteration, "full": KICKS_FULL * steps_per_iteration},
    )


def t800_footwork_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Mjlab's G1 velocity PPO, logged as T800 K-1 footwork."""
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.experiment_name = "t800_k1_footwork"
    return cfg
