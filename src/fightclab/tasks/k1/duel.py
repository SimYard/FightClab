"""A second T800 in a tracking task, moved by a frozen tracker on its own motion, frame for frame with the learner's.

rsl_rl trains one policy per env, so the opponent is part of the env, like the makiwara: a zero-dim action term runs
a TorchScript copy of a trained tracker (``tools/export_policy.py``) on the actor's observations read off the opponent,
and its motion command keeps the learner's frame. Its observations carry no noise: a scripted bot.
"""

import copy
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.spec_config import CollisionCfg

from fightclab.robots.common import STIFF_COLLISION
from fightclab.tasks.k1 import makiwara

ENTITY = "opponent"  # also its action term's and observation group's name
COMMAND = "opponent_motion"
HEAD_HITS = "opponent_fist_on_head"
# m between the robots until the first reset puts the opponent on its motion: one inside the other, their 76 contacts
# overflowed the solver's buffers.
APART = 2.0
# What the robots touch each other with: capsules and boxes fitted to the upper body's visual meshes
# (``tools/fit_primitives_t800.py`` writes them; derived data, not in git). The vendor's collision primitives are much
# smaller (the head an 18 cm ball for a 21 x 16 x 22 cm helmet, a forearm a 7 x 7 x 16 cm brick for 14 x 10 x 26 cm)
# and frictionless (condim 1): a fist slid up a glove into the forehead. The trackers learned the floor and their own
# bodies on those primitives, so those stay for that.
SHELLS = Path.home() / "fw_clips" / "t800_primitives.json"
STRIKERS = re.compile("ELBOW_YAW")  # the forearms, the fist's end included (the robot has no hands)
SHELL_FRICTION = 0.5  # no data for plastic shells; as the floor's rubber, to calibrate
SHELL_GROUP = 3  # not drawn
# contype bits of each robot's strikers and targets. A striker meets the other robot's strikers and targets, a
# target only its strikers: no head on head, no robot on itself or the floor (those are the vendor's primitives).
STRIKE, TARGET = {"robot": 2, ENTITY: 8}, {"robot": 4, ENTITY: 16}


class FollowerMotionCommand(MotionCommand):
    """A motion on another entity, frame for frame with the ``leader`` command's (the motions as long)."""

    cfg: "FollowerMotionCommandCfg"
    _pending_forward: bool

    @property
    def leader(self) -> MotionCommand:
        """The command whose frame this one keeps."""
        term = self._env.command_manager.get_term(self.cfg.leader)
        assert isinstance(term, MotionCommand)
        return term

    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        """At a reset, or where the leader's motion wrapped around: the leader's frame."""
        assert self.motion.time_step_total == self.leader.motion.time_step_total, "the motions must be as long"
        self.time_steps[env_ids] = self.leader.time_steps[env_ids]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """One frame on, then the leader's frame wherever the leader jumped."""
        self.time_steps[slice(None) if env_ids is None else env_ids] += 1
        jumped = (self.time_steps != self.leader.time_steps).nonzero().flatten()
        if len(jumped) > 0:
            self._resample_command(jumped)
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()


@dataclass(kw_only=True)
class FollowerMotionCommandCfg(MotionCommandCfg):
    """``MotionCommandCfg`` and the ``leader`` command's name; sampling_mode "uniform" copies the leader's frame."""

    leader: str

    def build(self, env: ManagerBasedRlEnv) -> FollowerMotionCommand:
        """The command."""
        return FollowerMotionCommand(self, env)


class OpponentAction(ActionTerm):
    """The opponent's joints, moved by a frozen policy on the env's observation group of the same name."""

    cfg: "OpponentActionCfg"

    def __init__(self, cfg: "OpponentActionCfg", env: ManagerBasedRlEnv) -> None:
        """The opponent's joint action and the policy, which fails loudly if its file is missing."""
        super().__init__(cfg, env)
        self.joints = cfg.joints.build(env)
        self.policy = torch.jit.load(cfg.policy, map_location=env.device)

    @property
    def action_dim(self) -> int:
        """None of the learner's actions."""
        return 0

    @property
    def raw_action(self) -> torch.Tensor:
        """The policy's last actions."""
        return self.joints.raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        """The policy's actions on the opponent's latest observations."""
        del actions  # none: the policy acts
        with torch.no_grad():
            self.joints.process_actions(self.policy(self._env.obs_buf[ENTITY]))

    def apply_actions(self) -> None:
        """The joint targets."""
        self.joints.apply_actions()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """No last actions."""
        self.joints.reset(env_ids)


@dataclass(kw_only=True)
class OpponentActionCfg(ActionTermCfg):
    """The opponent's joint action (the learner's, on the opponent) and its policy's file."""

    joints: JointPositionActionCfg
    policy: str  # a TorchScript actor: the actor's observations in, actions out

    def build(self, env: ManagerBasedRlEnv) -> OpponentAction:
        """The action term."""
        return OpponentAction(self, env)


def opponent_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The opponent's ``last_action`` observation."""
    return env.action_manager.get_term(ENTITY).raw_action


def opponent_observations(actor: ObservationGroupCfg) -> ObservationGroupCfg:
    """The actor's observations read off the opponent and its command, without noise."""
    group = copy.deepcopy(actor)
    group.enable_corruption = False
    for name, term in group.terms.items():
        if term.func.__name__ == "last_action":
            term.func, term.params = opponent_last_action, {}
            continue
        signature, pointed = inspect.signature(term.func).parameters, False
        if "command_name" in term.params:
            term.params["command_name"], pointed = COMMAND, True
        if "sensor_name" in term.params:
            term.params["sensor_name"], pointed = f"{ENTITY}/{term.params['sensor_name'].split('/', 1)[1]}", True
        if "asset_cfg" in signature:
            asset = term.params.get("asset_cfg", signature["asset_cfg"].default)
            term.params["asset_cfg"], pointed = replace(asset, name=ENTITY), True
        if not pointed:
            raise ValueError(f"the actor's {name} term reads nothing known to point at the opponent")
    return group


def fist_on_head(substeps: int) -> ContactSensorCfg:
    """The opponent's net force on the learner's head, per physics substep of a policy step.

    Its whole body, not its fist's ball: with no hands, the jab lands with the forearm's end as well.
    """
    return ContactSensorCfg(
        name=HEAD_HITS,
        primary=ContactMatch(mode="body", pattern="LINK_HEAD_.*", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity=ENTITY),
        fields=("found", "force"),
        reduce="netforce",
        history_length=substeps,
    )


def head_hits(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The impulse of the opponent's fist into the head this step, in makiwara hits (``makiwara.IMPULSE_SCALE``)."""
    force = env.scene[HEAD_HITS].data.force_history  # (envs, head bodies, substeps, 3)
    return torch.linalg.norm(force, dim=-1).sum(dim=(1, 2)) * makiwara.PHYSICS_DT / makiwara.IMPULSE_SCALE


def _apart(spec_fn: Callable[[], mujoco.MjSpec]) -> mujoco.MjSpec:
    """The robot's spec with its root APART along x in the model's default pose, not only in the keyframe."""
    spec = spec_fn()
    spec.worldbody.first_body().pos[0] += APART
    return spec


def _shells(spec_fn: Callable[[], mujoco.MjSpec]) -> mujoco.MjSpec:
    """The robot's spec with SHELLS' primitives as ``<body>_<k>_striker`` or ``_target`` collision geoms."""
    spec = spec_fn()
    for body, prims in json.loads(SHELLS.read_text()).items():  # missing: FileNotFoundError
        role = "striker" if STRIKERS.search(body) else "target"
        for k, p in enumerate(prims):
            name = f"{body}_{k}_{role}"
            if p["type"] == "capsule":
                kind = mujoco.mjtGeom.mjGEOM_CAPSULE
                spec.body(body).add_geom(name=name, type=kind, fromto=p["fromto"], size=[p["radius"], 0, 0])
            else:
                kind = mujoco.mjtGeom.mjGEOM_BOX
                spec.body(body).add_geom(name=name, type=kind, pos=p["pos"], quat=p["quat"], size=p["size"])
            spec.geom(name).group = SHELL_GROUP
    return spec


def shell_collisions(own: str, other: str) -> tuple[CollisionCfg, CollisionCfg]:
    """``own``'s strikers and targets touching ``other``'s (STRIKE, TARGET), with friction, as stiff as the rest."""
    common = {"condim": 3, "priority": 0, "friction": (SHELL_FRICTION,), "solref": STIFF_COLLISION.solref}
    return (
        CollisionCfg(
            geom_names_expr=(".*_striker",),
            contype=STRIKE[own],
            conaffinity=STRIKE[other] | TARGET[other],
            disable_other_geoms=False,
            **common,
        ),
        CollisionCfg(
            geom_names_expr=(".*_target",),
            contype=TARGET[own],
            conaffinity=STRIKE[other],
            disable_other_geoms=False,
            **common,
        ),
    )


def add_opponent(cfg: ManagerBasedRlEnvCfg, motion: str, policy: str) -> None:
    """A second ``robot`` on ``motion`` (as long as the ``motion`` command's), moved by ``policy``.

    Where the opponent's tracking would end the episode, it is cut short instead: its fall is no failure of the
    learner's.
    """
    robot = cfg.scene.entities["robot"]
    opponent = cfg.scene.entities[ENTITY] = copy.deepcopy(robot)
    opponent.spec_fn = partial(_shells, partial(_apart, robot.spec_fn))
    robot.spec_fn = partial(_shells, robot.spec_fn)
    robot.collisions = (*robot.collisions, *shell_collisions("robot", ENTITY))
    opponent.collisions = (*opponent.collisions, *shell_collisions(ENTITY, "robot"))
    x, y, z = opponent.init_state.pos
    opponent.init_state.pos = (x + APART, y, z)
    leader = cfg.commands["motion"]
    assert isinstance(leader, MotionCommandCfg)
    follower = FollowerMotionCommandCfg(**{f.name: getattr(leader, f.name) for f in fields(leader)}, leader="motion")
    follower.entity_name, follower.motion_file, follower.sampling_mode = ENTITY, motion, "uniform"
    follower.pose_range, follower.velocity_range, follower.joint_position_range = {}, {}, (0.0, 0.0)
    cfg.commands[COMMAND] = follower
    joints = cfg.actions["joint_pos"]
    assert isinstance(joints, JointPositionActionCfg)
    cfg.actions[ENTITY] = OpponentActionCfg(
        entity_name=ENTITY, joints=replace(joints, entity_name=ENTITY), policy=policy
    )
    cfg.observations[ENTITY] = opponent_observations(cfg.observations["actor"])
    for name, term in list(cfg.terminations.items()):
        if term.params.get("command_name") == "motion":
            cut = copy.deepcopy(term)
            cut.params["command_name"], cut.time_out = COMMAND, True
            cfg.terminations[f"{ENTITY}_{name}"] = cut
    cfg.sim.nconmax = cfg.sim.nconmax and 2 * cfg.sim.nconmax  # two robots' contacts
    cfg.sim.njmax = cfg.sim.njmax and 2 * cfg.sim.njmax
