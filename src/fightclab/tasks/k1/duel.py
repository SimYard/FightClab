"""A second T800 in a tracking task: it jabs at the learner at half of the learner's catches, like a makiwara.

rsl_rl trains one policy per env, so the opponent is part of the env: a zero-dim action term runs a TorchScript copy
of a trained tracker (``tools/export_policy.py``) on the actor's observations read off the opponent, and its motion
command plays a jab clip in front of the learner at half the catches of the learner's whole library motion. Training
on catches alone forgot the jab (docs/duel.md, "Growing the skill library"). Its observations carry no noise.
"""

import copy
import inspect
import json
import math
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
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, yaw_quat
from mjlab.utils.spec_config import CollisionCfg

from fightclab.robots.common import STIFF_COLLISION
from fightclab.tasks.k1 import makiwara
from fightclab.tasks.k1.footwork import OPPONENT_YAW

ENTITY = "opponent"  # also its action term's and observation group's name
COMMAND = "opponent_motion"
HEAD_HITS = "opponent_fist_on_head"
HEAD_CAP = 2.0  # makiwara hits a step at most, as the makiwara pays: a 535 N s spike blew the critic's loss up
# m pelvis to pelvis: the standing jab's full reach to a T800 head (0.965 m, the coach) less the makiwara's depth, as a
# jab punches through its target. At full reach a still head was touched by 31% of jabs, with 1.8 N s.
RANGE = 0.88
LEAD = 2 / 30  # s from the catch's start to the fist leaving the guard (the coach: the catch starts 0.05-0.08 s early)
# The share of catches the opponent jabs at: half, as many without a hit as with one. Hit at every catch, the learner,
# who knows the motion's phase, braced for it: from the fist's arrival its glove held 1-2 cm forward and 3 cm down of
# the catch's place, where the fist pushes it back from, and with no jab it was off by more than the coach's 2.5 cm in
# 96-99% of the hold (11% before). A catch with no hit makes that bracing cost.
SHARE = 0.5
ONSET = 0.02  # m of the fist toward the opponent: it has left the guard
# m from the env's origin where the opponent waits between jabs: out of the learner's way (its reference's pelvis
# spans x -3.8 to 5.4 m and y -2.6 to 5.9 m; at 4 m it walked through the opponent). Each env is its own world.
PARK = (0.0, 20.0)
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
# contype bit of the opponent's own (the vendor's) primitives. They meet it (self-collisions) and bit 1 (the floor,
# the learner's primitives); the makiwara, which only meets bit 1 (``makiwara.py``), doesn't meet them: the opponent
# appears before a catch, where the makiwara still stands when a strike comes right before it.
BODY = 64


class SpawnedMotionCommand(MotionCommand):
    """A clip on another entity, played in front of the leader's robot at SHARE of the leader's ``cue`` skills.

    The clip (guard, jab, guard) starts from its first frame so that its fist leaves the guard LEAD into the cue, RANGE
    in front of where the robot's pelvis will be at the cue (the reference's travel until then, turned by the robot's
    heading off the reference's), facing it. Started later, mid-guard, its policy jerked back and threw a malformed
    jab. It passes through the makiwara (BODY). The rest of the time, after a reset, and when the leader's motion
    jumps, it stands in guard PARK away. Like the makiwara, the learner doesn't see it, and the learner's whole library
    goes on around it.
    """

    cfg: "SpawnedMotionCommandCfg"
    _pending_forward: bool

    def __init__(self, cfg: "SpawnedMotionCommandCfg", env: ManagerBasedRlEnv) -> None:
        """Parked; the clip's onset (its fist ONSET toward the opponent) and first pelvis."""
        super().__init__(cfg, env)
        n = self.num_envs
        self.on = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.turn = torch.zeros(n, 4, device=self.device)  # the clip's placement: a yaw quaternion and an offset
        self.offset = torch.zeros(n, 3, device=self.device)
        self.origin = self.motion.body_pos_w[0, 0].clone()  # the clip's first pelvis, the placement's pivot
        fist = self.motion.body_pos_w[:, self.cfg.body_names.index(makiwara.FIST)]
        reach = (fist - fist[0]) @ self.toward(torch.zeros(1, device=self.device))[0]
        self.onset = int(torch.argmax((reach > ONSET).int()))
        self.cue: torch.Tensor | None = None  # per leader step: the step of the cue its clip starts for, or -1
        self.previous = torch.zeros(n, dtype=torch.long, device=self.device)  # the leader's step at the last update

    @property
    def leader(self) -> MotionCommand:
        """The learner's command."""
        term = self._env.command_manager.get_term(self.cfg.leader)
        assert isinstance(term, MotionCommand)
        return term

    def toward(self, turn: torch.Tensor) -> torch.Tensor:
        """The direction to the opponent, OPPONENT_YAW, turned by ``turn`` rad (envs,)."""
        yaw = OPPONENT_YAW + turn
        return torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1)

    def _cues(self) -> torch.Tensor:
        """Per leader step, the cue step whose clip starts there (or -1): from the leader's motion's labels."""
        leader = self.leader
        labels = Path(leader.cfg.motion_file).with_suffix(".labels.txt").read_text().split()
        total = leader.motion.time_step_total
        names = [labels[min(round(t * self._env.step_dt * 30), len(labels) - 1)] for t in range(total)]
        cue = torch.full((total,), -1, dtype=torch.long, device=self.device)
        for t in range(1, total):
            start = t + round(LEAD / self._env.step_dt) - self.onset
            if names[t] == self.cfg.cue and names[t - 1] != self.cfg.cue and start >= 0:
                cue[start] = t
        return cue

    def _place(self, env_ids: torch.Tensor, yaw: torch.Tensor, xy: torch.Tensor) -> None:
        """The clip's first frame at ``xy`` turned by ``yaw``, written into the sim."""
        self.time_steps[env_ids] = 0
        self.turn[env_ids] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        self.offset[env_ids] = torch.cat((xy, self.origin[2].expand(len(env_ids), 1)), dim=-1)
        pos, quat = self.body_pos_w[env_ids, 0], self.body_quat_w[env_ids, 0]
        lin, ang = self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0]
        self._write_reference_state_to_sim(
            env_ids, pos, quat, lin, ang, self.joint_pos[env_ids], self.joint_vel[env_ids]
        )
        self._pending_forward = True

    def _park(self, env_ids: torch.Tensor) -> None:
        self.on[env_ids] = False
        origins = self._env.scene.env_origins[env_ids, :2]
        park = origins + torch.tensor(PARK, device=self.device)
        self._place(env_ids, torch.zeros(len(env_ids), device=self.device), park)

    def _spawn(self, env_ids: torch.Tensor) -> None:
        leader, cue = self.leader, self.cue
        assert cue is not None
        now, then = leader.time_steps[env_ids], cue[leader.time_steps[env_ids]]
        heading = yaw_quat(quat_mul(leader.robot_anchor_quat_w[env_ids], quat_inv(leader.anchor_quat_w[env_ids])))
        turn = 2 * torch.atan2(heading[:, 3], heading[:, 0])
        travel = leader.motion.body_pos_w[then, 0] - leader.motion.body_pos_w[now, 0]
        target = leader.robot_body_pos_w[env_ids, 0] + quat_apply(heading, travel) + RANGE * self.toward(turn)
        self.on[env_ids] = True
        self._place(env_ids, math.pi + turn, target[:, :2])

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """A reset: parked until a cue it jabs at."""
        self._park(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """A frame on while playing; parked at the clip's end or where the leader jumped; spawned at a cue."""
        if self.cue is None:
            self.cue = self._cues()
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        now = self.leader.time_steps[ids]
        self.time_steps[ids] += self.on[ids].long()
        jumped = now != self.previous[ids] + 1 if env_ids is None else torch.zeros_like(now, dtype=torch.bool)
        done = ids[self.on[ids] & ((self.time_steps[ids] >= self.motion.time_step_total - 1) | jumped)]
        if len(done) > 0:
            self._park(done)
        cued = ids[~self.on[ids] & (self.cue[now] >= 0)]
        cued = cued[torch.rand(len(cued), device=self.device) < SHARE]
        if len(cued) > 0:
            self._spawn(cued)
        self.previous[ids] = now
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()

    def _placed(self, pos: torch.Tensor) -> torch.Tensor:
        """Clip positions (envs, ..., 3) where the clip is placed."""
        shape = (-1,) + (1,) * (pos.dim() - 2) + (4,)
        return quat_apply(self.turn.view(shape).expand(*pos.shape[:-1], 4), pos - self.origin) + self.offset.view(
            (*shape[:-1], 3)
        )

    def _turned(self, v: torch.Tensor) -> torch.Tensor:
        shape = (-1,) + (1,) * (v.dim() - 2) + (4,)
        return quat_apply(self.turn.view(shape).expand(*v.shape[:-1], 4), v)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """The clip's bodies where it is placed."""
        return self._placed(self.motion.body_pos_w[self.time_steps])

    @property
    def body_quat_w(self) -> torch.Tensor:
        """The clip's bodies' orientations, turned."""
        q = self.motion.body_quat_w[self.time_steps]
        return quat_mul(self.turn[:, None].expand_as(q), q)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """The clip's bodies' velocities, turned."""
        return self._turned(self.motion.body_lin_vel_w[self.time_steps])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """The clip's bodies' angular velocities, turned."""
        return self._turned(self.motion.body_ang_vel_w[self.time_steps])

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """The clip's anchor where it is placed."""
        return self._placed(self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """The clip's anchor's orientation, turned."""
        return quat_mul(self.turn, self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """The clip's anchor's velocity, turned."""
        return self._turned(self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """The clip's anchor's angular velocity, turned."""
        return self._turned(self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index])


@dataclass(kw_only=True)
class SpawnedMotionCommandCfg(MotionCommandCfg):
    """``MotionCommandCfg`` (its motion: the clip), the ``leader`` command's name and the ``cue`` skill's label."""

    leader: str
    cue: str

    def build(self, env: ManagerBasedRlEnv) -> SpawnedMotionCommand:
        """The command."""
        return SpawnedMotionCommand(self, env)


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
    impulse = torch.linalg.norm(force, dim=-1).sum(dim=(1, 2)) * makiwara.PHYSICS_DT
    return (impulse / makiwara.IMPULSE_SCALE).clamp(max=HEAD_CAP)


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


def add_opponent(cfg: ManagerBasedRlEnvCfg, clip: str, policy: str, cue: str) -> None:
    """A second ``robot``, moved by ``policy``, playing ``clip`` at SHARE of the ``cue``s of the ``motion`` command.

    See ``SpawnedMotionCommand``. Where the opponent's tracking would end the episode, it is cut short instead: its
    fall is no failure of the learner's.
    """
    robot = cfg.scene.entities["robot"]
    opponent = cfg.scene.entities[ENTITY] = copy.deepcopy(robot)
    opponent.spec_fn = partial(_shells, partial(_apart, robot.spec_fn))
    robot.spec_fn = partial(_shells, robot.spec_fn)
    robot.collisions = (*robot.collisions, *shell_collisions("robot", ENTITY))
    assert opponent.collisions == (STIFF_COLLISION,), "the opponent's own primitives are put on BODY"
    own = replace(STIFF_COLLISION, contype=BODY, conaffinity=BODY | 1)
    opponent.collisions = (own, *shell_collisions(ENTITY, "robot"))
    x, y, z = opponent.init_state.pos
    opponent.init_state.pos = (x + APART, y, z)
    leader = cfg.commands["motion"]
    assert isinstance(leader, MotionCommandCfg)
    spawned = SpawnedMotionCommandCfg(
        **{f.name: getattr(leader, f.name) for f in fields(leader)}, leader="motion", cue=cue
    )
    spawned.entity_name, spawned.motion_file = ENTITY, clip
    spawned.pose_range, spawned.velocity_range, spawned.joint_position_range = {}, {}, (0.0, 0.0)
    cfg.commands[COMMAND] = spawned
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
