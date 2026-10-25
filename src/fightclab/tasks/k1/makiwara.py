"""A makiwara for the K-1 jabs: a padded post on a spring that stands where each jab of the reference lands.

The tracked jab is shadow boxing: the arm reaches full extension in the air. In front of a makiwara 20 cm inside that
reach the trained T800 kept driving its fist toward the point behind the pad, pressed into it for 0.12 s and shoved
itself 25 cm back; nothing it had learned said what to do on contact. Here the makiwara stands in front of every
strike of the motion, DEPTH inside the reference fist's farthest reach, so the fist meets it while still fast, and
is parked out of the way in footwork. The lead fist isn't tracked while it touches the pad (pushing on gets it
nothing), the impulse of the first IMPACT_STEPS of a touch is paid for, less what comes back into the trunk, a
touch longer than PUSH_STEPS costs, and so does the pelvis falling behind the reference's travel in the strike.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
from mjlab.utils.spec_config import CollisionCfg

from fightclab.tasks.k1.footwork_motion import FPS
from fightclab.tasks.k1.strikes import STRIKES, H

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg

ENTITY, SENSOR = "makiwara", "makiwara_hits"
FIST = "LINK_WRIST_END_L"  # the lead fist
ARM = re.compile("SHOULDER|ELBOW|WRIST")  # the lead arm's links, with an _L
# A 1.75 m post on a spring hinge at the floor, padded from 1.05 to 1.72 m (body jabs aimed from 1.15 m to head jabs
# at 1.66 m): about 5 kg at the pad, like a head, giving 4 cm under 300 N.
POST_MASS, POST_HEIGHT, PAD = 10.0, 1.75, (1.05, 1.72)
STIFFNESS, DAMPING = 13700.0, 230.0  # N m/rad
PAD_DEPTH, POST_DEPTH = 0.1, 0.08  # m along the punch
# The K-1 coach: aim a jab 5-10 cm behind the surface (the reference fist peaks at 3.3 m/s 22-30 cm before full
# reach and is down to 1.2 m/s 2.5 cm before it), and let it find its range: the face goes 0.02-0.12 H inside full
# reach, up to 0.02 H to either side. At 2.5 cm a stance drift of 2 cm made a jab miss.
DEPTH, SIDE = (0.02 * H, 0.12 * H), 0.02 * H
# The step jab is for a target beyond standing reach (the coach): its pad 0.02-0.06 H inside, so the fist lands
# 0.03-0.12 s before the lead foot. Deeper, it landed 0.14 s before, on one foot and with half the reference's
# momentum; on its shallowest third of pads it already stepped the full 0.1 H with 30 kg m/s.
STRIKE_DEPTH = {"step_jab": (0.02 * H, 0.06 * H)}
PARK = (0.0, 0.0, 10.0)  # m above the env's origin between strikes
# The coach: score the impulse of the first 40 ms, never the total (a push piles it up on a spring-backed target),
# and nothing for a touch longer than 1.5 times a free 3 kg fist bouncing off this pad (about 50 ms).
IMPACT_STEPS = 2  # policy steps of a touch whose impulse is the hit (40 ms)
PUSH_STEPS = 4  # a touch over 80 ms is a push: no score, and each further step costs
IMPULSE_SCALE = 15.0  # N s of hit that earn 1 (the coach: a good jab's first 40 ms is 10-18 N s)
PUSH_COST = 0.2  # per policy step of pushing, against 1 for IMPULSE_SCALE
HIT_CAP = 2.0  # IMPULSE_SCALEs a hit is paid at most: an unbounded 535 N s spike blew the critic's loss up
LAG_FREE, LAG_COST = 0.005, 0.005  # m of lag free in a strike; cost per cm beyond it per policy step
LAG_BODIES = ("LINK_BASE", "LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")  # the worst of these is charged
PHYSICS_DT = 0.005  # s per substep of the contact sensor's history


def step_labels(motion: MotionCommand, step_dt: float) -> list[str]:
    """The skill at each of ``motion``'s env steps, from the labels next to its file (``steps.py``, at FPS)."""
    labels = Path(motion.cfg.motion_file).with_suffix(".labels.txt").read_text().split()
    return [labels[min(round(t * step_dt * FPS), len(labels) - 1)] for t in range(motion.motion.time_step_total)]


def makiwara_spec() -> mujoco.MjSpec:
    """The post, its hinge at the floor bending toward local +x (the punch), the pad on its -x face."""
    spec = mujoco.MjSpec()
    post = spec.worldbody.add_body(name="post")
    post.add_joint(name="bend", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 1, 0), stiffness=STIFFNESS, damping=DAMPING)
    post.add_geom(
        name="post",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(POST_DEPTH / 2, 0.06, POST_HEIGHT / 2 - 0.01),
        pos=(0, 0, POST_HEIGHT / 2 + 0.01),  # clear of the floor
        mass=POST_MASS,
        rgba=(0.55, 0.38, 0.2, 1),
    )
    post.add_geom(
        name="pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(PAD_DEPTH / 2, 0.15, (PAD[1] - PAD[0]) / 2),
        pos=(-(POST_DEPTH + PAD_DEPTH) / 2, 0, sum(PAD) / 2),
        mass=1.0,
        rgba=(0.8, 0.75, 0.55, 1),
    )
    return spec


def makiwara_entity() -> EntityCfg:
    """The makiwara, a mocap prop placed by ``Makiwara`` every step.

    It meets what has bit 1 in its contype (the floor, the learner's primitives), not what only meets bit 1: the
    duel's opponent passes through it (``duel.BODY``).
    """
    collision = CollisionCfg(geom_names_expr=(".*",), contype=0, conaffinity=1, condim=3, priority=0)
    return EntityCfg(spec_fn=makiwara_spec, init_state=EntityCfg.InitialStateCfg(pos=PARK), collisions=(collision,))


def makiwara_hits() -> ContactSensorCfg:
    """The lead fist's net force on the pad, per physics substep of a policy step."""
    return ContactSensorCfg(
        name=SENSOR,
        primary=ContactMatch(mode="geom", pattern="pad", entity=ENTITY),
        secondary=ContactMatch(mode="body", pattern=FIST, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=4,
    )


def fists_on_makiwara(
    env: ManagerBasedRlEnv, command_name: str, body_names: tuple[str, ...], std: float
) -> torch.Tensor:
    """``tracking.bodies_pos`` with only the height of the lead fist (``body_names[0]``) while it touches the pad.

    Not its reach: pushing on toward the reference's full reach behind the pad gets nothing. Its height still: with
    it untracked too, the fist slid 4-5 cm up the pad in every touch and left it at 2 m/s upward (the K-1 coach: on a
    head it paws up to the forehead, on a catching glove it climbs over it).
    """
    motion = env.command_manager.get_term(command_name)
    assert isinstance(motion, MotionCommand)
    bodies = [motion.cfg.body_names.index(name) for name in body_names]
    offset = (motion.body_pos_relative_w[:, bodies] - motion.robot_body_pos_w[:, bodies]).square()
    error = offset.sum(dim=-1)
    touching = env.scene[SENSOR].data.found[:, 0] > 0
    error[:, 0] = torch.where(touching, offset[:, 0, 2], error[:, 0])
    return torch.exp(-error.mean(dim=-1) / std**2)


class Makiwara:
    """Places the makiwara for each strike of the motion and pays for hits, charges for pushes.

    A strike is a run of the motion's frames labelled with a ``strikes.STRIKES`` command (``steps.py`` writes the
    labels next to the motion, at 30 fps). The pad's face goes DEPTH before the reference fist's farthest point of
    the strike, on its line from the guard; that point is taken relative to the reference's anchor when the strike
    starts and put on the robot's (the robot drifts several cm from the reference in an episode). One hit per strike.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        """Each frame's strike, and each strike's farthest frame and direction."""
        motion = env.command_manager.get_term(cfg.params["command_name"])
        assert isinstance(motion, MotionCommand)
        frames = motion.motion.time_step_total
        names = step_labels(motion, env.step_dt)
        fist = motion.motion.body_pos_w[:, motion.cfg.body_names.index(FIST)]
        self.strike = torch.full((frames,), -1, dtype=torch.long, device=env.device)
        far, direction, depths = [], [], []
        t = 0
        while t < frames:
            if names[t] not in STRIKES:
                t += 1
                continue
            end = t
            while end < frames and names[end] == names[t]:
                end += 1
            reach = (fist[t:end, :2] - fist[t, :2]).norm(dim=-1)
            far.append(t + int(reach.argmax()))
            way = fist[far[-1], :2] - fist[t, :2]
            direction.append(torch.cat((way / way.norm().clamp(min=1e-6), way.new_zeros(1))))
            depths.append(STRIKE_DEPTH.get(names[t], DEPTH))
            self.strike[t:end] = len(far) - 1
            t = end
        self.far = torch.tensor(far, device=env.device)
        self.direction = torch.stack(direction)
        self.depths = torch.tensor(depths, device=env.device)  # (strikes, 2): each strike's range of pad depth
        n = env.num_envs
        self.current = torch.full((n,), -1, dtype=torch.long, device=env.device)
        self.pose = torch.zeros(n, 7, device=env.device)
        self.toward = torch.zeros(n, 3, device=env.device)
        self.depth = torch.zeros(n, device=env.device)  # m inside full reach of the current strike's pad (for evals)
        self.touch = torch.zeros(n, dtype=torch.long, device=env.device)
        self.impulse = torch.zeros(n, device=env.device)
        self.paid = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.park = torch.tensor((*PARK, 1.0, 0.0, 0.0, 0.0), device=env.device)
        self.envs = torch.arange(n, device=env.device)
        robot = env.scene["robot"]
        mass = env.sim.mj_model.body_mass[robot.indexing.body_ids.cpu().numpy()]
        arm = [i for i, name in enumerate(robot.body_names) if name.endswith("_L") and ARM.search(name)]
        mass[arm] = 0.0  # the trunk and legs: the fist's arm flies back after any hit
        self.trunk_mass = torch.tensor(mass, dtype=torch.float32, device=env.device)
        self.lag = [motion.cfg.body_names.index(name) for name in LAG_BODIES]
        self.start = torch.zeros(n, 2, len(LAG_BODIES), 2, device=env.device)  # the robot's and the reference's

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        """A fresh strike after a reset: the pose, the start and the hit's bookkeeping are taken anew."""
        ids = slice(None) if env_ids is None else env_ids
        self.current[ids], self.touch[ids], self.impulse[ids], self.paid[ids] = -1, 0, 0.0, False

    def __call__(self, env: ManagerBasedRlEnv, *, command_name: str) -> torch.Tensor:
        """The hit bonus minus the push and lag costs, this step; then places the makiwara for the next one."""
        motion = env.command_manager.get_term(command_name)
        assert isinstance(motion, MotionCommand)
        force = env.scene[SENSOR].data.force_history[:, 0]  # (envs, substeps, 3)
        push = (force * self.toward[:, None]).sum(dim=-1).abs().sum(dim=-1) * PHYSICS_DT
        touching = push > 0
        ended = ~touching & (self.touch > 0)
        pay = ended & ~self.paid & (self.touch <= PUSH_STEPS)
        self.touch = torch.where(touching, self.touch + 1, 0)
        self.impulse += torch.where(touching & (self.touch <= IMPACT_STEPS) & ~self.paid, push, 0.0)
        # A hit is what the pad got less what came back into the trunk: its momentum away from the pad as the touch
        # ends. The body carried in and the legs braced against the floor both pay; the arm and chest alone hitting
        # harder than that only shove the robot back (mk2: 23 N s hits on 8 kg m/s, the pelvis 4-8 cm behind after).
        velocity = env.scene["robot"].data.body_com_lin_vel_w
        trunk = ((velocity * self.trunk_mass[:, None]).sum(dim=1) * self.toward).sum(dim=-1)
        hit = (self.impulse - (-trunk).clamp(min=0.0)).clamp(min=0.0)
        bonus = torch.where(pay, (hit / IMPULSE_SCALE).clamp(max=HIT_CAP), 0.0)
        self.paid |= ended  # one hit per strike; a push forfeits it
        self.impulse[ended] = 0.0
        reward = bonus - PUSH_COST * (self.touch > PUSH_STEPS).float()

        strike = self.strike[motion.time_steps]
        new = (strike >= 0) & (strike != self.current)
        if new.any():
            ids = new.nonzero().squeeze(-1)
            origin = env.scene.env_origins[ids]
            far = motion.motion.body_pos_w[self.far[strike[ids]], motion.cfg.body_names.index(FIST)] + origin
            turn = yaw_quat(quat_mul(motion.robot_anchor_quat_w[ids], quat_inv(motion.anchor_quat_w[ids])))
            far = motion.robot_anchor_pos_w[ids] + quat_apply(turn, far - motion.anchor_pos_w[ids])
            toward = quat_apply(turn, self.direction[strike[ids]])
            low, high = self.depths[strike[ids]].unbind(-1)
            depth = (low + (high - low) * torch.rand(len(ids), device=env.device))[:, None]
            self.depth[ids] = depth[:, 0]
            side = torch.empty(len(ids), 1, device=env.device).uniform_(-SIDE, SIDE)
            left = torch.stack((-toward[:, 1], toward[:, 0], toward[:, 2]), 1)
            base = far + (PAD_DEPTH + POST_DEPTH / 2 - depth) * toward + side * left
            base[:, 2] = origin[:, 2]
            half = torch.atan2(toward[:, 1], toward[:, 0]) / 2
            self.pose[ids] = torch.cat(
                (base, half.cos()[:, None], torch.zeros_like(base[:, :2]), half.sin()[:, None]), 1
            )
            self.toward[ids] = toward
            self.paid[ids], self.touch[ids], self.impulse[ids] = False, 0, 0.0
            self.start[ids, 0] = motion.robot_body_pos_w[ids][:, self.lag, :2]
            self.start[ids, 1] = motion.body_pos_w[ids][:, self.lag, :2]
        self.current = strike
        # The worst of the pelvis and the feet falling behind the reference's own travel since the strike started,
        # beyond LAG_FREE. The pelvis alone: nothing else charged the shove or made the robot walk back (mk2 stayed
        # 4 cm behind). The feet: mk3 got the reference's body over its lead foot by moving the foot 4 cm back, not
        # the body forward, and slid its rear foot 4-6 cm back after the step jab; the torso-anchored terms can't tell.
        direction = self.direction[strike.clamp(min=0), None, :2]
        ref_way = ((motion.body_pos_w[:, self.lag, :2] - self.start[:, 1]) * direction).sum(-1)
        way = ((motion.robot_body_pos_w[:, self.lag, :2] - self.start[:, 0]) * self.toward[:, None, :2]).sum(-1)
        lag = (ref_way - way - LAG_FREE).clamp(min=0.0).amax(dim=-1)
        reward = reward - LAG_COST * (strike >= 0) * lag / 0.01
        park = self.park + torch.cat((env.scene.env_origins, torch.zeros_like(self.pose[:, :4])), 1)
        pose = torch.where((strike >= 0)[:, None], self.pose, park)
        # Every env by index: with env_ids=None mjlab's mocap write mismatches its own shapes.
        env.scene[ENTITY].write_mocap_pose_to_sim(pose, env_ids=self.envs)
        return reward
