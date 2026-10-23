"""Adversarial motion prior (AMP, Peng et al. 2021) as a reward term: style from video clips, no hand-made shaping.

A discriminator learns to tell the robot's state transitions from the clips' (least squares GAN with a gradient
penalty on the clips) and pays ``max(0, 1 - (d - 1)^2 / 4)``: 1 where a transition looks like the clips. rsl_rl has
no AMP, so the discriminator lives in the reward term and takes one Adam step per env step on that step's batch.
A state is the robot's pose seen from its own heading: joint angles and speeds, pelvis height, tilt and velocities,
and where the feet, the wrists and the head are relative to the pelvis.

Clips come from tutorials, where coaches also talk and point. Keep the guarded, orthodox runs of motion npz files:

    uv run python -m fightclab.tasks.k1.amp OUT_DIR IN.npz...
"""

import sys
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul
from torch import nn

from fightclab.motion import runs

KEYS = ("LINK_FOOT_L", "LINK_FOOT_R", "LINK_WRIST_END_L", "LINK_WRIST_END_R", "LINK_HEAD_YAW")
GUARD = 0.43  # m, farthest wrist from the head's base: the coaches' guards are 0.31-0.41, talking 0.46 and up
# m a wrist may rise above the head's base; the stance's are within 0.02 of it. In 2 of 29 chunks GMR's IK, warm
# started frame to frame, got stuck with an arm straight up while the coach's glove was at his face.
RAISED = 0.10
FOOT_UP = 0.3  # m: the highest a foot body gets in footwork (steps lift it to ~0.2); kicks go higher
MIN_RUN = 50  # frames (1 s) a kept run lasts at least


def features(  # noqa: PLR0917
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """One state per row; world-frame inputs, (w, x, y, z) quaternions, ``keys`` (N, K, 3)."""
    w, x, y, z = root_quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]

    def heading_frame(v: torch.Tensor) -> torch.Tensor:  # (N, K, 3)
        return torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), -1)

    down = torch.tensor((0.0, 0.0, -1.0), device=root_quat.device).expand(len(root_quat), 3)
    rel = heading_frame(keys - root_pos[:, None])
    return torch.cat(
        (
            joint_pos,
            joint_vel,
            root_pos[:, 2:],
            quat_apply_inverse(root_quat, down),
            heading_frame(root_lin_vel[:, None])[:, 0],
            quat_apply_inverse(root_quat, root_ang_vel),
            rel.flatten(1),
        ),
        -1,
    )


def clip_features(path: Path, key_ids: list[int], device: str) -> torch.Tensor:
    """States of a tracking-task motion npz (50 fps, body 0 is the pelvis)."""
    d = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in np.load(path).items()}
    return features(
        d["joint_pos"], d["joint_vel"], d["body_pos_w"][:, 0], d["body_quat_w"][:, 0],
        d["body_lin_vel_w"][:, 0], d["body_ang_vel_w"][:, 0], d["body_pos_w"][:, key_ids],
    )  # fmt: skip


class style:
    """AMP reward.

    Params: ``sources`` ({npz glob: share of the clip samples the discriminator sees}), ``train`` (False in play),
    ``lr``, ``gp`` (gradient penalty weight), ``logit_reg`` (weight on the output layer's squared weights).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        """Load the clips' transitions and set up the discriminator."""
        self.robot = env.scene["robot"]
        self.key_ids = [self.robot.body_names.index(n) for n in KEYS]
        refs, weights = [], []
        for pattern, share in cfg.params["sources"].items():
            paths = sorted(Path(pattern).parent.glob(Path(pattern).name))
            assert paths, f"no clips match {pattern}"
            clips = [clip_features(p, self.key_ids, env.device) for p in paths]
            ref = torch.cat([torch.cat((f[:-1], f[1:]), -1) for f in clips])
            refs.append(ref)
            weights.append(torch.full((len(ref),), share / len(ref), device=env.device))
        self.ref, self.weights = torch.cat(refs), torch.cat(weights)
        # A floor on the scale: the clips' head joints never move (std 0), and scaled by 1e-3 the robot's head alone
        # told the discriminator who was who, whatever the feet did.
        self.mean, self.std = self.ref.mean(0), self.ref.std(0).clamp(min=0.05)
        dim = self.ref.shape[1]
        self.disc = nn.Sequential(nn.Linear(dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
        self.disc.to(env.device)
        # isaacgymenvs' AMP settings. With lr 1e-4 and none of the regularization, the discriminator scored nearly every
        # robot transition at or below -1, where the reward is flat at 0, and the style stopped teaching anything.
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=cfg.params.get("lr", 5e-5), weight_decay=1e-4)
        self.logit_reg = cfg.params.get("logit_reg", 0.05)
        self.train, self.gp = cfg.params.get("train", True), cfg.params.get("gp", 5.0)
        self.prev: torch.Tensor | None = None
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def state(self) -> torch.Tensor:
        """The robots' current states."""
        d = self.robot.data
        return features(
            d.joint_pos, d.joint_vel, d.root_link_pos_w, d.root_link_quat_w, d.root_link_lin_vel_w,
            d.root_link_ang_vel_w, d.body_link_pos_w[:, self.key_ids],
        )  # fmt: skip

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Drop the transition that spans the reset."""
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv, **kwargs: object) -> torch.Tensor:
        """Pay each robot for its last transition, then train the discriminator on them."""
        cur = self.state()
        x = torch.cat((cur if self.prev is None else self.prev, cur), -1)
        with torch.no_grad():
            d = self.disc((x - self.mean) / self.std).squeeze(-1)
        reward = torch.clamp(1 - 0.25 * (d - 1) ** 2, min=0) * self.valid
        if self.train and self.valid.any():
            self.update(x[self.valid])
        self.prev, self.valid[:] = cur, True
        return reward

    def update(self, agent: torch.Tensor) -> None:
        """One least-squares GAN step; rsl_rl steps the env in inference mode, so grads are switched back on."""
        with torch.inference_mode(False), torch.enable_grad():
            agent = ((agent - self.mean) / self.std).clone()
            ref = self.ref[torch.multinomial(self.weights, len(agent), replacement=True)]
            ref = ((ref - self.mean) / self.std).clone().requires_grad_(True)
            d_ref, d_agent = self.disc(ref), self.disc(agent)
            (grad,) = torch.autograd.grad(d_ref.sum(), ref, create_graph=True)
            loss = ((d_ref - 1) ** 2).mean() + ((d_agent + 1) ** 2).mean() + self.gp * grad.pow(2).sum(-1).mean()
            loss = loss + self.logit_reg * self.disc[-1].weight.square().sum()  # keeps the logits from saturating
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


def in_guard(clip: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Per frame: both wrists near the head (no punch, no talking) and not above it (no IK artifact)."""
    pos, i = clip["body_pos_w"], names.index
    wrists = pos[:, [i("LINK_WRIST_END_L"), i("LINK_WRIST_END_R")]]
    reach = np.linalg.norm(wrists - pos[:, i("LINK_HEAD_YAW"), None], axis=-1).max(1)
    raised = wrists[..., 2].max(1) - pos[:, i("LINK_HEAD_YAW"), 2]
    return (reach < GUARD) & (raised < RAISED)


def guarded_runs(clip: dict[str, np.ndarray], names: list[str]) -> list[slice]:
    """Runs of at least MIN_RUN frames in guard, orthodox, facing the camera, feet low.

    In guard: both wrists near the head and not above it. Orthodox: the left foot ahead along the pelvis' heading.
    Facing: the pelvis within 90 degrees of the clip's mean heading; coaches walking back to their mark turn their
    back on the camera, which stands in for the opponent.

    ponytail: the mean heading stands in for the camera; GVHMR's in-camera pose would give it exactly.
    """
    pos, i = clip["body_pos_w"], names.index
    w, x, y, z = clip["body_quat_w"][:, 0].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    feet = pos[:, i("LINK_FOOT_L"), :2] - pos[:, i("LINK_FOOT_R"), :2]
    ok = in_guard(clip, names) & (np.cos(yaw) * feet[:, 0] + np.sin(yaw) * feet[:, 1] > 0)
    ok &= pos[:, [i("LINK_FOOT_L"), i("LINK_FOOT_R")], 2].max(1) < FOOT_UP  # no kicks
    mean = np.angle(np.exp(1j * yaw[ok]).sum())
    ok &= np.cos(yaw - mean) > 0
    return [slice(a, b) for a, b in runs(ok) if b - a >= MIN_RUN]


class reset_to_clips:
    """Reset event: a share of the robots start from random clip frames (reference state initialization).

    Without it the policies settled into a low, wide, upright stance the clips never show, where the style reward was
    flat at 0 and could not pull them back. Each frame keeps its pelvis height, pose and velocities, turned to a random
    heading at the env's origin. Params: ``sources`` (npz globs, as the style term's), ``share`` of resets.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        """Load every clip frame's root and joint state."""
        paths = [p for pattern in cfg.params["sources"] for p in sorted(Path(pattern).parent.glob(Path(pattern).name))]
        clips = [np.load(p) for p in paths]

        def stack(key: str, body: int | None = None) -> torch.Tensor:
            parts = [c[key] if body is None else c[key][:, body] for c in clips]
            return torch.tensor(np.concatenate(parts), dtype=torch.float32, device=env.device)

        self.root = torch.cat(
            (stack("body_pos_w", 0), stack("body_quat_w", 0), stack("body_lin_vel_w", 0), stack("body_ang_vel_w", 0)),
            -1,
        )
        self.joint_pos, self.joint_vel = stack("joint_pos"), stack("joint_vel")
        self.robot = env.scene["robot"]

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, sources: object, share: float) -> None:
        """Move about ``share`` of the resetting robots to clip frames."""
        del sources  # Loaded at init.
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        ids = env_ids[torch.rand(len(env_ids), device=env.device) < share]
        if len(ids) == 0:
            return
        k = torch.randint(len(self.root), (len(ids),), device=env.device)
        root = self.root[k].clone()
        zero = torch.zeros(len(ids), device=env.device)
        turn = quat_from_euler_xyz(zero, zero, (torch.rand(len(ids), device=env.device) * 2 - 1) * torch.pi)
        root[:, 3:7] = quat_mul(turn, root[:, 3:7])
        root[:, 7:10], root[:, 10:13] = quat_apply(turn, root[:, 7:10]), quat_apply(turn, root[:, 10:13])
        root[:, :3] += env.scene.env_origins[ids] - torch.nn.functional.pad(root[:, :2], (0, 1))
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        self.robot.write_joint_state_to_sim(self.joint_pos[k], self.joint_vel[k], env_ids=ids)


if __name__ == "__main__":
    from fightclab.robots import t800

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    m = t800.get_spec().compile()
    body_names = [m.body(b).name for b in range(1, m.nbody)]
    for path in map(Path, sys.argv[2:]):
        clip = dict(np.load(path))
        guarded = guarded_runs(clip, body_names)
        for k, run in enumerate(guarded):
            np.savez(out / f"{path.stem}_{k:02d}.npz", **{n: v if n == "fps" else v[run] for n, v in clip.items()})
        kept = sum(r.stop - r.start for r in guarded)
        print(f"{path.stem}: kept {kept / 50:.0f} of {len(clip['joint_pos']) / 50:.0f} s in {len(guarded)} runs")
