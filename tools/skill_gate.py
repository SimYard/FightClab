"""Per-skill checks of a T800 K-1 tracker: a fine-tune is accepted only if no skill falls below the accepted one's.

    uv run python tools/skill_gate.py CHECKPOINT OUT.json [BASELINE.json]

Rolls the policy out in Fightclab-K1-Makiwara-T800's training env (pushes off, deterministic actions; ENVS envs for
SECONDS) and measures each skill of the motion by its labels:
- strikes (jab, jab_body, step_jab): the share of strikes touching the pad, the first 40 ms of impulse (N s, median),
  the share pushing longer than 80 ms, the recoil tail (pelvis more than 1.75 cm behind its pre-strike spot within
  0.5 s of the touch), the fist's height error at the touch (cm, p90 of its size);
- catch: the K-1 coach's checks (``docs/duel.md``), each as the share of catches passing; positions relative to the
  torso (the tracked bodies don't include the head) and the elbow-pitch link (the tracked elbow);
- footwork moves: the pelvis travel's error against the reference's over a move (cm, median), and the feet-slip cost;
- every skill: the mean body position error (cm); overall falls per minute.
With BASELINE.json, prints each metric against it with TOLERANCE and exits 1 if any fails. A fine-tune that loses a
skill without falling (duel2: the jab's 40 ms impulse 20 -> 10 N s) shows in no training curve, only here.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommand
from scipy.spatial.transform import Rotation

from fightclab.tasks.k1 import catch, makiwara
from fightclab.tasks.k1.footwork import OPPONENT_YAW
from fightclab.tasks.k1.steps import CATCH, COMMANDS
from fightclab.tasks.k1.strikes import STRIKES

TASK = "Fightclab-K1-Makiwara-T800"
ENVS, SECONDS = 512, 15.0  # about 800 strikes and 500 catches
TOUCH = 1e-4  # N s of push in a step that counts as touching the pad
PUSH, RECOIL, RECOIL_WINDOW = 0.08, 0.0175, 0.5  # s, m, s
COVER, DEPTH, ON_TIME, ELBOW = 0.025, (-0.03, 0.04), 0.15, 0.03  # the coach's catch checks (m, m, s, m)
LEAD, PITCH, YAW, HOME, HOME_AFTER = 0.025, 5.0, 4.0, 0.04, 0.15  # (m, deg, deg, m, s)
# Per metric: which way is worse, how much worse than the baseline passes, and whether that is a fraction of it.
TOLERANCE: dict[str, tuple[str, float, bool]] = {
    "touch": ("lower", 0.05, False),
    "j40": ("lower", 0.15, True),
    "push": ("higher", 0.03, False),
    "recoil": ("higher", 0.05, False),
    "height_p90_cm": ("higher", 1.0, False),
    "body_cm": ("higher", 0.2, True),
    "cover": ("lower", 0.05, False),
    "on_time": ("lower", 0.05, False),
    "elbow": ("lower", 0.05, False),
    "still": ("lower", 0.05, False),
    "travel_cm": ("higher", 1.0, False),
    "slip": ("higher", 0.2, True),
    "falls_per_min": ("higher", 0.5, False),
}
MOVES = tuple(k for k, v in COMMANDS.items() if v is not None)


def rollout(checkpoint: str) -> tuple[dict[str, np.ndarray], list[str], float]:
    """Per step and env: what the checks need; the motion's labels; the policy step."""
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = ENVS
    cfg.events.pop("push_robot")
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env)
    runner = load_runner_cls(TASK)(venv, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    motion = env.command_manager.get_term("motion")
    assert isinstance(motion, MotionCommand)
    pad = env.reward_manager.get_term_cfg("makiwara").func
    names = list(motion.cfg.body_names)
    body = {
        k: names.index(n)
        for k, n in (
            ("pelvis", "LINK_BASE"),
            ("torso", "LINK_WAIST_YAW"),
            ("fist", makiwara.FIST),
            ("glove", "LINK_WRIST_END_R"),
            ("elbow", "LINK_ELBOW_PITCH_R"),
        )
    }
    slip = env.reward_manager.active_terms.index("feet_slip")
    keys = (
        "t",
        "done",
        "fell",
        "strike",
        "push",
        "toward",
        "body_err",
        "slip",
        "quat",
        "ref_quat",
        *body,
        *(f"ref_{k}" for k in body),
    )
    rec: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    obs = venv.get_observations()
    for _ in range(round(SECONDS / env.step_dt)):
        with torch.no_grad():
            obs, _, dones, _ = venv.step(policy(obs))
        force = env.scene[makiwara.SENSOR].data.force_history[:, 0]
        rec["t"].append(motion.time_steps.clone())
        rec["done"].append(dones.bool())
        rec["fell"].append(env.termination_manager.terminated.clone())
        rec["strike"].append(pad.strike[motion.time_steps].clone())
        rec["push"].append((force * pad.toward[:, None]).sum(-1).abs().sum(-1) * makiwara.PHYSICS_DT)
        rec["toward"].append(pad.toward.clone())
        rec["body_err"].append((motion.body_pos_relative_w - motion.robot_body_pos_w).norm(dim=-1).mean(-1))
        rec["slip"].append(-env.reward_manager._step_reward[:, slip])
        rec["quat"].append(motion.robot_body_quat_w[:, body["torso"]].clone())
        rec["ref_quat"].append(motion.body_quat_w[:, body["torso"]].clone())
        for k, i in body.items():
            rec[k].append(motion.robot_body_pos_w[:, i].clone())
            rec[f"ref_{k}"].append(motion.body_pos_w[:, i].clone())
    labels = Path(motion.cfg.motion_file).with_suffix(".labels.txt").read_text().split()
    return {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}, labels, env.step_dt


def runs(mask: np.ndarray, done: np.ndarray) -> list[tuple[int, int, int]]:
    """(env, first step, step after) of each run of ``mask`` wholly inside the rollout and one episode."""
    out = []
    steps, envs = mask.shape
    for e in range(envs):
        t = 1
        while t < steps:
            if mask[t, e] and not mask[t - 1, e] and not done[t - 1, e]:
                end = t
                while end < steps and mask[end, e] and not done[end - 1, e]:
                    end += 1
                if end < steps and not mask[end, e]:
                    out.append((e, t, end))
                t = end
            t += 1
    return out


def measure(rec: dict[str, np.ndarray], labels: list[str], dt: float) -> dict[str, dict[str, float]]:
    """Each skill's metrics (see the module's docstring)."""
    label = np.array(labels)[np.minimum(np.round(rec["t"] * dt * 30).astype(int), len(labels) - 1)]
    done = rec["done"]
    out: dict[str, dict[str, float]] = {}
    for name in STRIKES:
        rows = []
        for e, t0, t1 in runs((label == name) & (rec["strike"] >= 0), done):
            hit = np.nonzero(rec["push"][t0:t1, e] > TOUCH)[0]
            if not len(hit):
                rows.append((0.0, np.nan, np.nan, np.nan, np.nan))
                continue
            th = t0 + int(hit[0])
            n = int(np.argmax(rec["push"][th : t1 + 1, e] <= TOUCH))
            u = rec["toward"][min(t0 + 3, t1 - 1), e]
            window = slice(th, min(th + round(RECOIL_WINDOW / dt), t1))
            behind = -min(0.0, float(((rec["pelvis"][window, e] - rec["pelvis"][t0, e]) @ u).min()))
            height = rec["fist"][th, e, 2] - rec["ref_fist"][th, e, 2]
            rows.append((1.0, rec["push"][th : th + 2, e].sum(), float(n * dt > PUSH), float(behind > RECOIL), height))
        a = np.array(rows, dtype=float)
        touched = a[a[:, 0] > 0]
        out[name] = {
            "n": len(a),
            "touch": a[:, 0].mean(),
            "j40": np.median(touched[:, 1]),
            "push": touched[:, 2].mean(),
            "recoil": touched[:, 3].mean(),
            "height_p90_cm": 100 * np.percentile(np.abs(touched[:, 4]), 90),
        }
    x = np.array((np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW), 0.0))
    frame = np.stack((x, np.cross((0.0, 0.0, 1.0), x), (0.0, 0.0, 1.0)), axis=1)
    step = 1 / 30 / dt  # policy steps per motion frame
    hold = (round(catch.OUT * step), round((catch.OUT + catch.HOLD) * step) + 1)
    after = round(catch.FRAMES * step) + round(HOME_AFTER / dt)
    checks = []
    for e, s, _end in runs(label == CATCH, done):
        if s + after >= len(done) or done[s : s + after, e].any():
            continue
        rel = {k: rec[k][:, e] - rec["torso"][:, e] for k in ("glove", "elbow", "fist")}
        ref = {k: rec[f"ref_{k}"][:, e] - rec["ref_torso"][:, e] for k in ("glove", "elbow")}
        off = (rel["glove"] - ref["glove"][s + hold[0] : s + hold[1]].mean(0)) @ frame
        inside = (np.hypot(off[:, 1], off[:, 2]) <= COVER) & (off[:, 0] >= DEPTH[0]) & (off[:, 0] <= DEPTH[1])
        arrive = np.nonzero(inside[s : s + hold[1]])[0]
        span = slice(s, s + round(catch.FRAMES * step) + 1)
        yaw, pitch, _ = Rotation.from_quat(rec["quat"][span, e][:, [1, 2, 3, 0]]).as_euler("ZYX", degrees=True).T
        still = (
            np.linalg.norm(rel["fist"][span] - rel["fist"][s - 1], axis=1).max() <= LEAD
            and np.abs(pitch - pitch[0]).max() <= PITCH
            and np.abs(yaw - yaw[0]).max() <= YAW
            and np.linalg.norm(rel["glove"][s + after] - rel["glove"][s - 1]) <= HOME
        )
        elbow = np.linalg.norm(
            rel["elbow"][s + hold[0] : s + hold[1]] - ref["elbow"][s + hold[0] : s + hold[1]], axis=1
        )
        checks.append(
            (
                inside[s + hold[0] : s + hold[1]].all(),
                len(arrive) > 0 and arrive[0] * dt <= ON_TIME,
                (elbow <= ELBOW).all(),
                still,
            )
        )
    c = np.array(checks, dtype=float)
    out[CATCH] = {"n": len(c), **dict(zip(("cover", "on_time", "elbow", "still"), c.mean(0), strict=True))}
    travel, slip = [], []
    for e, t0, t1 in runs(np.isin(label, MOVES), done):
        moved = rec["pelvis"][t1 - 1, e, :2] - rec["pelvis"][t0, e, :2]
        ref_moved = rec["ref_pelvis"][t1 - 1, e, :2] - rec["ref_pelvis"][t0, e, :2]
        travel.append(np.linalg.norm(moved - ref_moved))
        slip.append(rec["slip"][t0:t1, e].mean())
    out["footwork"] = {"n": len(travel), "travel_cm": 100 * np.median(travel), "slip": float(np.mean(slip))}
    for name in (*STRIKES, CATCH):
        out[name]["body_cm"] = 100 * rec["body_err"][label == name].mean()
    out["footwork"]["body_cm"] = 100 * rec["body_err"][np.isin(label, MOVES)].mean()
    falls = rec["fell"].sum()
    out["all"] = {"falls_per_min": falls / (done.shape[1] * done.shape[0] * dt / 60)}
    return {skill: {k: float(v) for k, v in m.items()} for skill, m in out.items()}


def compare(result: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]) -> bool:
    """Prints each metric against the baseline's; whether all pass."""
    ok = True
    for skill, metrics in result.items():
        for k, v in metrics.items():
            if k == "n":
                continue
            worse, tol, relative = TOLERANCE[k]
            base = baseline[skill][k]
            limit = tol * abs(base) if relative else tol
            passed = v >= base - limit if worse == "lower" else v <= base + limit
            ok &= passed
            print(f"  {skill:9s} {k:14s} {v:8.3f}  baseline {base:8.3f}  {'ok' if passed else 'FAIL'}")
    return ok


if __name__ == "__main__":
    result = measure(*rollout(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=1))
    for skill, metrics in result.items():
        print(skill, {k: round(v, 3) for k, v in metrics.items()})
    if len(sys.argv) > 3:  # noqa: PLR2004
        accepted = compare(result, json.loads(Path(sys.argv[3]).read_text()))
        print("accepted" if accepted else "REJECTED")
        sys.exit(0 if accepted else 1)
