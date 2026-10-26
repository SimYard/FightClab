"""Per-skill checks of a T800 K-1 tracker: a fine-tune is accepted only if no skill falls below the accepted one's.

    uv run python tools/skill_gate.py CHECKPOINT OUT.json [BASELINE.json]

Rolls the policy out in Fightclab-K1-Makiwara-T800's training env (pushes off, deterministic actions, seed SEED;
ENVS envs for SECONDS) and measures each skill of the motion by its labels, counting only skills whose motion frames
follow one another from the frame before to the frame after (no reset or wrap of the motion cut them):
- strikes (jab, jab_body, step_jab): the share of strikes touching the pad, the first 40 ms of impulse (N s, median),
  the share pushing longer than 80 ms, the recoil tail (pelvis more than 1.75 cm behind its pre-strike spot within
  0.5 s of the touch), the fist's height error at the touch (cm, p90 of its size), the share of hits that slid along
  the pad past ``makiwara.SLIDE_FREE`` and the slide (cm, median; the makiwara's own measure), the share back BACK
  in front of the pad BACK_AFTER after the touch;
- catch: the K-1 coach's checks (``docs/duel.md``), each as the share of catches passing; positions relative to the
  torso (the tracked bodies don't include the head) and the elbow-pitch link (the tracked elbow);
- footwork moves: the pelvis travel's error against the reference's over a move (cm, median), and the feet-slip cost;
- every skill: the mean body position error (cm); overall falls per minute.
Each record pairs the robot with the reference frame it was just tracking. A skill missing from the rollout, or never
touching the pad, gets NaN, which fails. With BASELINE.json, prints each metric against it with TOLERANCE (or against
BOUNDS) and exits 1 if any fails. A fine-tune that loses a skill without falling (duel2: the jab's 40 ms impulse
20 -> 10 N s) shows in no training curve, only here. A checkpoint from before the tasks saw an opponent loads only
widened by ``tools/expand_inputs.py``.
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
from fightclab.tasks.k1.strikes import STRIKES, H

TASK = "Fightclab-K1-Makiwara-T800"
ENVS, SECONDS = 512, 15.0  # about 800 strikes and 500 catches
SEED = 0  # one set of starts and noises for every checkpoint: the comparison is paired
TOUCH = 1e-4  # N s of push in a step that counts as touching the pad
PUSH, RECOIL, RECOIL_WINDOW = 0.08, 0.0175, 0.5  # s, m, s
COVER, DEPTH, ON_TIME, ELBOW = 0.025, (-0.03, 0.04), 0.15, 0.03  # the coach's catch checks (m, m, s, m)
LEAD, PITCH, YAW, HOME, HOME_AFTER = 0.025, 5.0, 4.0, 0.04, 0.15  # (m, deg, deg, m, s)
BACK, BACK_AFTER = 0.03 * H, 0.1  # the fist this far in front of the pad this soon after its touch (m, s; the coach)
# Per metric: which way is worse, how much worse than the baseline passes, and whether that is a fraction of it.
# Two seeds of one checkpoint differed by 0.03 in rates, 0.8 N s in impulse and 1-5 falls in 128 env-minutes.
TOLERANCE: dict[str, tuple[str, float, bool]] = {
    "touch": ("lower", 0.05, False),
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
    "falls_per_min": ("higher", 0.05, False),
}
# Metrics held to bounds instead of the baseline: which way is worse, and the bound. The jab's first 40 ms: the coach's
# good jab is 10-18 N s, 12 his starting floor; duel5's 19-24 came from the arm driving on into the pad while the fist
# slid up it (13-18 cm, 19% pushes). The slide: the share past SLIDE_FREE, at most the RL review's 0.2, and its median
# under SLIDE_FREE (the coach). The way back: as many back as the slide's share leaves; a starting value.
BOUNDS: dict[str, tuple[str, float]] = {
    "j40": ("lower", 12.0),
    "slide": ("higher", 0.2),
    "slide_cm": ("higher", 100 * makiwara.SLIDE_FREE),
    "back": ("lower", 0.8),
}
MOVES = tuple(k for k, v in COMMANDS.items() if v is not None)


def rollout(checkpoint: str) -> tuple[dict[str, np.ndarray], list[str], float]:
    """Per step and env: what the checks need, at the reference frame just tracked; the labels; the policy step."""
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = ENVS
    cfg.seed = SEED
    cfg.events.pop("push_robot")
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env)
    runner = load_runner_cls(TASK)(venv, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    motion = env.command_manager.get_term("motion")
    assert isinstance(motion, MotionCommand)
    pad = env.reward_manager.get_term_cfg(makiwara.REWARD).func
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
    keys = ("t", "fell", "strike", "push", "toward", "slide", "gap", "body_err", "slip", "quat", *body)
    keys = (*keys, *(f"ref_{k}" for k in body))
    rec: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    obs = venv.get_observations()
    for _ in range(round(SECONDS / env.step_dt)):
        with torch.no_grad():
            obs, *_ = venv.step(policy(obs))
        # The step's rewards and metrics came before the command moved on a frame: that frame is the one tracked.
        t = (motion.time_steps - 1).clamp(min=0)
        force = env.scene[makiwara.SENSOR].data.force_history[:, 0]
        rec["t"].append(t)
        rec["fell"].append(env.termination_manager.terminated.clone())
        rec["strike"].append(pad.strike[t])
        rec["push"].append((force * pad.toward[:, None]).sum(-1).abs().sum(-1) * makiwara.PHYSICS_DT)
        rec["toward"].append(pad.toward.clone())
        rec["slide"].append(pad.slide.clone())
        rec["gap"].append(pad.gap.clone())
        rec["body_err"].append(motion.metrics["error_body_pos"].clone())
        rec["slip"].append(-env.reward_manager._step_reward[:, slip])
        rec["quat"].append(motion.robot_body_quat_w[:, body["torso"]].clone())
        for k, i in body.items():
            rec[k].append(motion.robot_body_pos_w[:, i].clone())
            rec[f"ref_{k}"].append(motion.motion.body_pos_w[t, i])  # no env origin: only differences and heights
    labels = Path(motion.cfg.motion_file).with_suffix(".labels.txt").read_text().split()
    return {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}, labels, env.step_dt


def runs(mask: np.ndarray, follows: np.ndarray) -> list[tuple[int, int, int]]:
    """(env, first record, record after) of each run of ``mask`` that no reset or wrap of the motion cut.

    ``follows``: each record's frame is the one after the previous record's; it must hold from the record before the
    run to the record after it.
    """
    out = []
    steps, envs = mask.shape
    for e in range(envs):
        start = None
        for k in range(1, steps):
            if not follows[k, e]:
                start = None
            elif mask[k, e] and not mask[k - 1, e]:
                start = k
            elif not mask[k, e] and mask[k - 1, e] and start is not None:
                out.append((e, start, k))
                start = None
    return out


def mean(x: np.ndarray) -> float:
    """The mean, NaN if empty."""
    return float(np.mean(x)) if len(x) else float("nan")


def strike_metrics(rec: dict[str, np.ndarray], spans: list[tuple[int, int, int]], dt: float) -> dict[str, float]:
    """A strike's metrics over its runs ``spans``."""
    rows = []
    for e, t0, t1 in spans:
        hit = np.nonzero(rec["push"][t0:t1, e] > TOUCH)[0]
        if not len(hit):
            rows.append((0.0, *(np.nan,) * 6))
            continue
        th = t0 + int(hit[0])
        off = rec["push"][th:t1, e] <= TOUCH
        n = int(np.argmax(off)) if off.any() else len(off)  # still touching at the strike's end: all of it
        u = rec["toward"][min(t0 + 3, t1 - 1), e]
        window = slice(th, min(th + round(RECOIL_WINDOW / dt), t1))
        behind = -min(0.0, float(((rec["pelvis"][window, e] - rec["pelvis"][t0, e]) @ u).min()))
        height = rec["fist"][th, e, 2] - rec["ref_fist"][th, e, 2]
        slide = rec["slide"][min(th + makiwara.SLIDE_WINDOW, t1 - 1), e]  # the makiwara's, once its window is over
        back = rec["gap"][min(th + round(BACK_AFTER / dt), t1 - 1), e] >= BACK
        rows.append(
            (1.0, rec["push"][th : th + 2, e].sum(), float(n * dt > PUSH), float(behind > RECOIL), height, slide, back)
        )
    a = np.array(rows, dtype=float).reshape(-1, 7)
    hits = a[a[:, 0] > 0]
    return {
        "n": len(a),
        "touch": mean(a[:, 0]),
        "j40": float(np.median(hits[:, 1])) if len(hits) else float("nan"),
        "push": mean(hits[:, 2]),
        "recoil": mean(hits[:, 3]),
        "height_p90_cm": 100 * float(np.percentile(np.abs(hits[:, 4]), 90)) if len(hits) else float("nan"),
        "slide": mean(hits[:, 5] > makiwara.SLIDE_FREE),
        "slide_cm": 100 * float(np.median(hits[:, 5])) if len(hits) else float("nan"),
        "back": mean(hits[:, 6]),
    }


def catch_metrics(
    rec: dict[str, np.ndarray], spans: list[tuple[int, int, int]], follows: np.ndarray, dt: float
) -> dict[str, float]:
    """The coach's catch checks, each as the share of the catches ``spans`` passing."""
    x = np.array((np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW), 0.0))
    frame = np.stack((x, np.cross((0.0, 0.0, 1.0), x), (0.0, 0.0, 1.0)), axis=1)
    step = 1 / 30 / dt  # policy steps per motion frame
    hold = (round(catch.OUT * step), round((catch.OUT + catch.HOLD) * step) + 1)
    after = round(catch.FRAMES * step) + round(HOME_AFTER / dt)
    checks = []
    for e, s, _end in spans:
        if s + after >= len(follows) or not follows[s + 1 : s + after + 1, e].all():
            continue
        rel = {k: rec[k][:, e] - rec["torso"][:, e] for k in ("glove", "elbow", "fist")}
        ref = {k: rec[f"ref_{k}"][:, e] - rec["ref_torso"][:, e] for k in ("glove", "elbow")}
        held = slice(s + hold[0], s + hold[1])
        off = (rel["glove"] - ref["glove"][held].mean(0)) @ frame
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
        elbow = np.linalg.norm(rel["elbow"][held] - ref["elbow"][held], axis=1)
        arrived = len(arrive) > 0 and arrive[0] * dt <= ON_TIME
        checks.append((inside[held].all(), arrived, (elbow <= ELBOW).all(), still))
    c = np.array(checks, dtype=float).reshape(-1, 4)
    return {"n": len(c), **{k: mean(c[:, i]) for i, k in enumerate(("cover", "on_time", "elbow", "still"))}}


def measure(rec: dict[str, np.ndarray], labels: list[str], dt: float) -> dict[str, dict[str, float]]:
    """Each skill's metrics (see the module's docstring)."""
    label = np.array(labels)[np.minimum(np.round(rec["t"] * dt * 30).astype(int), len(labels) - 1)]
    follows = np.zeros(rec["t"].shape, dtype=bool)
    follows[1:] = rec["t"][1:] == rec["t"][:-1] + 1
    out = {name: strike_metrics(rec, runs((label == name) & (rec["strike"] >= 0), follows), dt) for name in STRIKES}
    out[CATCH] = catch_metrics(rec, runs(label == CATCH, follows), follows, dt)
    travel, slip = [], []
    for e, t0, t1 in runs(np.isin(label, MOVES), follows):
        moved = rec["pelvis"][t1 - 1, e, :2] - rec["pelvis"][t0, e, :2]
        ref_moved = rec["ref_pelvis"][t1 - 1, e, :2] - rec["ref_pelvis"][t0, e, :2]
        travel.append(np.linalg.norm(moved - ref_moved))
        slip.append(rec["slip"][t0:t1, e].mean())
    median = float(np.median(travel)) if travel else float("nan")
    out["footwork"] = {"n": len(travel), "travel_cm": 100 * median, "slip": mean(np.array(slip))}
    for name in (*STRIKES, CATCH):
        out[name]["body_cm"] = 100 * mean(rec["body_err"][label == name])
    out["footwork"]["body_cm"] = 100 * mean(rec["body_err"][np.isin(label, MOVES)])
    minutes = rec["t"].size * dt / 60
    out["all"] = {"falls_per_min": float(rec["fell"].sum()) / minutes}
    return {skill: {k: float(v) for k, v in m.items()} for skill, m in out.items()}


def compare(result: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]) -> bool:
    """Prints each metric against the baseline's; whether all pass (NaN on either side fails)."""
    ok = True
    for skill, metrics in result.items():
        for k, v in metrics.items():
            if k == "n":
                continue
            if k in BOUNDS:
                worse, ref = BOUNDS[k]
                passed, what = v >= ref if worse == "lower" else v <= ref, "bound"
            else:
                worse, tol, relative = TOLERANCE[k]
                ref, what = baseline[skill][k], "baseline"
                limit = tol * abs(ref) if relative else tol
                passed = v >= ref - limit if worse == "lower" else v <= ref + limit
            ok &= passed
            print(f"  {skill:9s} {k:14s} {v:8.3f}  {what:8s} {ref:8.3f}  {'ok' if passed else 'FAIL'}")
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
