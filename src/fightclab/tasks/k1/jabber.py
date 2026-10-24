"""A scripted jabber and its catcher: two T800 motions in step, for the duel task (``duel.py``).

The attacker stands at standing jab range and throws head jabs at one aim on a random tempo, at least MIN_GAP of guard
between them (the RL review: nothing but its own movement may tell a jab is coming). The defender holds its guard and
catches each jab with the rear glove (``catch.py``), starting CATCH_LEAD before the fist moves, as the K-1 coach
timed it. The attacker's rows are turned to face the defender and moved RANGE along its opponent line, where the
coach measured the catch against the jab.

    python -m fightclab.tasks.k1.jabber OUT SECONDS [SEED]

writes OUT_defender.csv and OUT_attacker.csv (30 fps rows as ``steps.py``'s, with .labels.txt); ``motion_npz t800``
makes each a motion.
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import fightclab.tasks  # noqa: F401  (before fightclab.motion)
from fightclab.motion import ROBOTS, model, set_pose
from fightclab.tasks.k1 import catch, steps
from fightclab.tasks.k1.footwork import OPPONENT_YAW
from fightclab.tasks.k1.footwork_motion import FPS

# m pelvis to pelvis: the standing jab's full reach to a T800 head (0.965 m, the coach) less the makiwara's depth: a jab
# punches through its target. At full reach a still head was touched by 31% of jabs, with 1.8 N s.
RANGE = 0.88
AIM = steps.Aim(1.60, 0.06, 0.0)  # the face's centre on a T800 head (docs/duel.md: 1.56-1.64 m, 0.045-0.075 m)
JAB = 1.2  # s, the jab command
MIN_GAP, MAX_GAP = 2.0, 4.0  # s of guard between jabs
CATCH_LEAD = 2  # frames at 30 fps before the fist moves: 0.067 s (the coach: 0.05-0.08 s)
ONSET = 0.02  # m of the fist toward the opponent: it has left the guard
Motion = tuple[NDArray[np.float64], list[str]]  # 30 fps rows and each row's command


def attacker(seconds: float, rng: np.random.Generator) -> Motion:
    """Jabs at AIM on a random tempo, about ``seconds`` long, facing the opponent at the origin's side."""
    commands: list[steps.Command] = []
    while sum(c[1] for c in commands) < seconds:
        commands += [("stop", float(rng.uniform(MIN_GAP, MAX_GAP))), ("jab", JAB, AIM)]
    commands.append(("stop", MIN_GAP))
    return steps.build(commands)


def onsets(rows: NDArray[np.float64], labels: list[str]) -> list[int]:
    """The frame each jab's fist leaves the guard."""
    m, d, adr = model(ROBOTS["t800"])
    fist = m.body("LINK_WRIST_END_L").id
    toward = np.array((np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW), 0.0))
    out = []
    for start in (i for i in range(1, len(labels)) if labels[i] == "jab" and labels[i - 1] != "jab"):
        reach = []
        for r in rows[start : start + round(JAB * FPS)]:
            set_pose(m, d, adr, r)
            reach.append(d.xpos[fist] @ toward)
        out.append(start + int(np.argmax(np.array(reach) - reach[0] > ONSET)))
    return out


def defender(length: int, starts: list[int]) -> Motion:
    """The guard for ``length`` frames with a catch from each of ``starts``."""
    caught, _ = steps.build([("catch", 2 * catch.FRAMES / FPS)])  # the catch, then the guard it comes back to
    rows = np.repeat(caught[-1:], length, axis=0)
    labels = ["stop"] * length
    for s in starts:
        rows[s : s + catch.FRAMES + 1] = caught[: catch.FRAMES + 1]
        labels[s : s + catch.FRAMES + 1] = ["catch"] * (catch.FRAMES + 1)
    return rows, labels


def face(rows: NDArray[np.float64], to: NDArray[np.float64]) -> NDArray[np.float64]:
    """``rows`` turned 180 deg about the vertical through their first pelvis, which is moved RANGE from ``to``."""
    turn = Rotation.from_euler("z", np.pi)
    toward = np.array((np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW), 0.0))
    p0 = rows[0, :3].copy()
    out = rows.copy()
    out[:, :3] = turn.apply(rows[:, :3] - p0) + np.r_[to[:2] + RANGE * toward[:2], p0[2]]
    out[:, 3:7] = (turn * Rotation.from_quat(rows[:, 3:7])).as_quat()
    return out


def duel(seconds: float, seed: int = 0) -> dict[str, Motion]:
    """The defender's and the attacker's rows and labels, as long as each other."""
    rows, labels = attacker(seconds, np.random.default_rng(seed))
    guard, guard_labels = defender(len(rows), [s - CATCH_LEAD for s in onsets(rows, labels)])
    return {"defender": (guard, guard_labels), "attacker": (face(rows, guard[0, :3]), labels)}


if __name__ == "__main__":
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # noqa: PLR2004
    for side, (rows, labels) in duel(float(sys.argv[2]), seed).items():
        path = Path(f"{sys.argv[1]}_{side}.csv")
        np.savetxt(path, rows, delimiter=",", fmt="%.6f")
        path.with_suffix(".labels.txt").write_text("\n".join(labels))
        print(f"{len(rows) / FPS:.1f} s -> {path}")
