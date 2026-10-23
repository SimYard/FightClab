"""Footwork on command: motion matching over the footwork reference, composing a new motion for the tracking policy.

    uv run python -m fightclab.tasks.k1.steer REFERENCE.csv OUT.csv forward:3 left:2 pivot_left:2 stop:1 ...

Every STEP seconds the composer picks what comes next: the reference frame whose legs are posed like the current
one and whose next HORIZON moves as commanded, in the opponent's frame. If simply going on is nearly as good, the
current piece goes on; otherwise the chosen piece is turned and moved to continue from where the motion is, after a
JOIN-long blend. Planted feet are held still again at the end (``motion.lock_feet``). OUT.csv's rows are at
30 fps like the reference's, OUT.labels.txt names each row's command; ``motion_npz t800 OUT.csv OUT.npz`` makes it a
motion for the tracking task.

ponytail: offline, from a fixed list of commands; a fight needs the same choice made online, a few frames ahead.
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fightclab.motion import ROBOTS, contacts, lock_feet
from fightclab.tasks.k1.footwork import OPPONENT_YAW
from fightclab.tasks.k1.footwork_motion import FPS, blend, heading, place

# What each command asks: (forward m/s, left m/s, turn rad/s) in the opponent's frame. Between the reference's
# medians and 75th percentiles over 1 s (forward/back 0.26-0.43 m, sideways 0.20-0.34 m, turns 0.33-0.49 rad).
COMMANDS = {
    "stop": (0.0, 0.0, 0.0), "forward": (0.35, 0.0, 0.0), "back": (-0.35, 0.0, 0.0), "left": (0.0, 0.27, 0.0),
    "right": (0.0, -0.27, 0.0), "pivot_left": (0.0, 0.0, 0.45), "pivot_right": (0.0, 0.0, -0.45),
}  # fmt: skip
# The move is matched over the piece actually taken: footwork steps and pauses, so a frame whose next second holds a
# step often stands still for its first half, and choosing every half second on a one-second horizon only ever
# took the pauses.
STEP = HORIZON = 1.0  # s between choices, and the span whose move must match the command (0.5 s: a seam every step)
JOIN = 0.2  # s of blend into a new piece
MARGIN = 2.0  # how much better (in cost units) a jump must be than going on
MOVE_WEIGHT = 4.0  # the commanded move against the pose match
MOVE_SCALE = np.array((0.05, 0.05, 0.08))  # m, m, rad over HORIZON: a cost unit of the move


def features(rows: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per frame: the pose to match, and the move over the next HORIZON (forward, left, turn).

    The pose: the feet relative to the pelvis in its heading frame, the pelvis' velocity in the opponent's frame.
    """
    feet = contacts(rows, ROBOTS["t800"])[0]  # (2, N, 3)
    yaw = np.unwrap(heading(rows[:, 3:7]))
    c, s = np.cos(yaw), np.sin(yaw)
    rel = feet - rows[None, :, :3]
    feet_h = np.stack((c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1], rel[..., 2]), -1)
    opp = yaw + OPPONENT_YAW
    co, so = np.cos(opp), np.sin(opp)
    v = np.gradient(rows[:, :2], axis=0) * FPS
    vel = np.stack((co * v[:, 0] + so * v[:, 1], -so * v[:, 0] + co * v[:, 1]), -1)
    pose = np.concatenate((feet_h.transpose(1, 0, 2).reshape(len(rows), 6), vel), -1)
    h = round(HORIZON * FPS)
    end = np.minimum(np.arange(len(rows)) + h, len(rows) - 1)
    d = rows[end, :2] - rows[:, :2]
    move = np.stack((co * d[:, 0] + so * d[:, 1], -so * d[:, 0] + co * d[:, 1], yaw[end] - yaw), -1)
    return pose, move


def compose(rows: NDArray[np.float64], commands: list[tuple[str, float]]) -> tuple[NDArray[np.float64], list[str]]:
    """The motion that does ``commands`` ((name, seconds) in order) with pieces of ``rows``, and each row's name."""
    pose, move = features(rows)
    scale = pose.std(0) + 1e-6
    usable = np.arange(len(rows)) < len(rows) - round(HORIZON * FPS)  # a piece needs a future to run into
    n = round(STEP * FPS)
    i = int(np.argmin(np.where(usable, np.abs(move / MOVE_SCALE).sum(1), np.inf)))  # start standing
    out, labels = [rows[i : i + 1]], ["stop"]
    for name, seconds in commands:
        want = np.array(COMMANDS[name]) * HORIZON
        for _ in range(round(seconds / STEP)):
            cost = (((pose - pose[i]) / scale) ** 2).mean(1) + MOVE_WEIGHT * (((move - want) / MOVE_SCALE) ** 2).mean(1)
            cost[~usable] = np.inf
            j = int(np.argmin(cost))
            if cost[i + 1] <= cost[j] + MARGIN:  # go on
                piece = place(rows[i : i + n + 1], out[-1][-1])[1:]
            else:  # jump
                piece = place(rows[j : j + n], out[-1][-1])
                piece = np.concatenate((blend(out[-1][-1], piece[0], JOIN), piece))
                i = j - 1
            out.append(piece)
            labels += [name] * len(piece)
            i += n
    motion = np.concatenate(out)
    return lock_feet(motion, ROBOTS["t800"]), labels


if __name__ == "__main__":
    reference, out = np.loadtxt(sys.argv[1], delimiter=","), Path(sys.argv[2])
    commands = [(c.split(":")[0], float(c.split(":")[1])) for c in sys.argv[3:]]
    motion, labels = compose(reference, commands)
    np.savetxt(out, motion, delimiter=",", fmt="%.6f")
    out.with_suffix(".labels.txt").write_text("\n".join(labels))
    # What each command got: the composed motion's mean move per second in the opponent's frame.
    _, got = features(motion)
    for name in dict(commands):
        rows = [k for k, label in enumerate(labels) if label == name]
        print(f"{name:12s} asked {COMMANDS[name]}, got {tuple(np.round(got[rows].mean(0) / HORIZON, 2).tolist())} /s")
    print(f"{len(motion) / FPS:.1f} s -> {out}")
