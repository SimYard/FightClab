"""Helicopter kick reference from video: a Ginger Ninja tutorial clip through GVHMR and GMR, fitted to a robot.

Pipeline, each step in its own environment (``<robot>`` is g1, r1, h2, pm01, t800, t1 or asimov):
1. Cut the kick to a 30 fps clip with no cuts, the whole body in frame, a static camera.
2. GVHMR (~/GVHMR): ``python tools/demo/demo.py --video=<clip> -s`` -> hmr4d_results.pt
3. GMR (~/GMR): ``~/GMR/.venv/bin/python tools/gvhmr_to_csv.py <robot> <hmr4d_results.pt> /tmp/raw.csv``
4. This module: ``uv run python -m fightclab.tasks.helicopter_kick.video_motion <robot> /tmp/raw.csv /tmp/video.csv``
   (``--ground`` for motions that never leave the floor: ``motion.clean_ground`` instead of the kick's timing)
5. ``uv run python -m fightclab.tasks.helicopter_kick.motion_npz <robot> /tmp/video.csv``

The timing below was tuned on G1 and is used for every robot, but for the jumps of robots that cannot fly like it
(``JUMPS``).

What the cleanup fixes:
- the tutorial shows the kick in slow motion, about 2x (a 1.2 s flight would need a 1.8 m jump);
- monocular height drifts: on the ground the lowest foot point is put on the floor, using the contact
  phases read off the video (TAKEOFF and LANDING, in clip seconds);
- G1 cannot jump 42 cm like the person, so the flight is time-compressed to FLIGHT seconds, in which the
  center of mass flies a parabola;
- the push-off (PUSH_START to TAKEOFF) is retimed so the center of mass rises from the crouch with constant
  acceleration and reaches the flight's takeoff speed (~2.7 m/s) while the feet are still down; PUSH_SPEED
  sets its length. Before, the reference reached that speed only after takeoff, accelerating upward in the
  air, and tracking policies rose on their toes but never flew.

Fitting to G1's motors (check with ``python -m fightclab.feasibility``): as reconstructed, the kick needed up
to 5x the torque of G1's waist motors. The heavy torso has to follow a pelvis that tumbles, spins at up to
12 rad/s and swings asymmetric arms, and two small 5020 motors carry all of it. So the spin-up before
takeoff is slowed down, the yaw and the pelvis tilt smoothed, the waist roll/pitch damped and the
reconstruction jitter filtered out (``motion.fit``).

The kick stays head-high on purpose (heel ~1.4 m; a G1 opponent's head is at 1.23-1.29 m). A head kick needs
both the high jump and the lean away from the kicking leg, and those still ask the knees and the waist for
about 1.5x their torque. Damping the lean and smoothing harder fits the motors, but the heel then only gets
to 0.7-1.1 m. The tracking task tolerates the shortfall and lets the policy find how high G1 can kick.
"""

import sys

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from fightclab.feasibility import report
from fightclab.motion import FPS, JOINT_WINDOW, ROBOTS, clean_ground, fit, heights, lift_sunk_feet

G = 9.81
SLOWMO = 2.0
TAKEOFF, LANDING = 1.55, 2.75  # clip seconds, read off the video
FLIGHT = 0.55  # s on G1: a ~37 cm pelvis rise instead of the person's 42 cm
SPIN_UP_SLOW = 1.6  # the spin-up on the ground runs this much slower than the person
PUSH_START = 1.15  # clip second the push-off starts: the crouch's bottom
PUSH_SPEED = 1.0  # push-off length: 0.2 s at 1.0, ~1.7 g; at 2.0 it was 0.1 s and ~4.6 g, too fast to follow
HOLD = 0.4  # s of the final pose appended, so the motion settles instead of stopping dead
SUNK = 0.005  # m below the floor that the summary reports as a sunk foot
# FLIGHT and PUSH_SPEED of the robots that cannot jump like G1.
JUMPS = {
    # T1's crouch is shallow and its ankles barely point the toes (20 deg), so its center of mass rises only 10 cm in
    # the push-off, against G1's 21. That reaches ~1.3 m/s, a 0.3 s flight (~10 cm hop), not G1's 0.55 s; faster
    # push-offs were too short to survive the smoothing and left the takeoff speed short again.
    "t1": (0.30, 1.2),
    # Asimov's knee bends 86 deg and its ankles +-20: its push-off reaches ~1.5 m/s, a 0.4 s flight, not G1's 0.55 s.
    "asimov": (0.40, 1.2),
}


def push_start() -> float:
    """Output second the push-off starts."""
    return PUSH_START / SLOWMO * SPIN_UP_SLOW


def jump(robot: str) -> tuple[float, float]:
    """The robot's flight time and push-off speed."""
    return JUMPS.get(robot, (FLIGHT, PUSH_SPEED))


def phases(robot: str = "g1") -> tuple[float, float]:
    """Takeoff and landing in output (robot) seconds."""
    flight, push_speed = jump(robot)
    t0 = push_start() + (TAKEOFF - PUSH_START) / SLOWMO / push_speed
    return t0, t0 + flight


def clip_time(t: NDArray[np.float64], robot: str) -> NDArray[np.float64]:
    """Clip seconds shown at output seconds: spin-up slowed, push-off sped up, flight squeezed, landing real time."""
    t0, t1 = phases(robot)
    flight, push_speed = jump(robot)
    t_push = push_start()
    spin_up = t * SLOWMO / SPIN_UP_SLOW
    push = PUSH_START + (t - t_push) * SLOWMO * push_speed
    air = TAKEOFF + (t - t0) / flight * (LANDING - TAKEOFF)
    post = LANDING + (t - t1) * SLOWMO
    return np.select([t < t_push, t < t0, t < t1], [spin_up, push, air], post)


def resample(rows: NDArray[np.float64], src: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows linearly interpolated at clip frame positions ``src``; quaternions renormalized."""
    i = np.clip(np.floor(src).astype(int), 0, len(rows) - 2)
    w = (src - i)[:, None]
    q0, q1 = rows[i, 3:7], rows[i + 1, 3:7]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    out = (1 - w) * rows[i] + w * rows[i + 1]
    out[:, 3:7] = (1 - w) * q0 + w * q1
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=-1, keepdims=True)
    return out


def even_push(rows: NDArray[np.float64], robot: str) -> NDArray[np.float64]:
    """Retime the push-off so the grounded center of mass rises with constant acceleration from the crouch.

    The person's extension starts slow and ends fast; sped up enough to reach the takeoff speed, it asked for
    ~4.5 g in its last frames. Rising evenly from rest, the same height and takeoff speed take ~1.7 g.
    """
    t = np.arange(len(rows)) / FPS
    t_push, t0 = push_start(), phases(robot)[0]
    low, com = heights(rows, ROBOTS[robot])
    push = (t >= t_push) & (t <= t0)
    height = np.maximum.accumulate((com - low)[push])  # the grounded center of mass, made monotonic
    u = (t[push] - t[push][0]) / (t[push][-1] - t[push][0])
    target = height[0] + (height[-1] - height[0]) * u**2
    out = rows.copy()
    out[push] = resample(rows, np.interp(target, height, t[push]) * FPS)
    return out


def ground_and_fly(rows: NDArray[np.float64], robot: str) -> NDArray[np.float64]:
    """Lowest foot on the floor on the ground; in the air the center of mass flies a parabola.

    The parabola is for the center of mass, not the pelvis: the kicking leg swings up in the air, and with a
    ballistic pelvis the center of mass would rise with it, pushed by nothing.
    """
    t = np.arange(len(rows)) / FPS
    t0, t1 = phases(robot)
    low, com = heights(rows, ROBOTS[robot])
    above = com - rows[:, 2]  # center of mass over the pelvis, set by the pose
    z = rows[:, 2] - low
    air = (t > t0) & (t < t1)
    for ground in (t <= t0, t >= t1):  # smoothed apart, so the takeoff's kink stays out of the air
        z[ground] = savgol_filter(z[ground], JOINT_WINDOW, 3)
    com = z + above
    i0, i1 = np.flatnonzero(air)[[0, -1]] + (-1, 1)  # the ground frames the flight joins
    flight = t[i1] - t[i0]
    v0 = (com[i1] - com[i0]) / flight + G * flight / 2
    tau = t[air] - t[i0]
    z[air] = com[i0] + v0 * tau - G * tau**2 / 2 - above[air]
    out = rows.copy()
    out[:, 2] = z
    return out


def clean(rows: NDArray[np.float64], robot: str) -> NDArray[np.float64]:
    """A 30 fps clip-time robot motion -> 30 fps robot-time rows, fitted, grounded, with a G1-sized flight.

    Grounding comes after the smoothing: smoothing a grounded, ballistic height spread the takeoff's kink
    into the air, where the reference then kept accelerating upward with nothing to push against.
    """
    _t0, t1 = phases(robot)
    end = t1 + (len(rows) / FPS - LANDING) / SLOWMO
    t = np.arange(0.0, end, 1.0 / FPS)
    out = resample(rows, clip_time(t, robot) * FPS)
    out = np.concatenate([out, np.repeat(out[-1:], round(HOLD * FPS), axis=0)])
    return lift_sunk_feet(ground_and_fly(even_push(fit(out, ROBOTS[robot]), robot), robot), ROBOTS[robot])


if __name__ == "__main__":
    name, raw, out = sys.argv[1:4]
    robot = ROBOTS[name]
    grounded = sys.argv[4:] == ["--ground"]  # a motion without flight: the kick's timing does not apply
    data = np.loadtxt(raw, delimiter=",")
    rows = clean_ground(data, robot) if grounded else clean(data, name)
    np.savetxt(out, rows, delimiter=",", fmt="%.6f")
    low, com = heights(rows, robot)
    yaw = np.degrees(np.unwrap(2 * np.arctan2(rows[:, 5], rows[:, 6])))
    print(f"{len(rows)} frames, {len(rows) / FPS:.2f} s -> {out}")
    print(f"pelvis {rows[:, 2].min():.2f}..{rows[:, 2].max():.2f} m, turn {yaw[-1] - yaw[0]:+.0f} deg")
    if not grounded:  # the push-off has to reach the flight's takeoff speed while the feet are still down
        vz = np.diff(com) * FPS
        i0 = int(phases(name)[0] * FPS)
        print(
            "center of mass speed up between frames around takeoff:",
            " ".join(f"{v:+.1f}" for v in vz[i0 - 4 : i0 + 3]),
            "m/s",
        )
    print(f"feet below the floor by >5 mm in {(low < -SUNK).sum()} frames, worst {low.min() * 1000:.0f} mm")
    if name == "g1":  # ponytail: the inverse-dynamics report knows G1 only; generalize when tuning another robot
        print("G1 motors:")
        report(rows, FPS)
