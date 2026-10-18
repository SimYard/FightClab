"""Simulation settings shared by FightClab tasks."""

from dataclasses import dataclass

import mujoco
from mjlab.sim import MujocoCfg


@dataclass
class Rk4MujocoCfg(MujocoCfg):
    """MujocoCfg with the RK4 integrator, which mjlab's config does not offer (mujoco_warp's step has it).

    At the 5 ms step the first-order integrators (euler, implicitfast) do not conserve momentum when the
    body spins fast. A jump policy learned to whip the torso at ~60 rad/s and gained 4-7 m/s of
    center-of-mass speed per 20 ms with no contact at all, then flew 8 m. From that state the error per
    control step is 6.9 m/s with implicitfast at 5 ms, still 0.8 m/s at 1 ms, and 0.01 m/s with RK4 at 5 ms.
    """

    def apply(self, model: mujoco.MjModel) -> None:
        """Mjlab's settings, then RK4."""
        super().apply(model)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
