"""
systems.py
----------
Concrete thermal systems. Each class here implements ``ThermalSystem``
from ``core.py`` and owns exactly one piece of physics.

Two systems are provided, deliberately chosen to cover both the
"single-body cooling" case and the actual "heat exchange between two
bodies" case named in Day 6's task:

1. ``NewtonianCoolingSystem`` -- one body losing heat to a fixed
   ambient environment. Has a closed-form analytical solution, which
   is exactly what lets ``tests/test_core.py`` verify the numerical
   integrator is correct (not just "runs without crashing").

2. ``HeatExchangerSystem`` -- two bodies (e.g. a hot fluid loop and a
   cold fluid loop) exchanging heat with each other AND losing heat to
   ambient. This is the system that structurally resembles the actual
   thesis's thermal plant and is the one worth extending later.
"""

from __future__ import annotations

from src.core import SystemState, ThermalSystem


class NewtonianCoolingSystem(ThermalSystem):
    """A single body cooling toward a fixed ambient temperature.

    Physics (Newton's Law of Cooling):
        dT/dt = -k * (T - T_ambient)

    where k > 0 is a heat-transfer coefficient (1/s) folding in the
    body's surface area, its heat-transfer coefficient, its mass, and
    its specific heat capacity into a single lumped constant -- the
    standard "lumped capacitance" simplification used for a
    first thermal model.
    """

    def __init__(
        self,
        initial_temp: float = 350.0,
        ambient_temp: float = 293.15,
        k: float = 0.02,
        dt: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        initial_temp : float
            Starting temperature of the body, in Kelvin.
        ambient_temp : float
            Fixed ambient (room) temperature, in Kelvin.
        k : float
            Cooling rate constant, in 1/s. Larger k = faster cooling.
        dt : float
            Integration time step, in seconds.
        seed : int, optional
            Seeds ``self.rng`` (see ``ThermalSystem``). Unused for now
            -- this system is fully deterministic -- but here so a
            future randomized ``k``/``initial_temp`` doesn't require
            changing every call site.
        """
        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp
        self.k = k
        super().__init__(dt=dt, seed=seed)

    def reset(self) -> SystemState:
        return SystemState(t=0.0, temperatures={"body": self.initial_temp})

    def _dynamics(self, state: SystemState) -> dict[str, float]:
        T = state.temperatures["body"]
        dT_dt = -self.k * (T - self.ambient_temp)
        return {"body": dT_dt}


class HeatExchangerSystem(ThermalSystem):
    """Two bodies exchanging heat with each other and with ambient.

    A simple counter-flow-style lumped model: a "hot" body and a
    "cold" body exchange heat proportionally to their temperature
    difference (coefficient ``k_exchange``), while each also leaks
    heat to a shared ambient temperature (coefficient ``k_loss``).

    Physics:
        dT_hot/dt  = -k_exchange * (T_hot  - T_cold) - k_loss * (T_hot  - T_ambient)
        dT_cold/dt = +k_exchange * (T_hot  - T_cold) - k_loss * (T_cold - T_ambient)

    This is the shape of model this thesis's thermal plant actually
    needs: multiple coupled temperatures, not just one body cooling in
    isolation. It is the natural next step from
    ``NewtonianCoolingSystem`` and the more realistic of the two.
    """

    def __init__(
        self,
        hot_initial_temp: float = 360.0,
        cold_initial_temp: float = 290.0,
        ambient_temp: float = 293.15,
        k_exchange: float = 0.05,
        k_loss: float = 0.01,
        dt: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        hot_initial_temp, cold_initial_temp : float
            Starting temperatures of the two bodies, in Kelvin.
        ambient_temp : float
            Fixed ambient temperature both bodies leak heat toward.
        k_exchange : float
            Heat-transfer rate constant BETWEEN the two bodies (1/s).
        k_loss : float
            Heat-loss rate constant from EACH body to ambient (1/s).
        dt : float
            Integration time step, in seconds.
        seed : int, optional
            Seeds ``self.rng`` (see ``ThermalSystem``). Unused for now
            -- this system is fully deterministic -- but here so that
            future domain randomization (``k_exchange``/``k_loss``
            drawn per-episode, as the thesis's +/-10-15% ranges will
            need) is a one-line change, not a signature change.
        """
        self.hot_initial_temp = hot_initial_temp
        self.cold_initial_temp = cold_initial_temp
        self.ambient_temp = ambient_temp
        self.k_exchange = k_exchange
        self.k_loss = k_loss
        super().__init__(dt=dt, seed=seed)

    def reset(self) -> SystemState:
        return SystemState(
            t=0.0,
            temperatures={
                "hot": self.hot_initial_temp,
                "cold": self.cold_initial_temp,
            },
        )

    def _dynamics(self, state: SystemState) -> dict[str, float]:
        T_hot = state.temperatures["hot"]
        T_cold = state.temperatures["cold"]

        exchange = self.k_exchange * (T_hot - T_cold)
        dT_hot_dt = -exchange - self.k_loss * (T_hot - self.ambient_temp)
        dT_cold_dt = +exchange - self.k_loss * (T_cold - self.ambient_temp)

        return {"hot": dT_hot_dt, "cold": dT_cold_dt}
