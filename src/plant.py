"""
plant.py
--------
The ACTUATED thermal plant: the actual system this thesis's RL policy
(SAC + physics-informed reward + CBF-QP safety layer) will control.

``NewtonianCoolingSystem`` and ``HeatExchangerSystem`` (systems.py) are
passive -- they have no control input. Before any controller (classical
PID/MPC or the RL policy) can be designed or a surrogate dynamics model
can be learned, the plant needs an explicit actuator. This module adds
exactly that, as a ``ThermalSystem`` subclass, so it fits the same
``reset()`` / ``step()`` / seeded-``rng`` contract as everything else in
``src/`` and can become ``thermal_env.py``'s internal engine later by
adding a reward function and an ``observation``/``done`` wrapper -- not
by rewriting the physics again.

Physics (single lumped-capacitance body, energy balance):

    mCp * dT/dt = u(t) - k_loss * (T - T_amb) - d(t)

    u(t) : controllable heater power [W]      -- the future RL action
    d(t) : exogenous thermal load disturbance [W] -- UNMEASURED
    k_loss * (T - T_amb) : passive heat loss to ambient, same form as
                           ``NewtonianCoolingSystem`` in systems.py

``d(t)`` is drawn from THIS system's own seeded ``self.rng`` (inherited
from ``ThermalSystem``), never from the module-level ``np.random``
state, so that an episode's disturbance realization is fully
reproducible from its seed alone -- consistent with ``reseed()`` in
core.py.

Note on units: only temperature *differences* appear in the dynamics
(``T - T_amb``), so the model is scale-invariant between Celsius and
Kelvin. This module uses Celsius (matching the thesis's existing
config convention) rather than Kelvin (used by ``systems.py``) --
purely a labeling choice, not a physics change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import SystemState, ThermalSystem


@dataclass
class PlantConfig:
    """Physical parameters for :class:`ActuatedThermalPlant`."""

    mCp: float          # Thermal capacitance, J/K
    k_loss: float       # Heat-loss coefficient to ambient, W/K
    T_amb: float        # Ambient temperature, deg C
    dt: float = 1.0     # Integration time step, s


class ActuatedThermalPlant(ThermalSystem):
    """Single-body thermal plant with a controllable heater input.

    This is the plant whose *dynamics* the baseline ML models in
    ``experiments/baseline_dynamics_learning`` try to learn, and the
    same plant the thesis's RL policy will later act on. Keeping both
    on the identical class guarantees the baseline comparison and the
    RL results are about the *modeling/control approach*, not about
    two different simulated physical systems.
    """

    def __init__(
        self,
        config: PlantConfig,
        initial_temp: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = config
        self.initial_temp = (
            initial_temp if initial_temp is not None else config.T_amb
        )
        super().__init__(dt=config.dt, seed=seed)

    def reset(self) -> SystemState:
        return SystemState(t=0.0, temperatures={"T": self.initial_temp})

    def _dynamics(self, state: SystemState, u: float = 0.0, d: float = 0.0) -> dict[str, float]:
        T = state.temperatures["T"]
        dT_dt = (u - self.cfg.k_loss * (T - self.cfg.T_amb) - d) / self.cfg.mCp
        return {"T": dT_dt}

    def step(self, u: float = 0.0, d: float = 0.0) -> SystemState:  # type: ignore[override]
        """Advance one ``dt`` given a control input ``u`` and disturbance ``d``.

        Overrides ``ThermalSystem.step()`` (which takes no arguments)
        because this plant is actuated: the base class's no-argument
        loop is for the passive systems in ``systems.py``. This is
        exactly the signature ``Env.step(action)`` will need later,
        so no further interface change is expected when this becomes
        ``thermal_env.py``.
        """
        rates = self._dynamics(self._state, u=u, d=d)
        new_temps = {
            name: T + rates[name] * self.dt
            for name, T in self._state.temperatures.items()
        }
        self._state = SystemState(t=self._state.t + self.dt, temperatures=new_temps)
        return self._state

    def sample_disturbance(self, std: float) -> float:
        """Draw one UNMEASURED load-disturbance sample from this plant's own RNG.

        Using ``self.rng`` (seeded, private to this instance) rather
        than global ``np.random`` means two plants with different
        seeds never share a disturbance stream, and a given seed's
        episode is bit-for-bit reproducible regardless of what else
        has drawn from ``np.random`` elsewhere in the process.
        """
        return float(self.rng.normal(0.0, std))

    def max_physical_rate(self, u_max: float) -> float:
        """Loosest physically possible |dT/dt| (K/s) at full actuation.

        Used by the physics-informed metrics to sanity-check that a
        learned model's *implied* rate of change never exceeds what
        the plant's own energy balance permits -- a check that is
        independent of the unmeasured disturbance ``d``.
        """
        return u_max / self.cfg.mCp
