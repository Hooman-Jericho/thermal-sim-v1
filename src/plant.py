"""The actuated thermal plant controlled by the thesis's RL policy.

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

import math
from dataclasses import dataclass

from src.core import SystemState, ThermalSystem, check_euler_stability


def _finite_float(name: str, value: float) -> float:
    """Return ``value`` as a float, raising if it is not a finite number."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass
class PlantConfig:
    """Physical parameters for :class:`ActuatedThermalPlant`.

    Validated on construction: all values must be finite, ``mCp > 0``,
    ``k_loss >= 0``, ``dt > 0``, and forward Euler must be stable
    (``dt * k_loss / mCp < 2``).
    """

    mCp: float  # Thermal capacitance, J/K
    k_loss: float  # Heat-loss coefficient to ambient, W/K
    T_amb: float  # Ambient temperature, deg C
    dt: float = 1.0  # Integration time step, s

    def __post_init__(self) -> None:
        """Validate the parameters (see the class docstring)."""
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if the parameters are unphysical or unstable."""
        for name in ("mCp", "k_loss", "T_amb", "dt"):
            _finite_float(name, getattr(self, name))
        if self.mCp <= 0:
            raise ValueError(f"mCp must be > 0, got {self.mCp}")
        if self.k_loss < 0:
            raise ValueError(f"k_loss must be >= 0, got {self.k_loss}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        check_euler_stability(self.dt, self.k_loss / self.mCp, "k_loss / mCp")


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
        config.validate()  # the dataclass is mutable; re-check at use time
        self.cfg = config
        self.initial_temp = initial_temp if initial_temp is not None else config.T_amb
        super().__init__(dt=config.dt, seed=seed)

    def reset(self) -> SystemState:
        """Return the initial state: ``T = initial_temp`` (deg C) at ``t = 0``."""
        return SystemState(t=0.0, temperatures={"T": self.initial_temp})

    def _dynamics(
        self, state: SystemState, u: float = 0.0, d: float = 0.0
    ) -> dict[str, float]:
        """Return ``dT/dt`` (K/s) from the energy balance for inputs ``u``, ``d``.

        ``u`` is the heater power (W) and ``d`` the load disturbance (W).
        """
        T = state.temperatures["T"]
        dT_dt = (u - self.cfg.k_loss * (T - self.cfg.T_amb) - d) / self.cfg.mCp
        return {"T": dT_dt}

    def step(self, u: float = 0.0, d: float = 0.0) -> SystemState:
        """Advance one ``dt`` given a control input ``u`` and disturbance ``d``.

        Parameters
        ----------
        u : float
            Heater power, in W; must be finite.
        d : float
            Unmeasured load disturbance, in W; must be finite.

        Returns
        -------
        SystemState
            The new state, at time ``t + dt``.

        Raises
        ------
        ValueError
            If ``u`` or ``d`` is NaN or infinite.
        TypeError
            If ``u`` or ``d`` is not a real number.

        Notes
        -----
        Overrides ``ThermalSystem.step()`` (which takes no arguments)
        because this plant is actuated: the base class's no-argument
        loop is for the passive systems in ``systems.py``. This is
        exactly the signature ``Env.step(action)`` will need later,
        so no further interface change is expected when this becomes
        ``thermal_env.py``.
        """
        u = _finite_float("u", u)
        d = _finite_float("d", d)
        rates = self._dynamics(self._state, u=u, d=d)
        new_temps = {
            name: T + rates[name] * self.dt
            for name, T in self._state.temperatures.items()
        }
        self._state = SystemState(t=self._state.t + self.dt, temperatures=new_temps)
        return self._state

    def sample_disturbance(self, std: float) -> float:
        """Draw one UNMEASURED load-disturbance sample from this plant's own RNG.

        Parameters
        ----------
        std : float
            Standard deviation of the zero-mean Gaussian sample, in W.

        Returns
        -------
        float
            The disturbance sample, in W.

        Notes
        -----
        Using ``self.rng`` (seeded, private to this instance) rather
        than global ``np.random`` means two plants with different
        seeds never share a disturbance stream, and a given seed's
        episode is bit-for-bit reproducible regardless of what else
        has drawn from ``np.random`` elsewhere in the process.
        """
        return float(self.rng.normal(0.0, std))

    def max_physical_rate(self, u_max: float) -> float:
        """Return the loosest physically possible |dT/dt| (K/s) at full actuation.

        ``u_max`` is the maximum heater power, in W. Used by the
        physics-informed metrics to sanity-check that a learned model's
        *implied* rate of change never exceeds what the plant's own
        energy balance permits -- a check that is independent of the
        unmeasured disturbance ``d``.
        """
        return u_max / self.cfg.mCp
