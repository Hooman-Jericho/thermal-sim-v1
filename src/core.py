"""
core.py
-------
Base abstractions for thermal-sim-v1.

This module defines the class hierarchy every thermal system in this
project must follow. It intentionally mirrors the Gymnasium ``Env``
pattern (``reset`` / ``step``) so that this exact code can later grow,
with minimal changes, into the ``thermal_env.py`` RL environment used
in the RL training phase of the thesis (SAC / PPO / TD3 / CBF layer).

Design notes
------------
- ``ThermalSystem`` is an Abstract Base Class (ABC). It cannot be
  instantiated directly -- it only exists to guarantee that every
  concrete thermal system implements the same interface.
- State is kept explicit and typed (``SystemState``) instead of loose
  floats, so later code (loggers, RL wrappers, plots) can treat every
  system's history the same way regardless of how many state
  variables it has internally.
- Every subclass owns its own physics in ``_dynamics``; the base
  class owns the generic simulation loop (``simulate``) so that loop
  logic is written once and never duplicated per system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SystemState:
    """A single time-stamped snapshot of a thermal system.

    Attributes
    ----------
    t : float
        Simulation time, in seconds, since the system was reset.
    temperatures : dict[str, float]
        Named temperatures in the system, in Kelvin. Using a dict
        (instead of a single float) lets a two-body system report
        both bodies' temperatures without a different SystemState
        shape than a one-body system.
    """

    t: float
    temperatures: dict[str, float] = field(default_factory=dict)


class ThermalSystem(ABC):
    """Abstract base class for every thermal system in this project.

    A concrete subclass must implement:
      - ``reset()``       -> the system's initial ``SystemState``
      - ``_dynamics(state)`` -> the instantaneous rate of change of
        every temperature in the system (K/s), given the current
        state. This is where the physics equations live.

    The base class then implements the shared numerical machinery
    (``step`` and ``simulate``) on top of that single physics hook,
    using explicit forward-Euler integration -- the same fixed-dt
    stepping style Gymnasium environments use, on purpose, since this
    class will later BECOME an RL environment.
    """

    def __init__(self, dt: float = 1.0, seed: int | None = None) -> None:
        """
        Parameters
        ----------
        dt : float
            Integration time step, in seconds. Every ``step()`` call
            advances the simulation by exactly this much.
        seed : int, optional
            Seeds this system's own RNG (``self.rng``). Not used by
            either concrete system yet -- both are fully
            deterministic -- but plumbed through now so that adding
            domain randomization later (randomizing k, initial temps,
            etc. in ``reset()``/``_dynamics()``) is a one-line change
            in the subclass, not a signature change on every caller.
            This mirrors the ``seed`` argument Gymnasium's own
            ``Env.reset(seed=...)`` takes.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = dt
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state: SystemState = self.reset()

    @property
    def rng(self) -> np.random.Generator:
        """Seeded random generator for subclasses (domain randomization)."""
        return self._rng

    def reseed(self, seed: int | None) -> SystemState:
        """Re-seed the RNG and reset to a fresh initial state.

        This is the pattern a Gymnasium ``Env.reset(seed=...)`` needs:
        a new seed must produce a reproducible-but-different episode,
        not just a new random stream with no way back to a known
        state.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state = self.reset()
        return self._state

    # ------------------------------------------------------------------
    # Abstract interface -- every subclass MUST implement these.
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> SystemState:
        """Return the system to its initial condition and return it."""
        raise NotImplementedError

    @abstractmethod
    def _dynamics(self, state: SystemState) -> dict[str, float]:
        """Return dT/dt (K/s) for every named temperature in ``state``.

        This is the ONLY place physics equations should be written.
        Must return a dict with exactly the same keys as
        ``state.temperatures``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared machinery -- implemented once, inherited by every system.
    # ------------------------------------------------------------------

    @property
    def state(self) -> SystemState:
        """The system's current state (read-only view)."""
        return self._state

    def step(self) -> SystemState:
        """Advance the simulation by one ``dt`` using forward Euler.

        T(t + dt) = T(t) + dT/dt * dt

        Forward Euler is the simplest correct integrator and is what
        this project uses throughout (matching the discrete-time,
        fixed-dt convention the later RL environment will need).
        """
        rates = self._dynamics(self._state)
        if rates.keys() != self._state.temperatures.keys():
            raise RuntimeError(
                "_dynamics() must return a rate for every temperature "
                f"in state. Expected {set(self._state.temperatures)}, "
                f"got {set(rates)}."
            )

        new_temps = {
            name: T + rates[name] * self.dt
            for name, T in self._state.temperatures.items()
        }
        self._state = SystemState(t=self._state.t + self.dt, temperatures=new_temps)
        return self._state

    def simulate(self, duration: float) -> list[SystemState]:
        """Run the system forward for ``duration`` seconds from NOW.

        Returns the full trajectory (including the current state as
        the first entry) as a list of ``SystemState``, so callers can
        log it, plot it, or feed it straight to W&B later without any
        conversion.
        """
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

        n_steps = int(round(duration / self.dt))
        trajectory = [self._state]
        for _ in range(n_steps):
            trajectory.append(self.step())
        return trajectory
