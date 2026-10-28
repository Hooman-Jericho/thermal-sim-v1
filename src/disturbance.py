"""Load disturbance whose statistics do not depend on the time step.

Plant v1 drew a fresh white-noise value every step, so the temperature variance
scaled with ``dt`` (measured in the Day 48 validation: 0.00503 at dt = 1 s,
0.00251 at dt = 0.5 s): ``sigma_d`` meant nothing without the step it was chosen
for. Here the load is an Ornstein-Uhlenbeck process, sampled exactly:

    d_{k+1} = rho * d_k + sigma * sqrt(1 - rho**2) * eps_k,   rho = exp(-dt / tau)

Its stationary law is ``N(0, sigma**2)`` for every ``dt`` and ``tau``, so the
planner's "d ~ N(0, sigma_d)" holds as the *marginal*, and ``tau`` (seconds) states
how long a load persists. Halving ``dt`` changes the sampling, not the physics.
"""

from __future__ import annotations

import math

import numpy as np


class OUDisturbance:
    """Stationary Ornstein-Uhlenbeck load, exactly discretised.

    Parameters
    ----------
    std : float
        Stationary standard deviation, in W (``>= 0``).
    tau : float
        Correlation time, in s (``> 0``).
    dt : float
        Sampling step, in s (``> 0``).

    Notes
    -----
    The random generator is passed to :meth:`next` on every call instead of
    being stored. A plant's ``reseed()`` replaces its generator, and a stored
    reference would then silently keep using the old stream.
    """

    def __init__(self, std: float, tau: float, dt: float) -> None:
        for name, v in (("std", std), ("tau", tau), ("dt", dt)):
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if std < 0:
            raise ValueError(f"std must be >= 0, got {std}")
        if tau <= 0 or dt <= 0:
            raise ValueError(f"tau and dt must be > 0, got tau={tau}, dt={dt}")
        self.std, self.tau, self.dt = std, tau, dt
        self.rho = math.exp(-dt / tau)
        self._innovation = std * math.sqrt(1.0 - self.rho**2)
        self._prev: float | None = None

    def reset(self) -> None:
        """Forget the past: the next value is drawn from the stationary law."""
        self._prev = None

    def next(self, rng: np.random.Generator) -> float:
        """Return the next load value (W), drawing from ``rng``."""
        eps = float(rng.standard_normal())
        if self._prev is None:
            d = self.std * eps
        else:
            d = self.rho * self._prev + self._innovation * eps
        self._prev = d
        return d
