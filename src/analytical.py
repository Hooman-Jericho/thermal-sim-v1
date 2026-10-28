"""Closed-form reference solutions for :class:`~src.plant.ActuatedThermalPlant`.

These functions exist only to *validate* the simulator: they give the exact
answer that the forward-Euler loop in ``plant.py`` is trying to approximate,
so the tests and ``experiments/physics_validation`` can measure the error
instead of trusting it.

The plant is the lumped energy balance

    mCp * dT/dt = u - k_loss * (T - T_amb) - d

For *constant* ``u`` and ``d`` this is a first-order linear ODE with

    tau  = mCp / k_loss                      (time constant, s)
    T_ss = T_amb + (u - d) / k_loss          (steady state, deg C)
    T(t) = T_ss + (T0 - T_ss) * exp(-t / tau)

Three different "answers" are provided, because they answer different
questions:

* :func:`continuous_response` -- the true solution of the ODE.
* :func:`euler_response` -- what forward Euler computes, in closed form. The
  simulator must match this one to machine precision (it is a test of the
  *code*).
* The gap between the two is the integration (truncation) error (it is a
  property of the *method*, and is first order in ``dt``).

When ``k_loss == 0`` there is no heat loss, the plant is a pure integrator and
there is no steady state; the functions handle that case explicitly.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.plant import PlantConfig


def time_constant(cfg: PlantConfig) -> float:
    """Return the thermal time constant ``tau = mCp / k_loss`` in seconds.

    Returns ``math.inf`` when ``k_loss == 0`` (no heat loss, no decay).
    """
    if cfg.k_loss == 0.0:
        return math.inf
    return cfg.mCp / cfg.k_loss


def steady_state(cfg: PlantConfig, u: float, d: float = 0.0) -> float:
    """Return the steady-state temperature ``T_amb + (u - d) / k_loss`` in deg C.

    Raises
    ------
    ValueError
        If ``k_loss == 0``: a lossless body has no steady state (it keeps
        heating or cooling for as long as ``u != d``).
    """
    if cfg.k_loss == 0.0:
        raise ValueError("k_loss == 0: the plant has no steady state")
    return cfg.T_amb + (u - d) / cfg.k_loss


def continuous_response(
    cfg: PlantConfig,
    t: ArrayLike,
    T0: float,
    u: float,
    d: float = 0.0,
) -> NDArray[np.float64]:
    """Return the exact ODE solution ``T(t)`` for constant ``u`` and ``d``.

    Parameters
    ----------
    cfg : PlantConfig
        Plant parameters.
    t : array_like
        Times, in seconds since the start (``t >= 0``).
    T0 : float
        Initial temperature, in deg C.
    u, d : float
        Constant heater power and load disturbance, in W.

    Returns
    -------
    numpy.ndarray
        ``T`` at every requested time, in deg C.
    """
    times = np.asarray(t, dtype=np.float64)
    if cfg.k_loss == 0.0:
        return np.asarray(T0 + (u - d) * times / cfg.mCp, dtype=np.float64)
    T_ss = steady_state(cfg, u, d)
    tau = time_constant(cfg)
    return np.asarray(T_ss + (T0 - T_ss) * np.exp(-times / tau), dtype=np.float64)


def euler_response(
    cfg: PlantConfig,
    steps: ArrayLike,
    T0: float,
    u: float,
    d: float = 0.0,
) -> NDArray[np.float64]:
    """Return what forward Euler computes after ``steps`` steps, in closed form.

    One Euler step maps ``T - T_ss`` to ``(1 - dt / tau) * (T - T_ss)``, so after
    ``k`` steps ``T_k = T_ss + (T0 - T_ss) * (1 - dt / tau) ** k``. With
    ``k_loss == 0`` every step adds the same ``(u - d) * dt / mCp``.

    Parameters
    ----------
    cfg : PlantConfig
        Plant parameters (``cfg.dt`` is the step size).
    steps : array_like of int
        Step counts ``k >= 0``.
    T0 : float
        Initial temperature, in deg C.
    u, d : float
        Constant heater power and load disturbance, in W.

    Returns
    -------
    numpy.ndarray
        ``T_k`` for every requested ``k``, in deg C.
    """
    k = np.asarray(steps, dtype=np.int64)
    if cfg.k_loss == 0.0:
        return np.asarray(T0 + (u - d) * cfg.dt * k / cfg.mCp, dtype=np.float64)
    T_ss = steady_state(cfg, u, d)
    ratio = 1.0 - cfg.dt * cfg.k_loss / cfg.mCp
    return np.asarray(T_ss + (T0 - T_ss) * np.power(ratio, k), dtype=np.float64)


def euler_stationary_variance(cfg: PlantConfig, sigma_d: float) -> float:
    """Return the stationary variance of ``T`` under per-step white disturbance.

    The simulator draws a fresh ``d_k ~ N(0, sigma_d**2)`` for every step and
    holds it constant during that step. With ``a = 1 - dt / tau`` the
    deviation from the mean obeys the AR(1) recursion

        x_{k+1} = a * x_k - (dt / mCp) * d_k

    whose stationary variance is ``(dt / mCp)**2 * sigma_d**2 / (1 - a**2)``.
    For small ``dt`` this is about ``dt * sigma_d**2 / (2 * k_loss * mCp)``: it
    scales with ``dt``, so ``sigma_d`` only has a physical meaning together
    with the time step it was chosen for.

    Raises
    ------
    ValueError
        If ``k_loss == 0`` (the variance grows without bound) or
        ``sigma_d < 0``.
    """
    if cfg.k_loss == 0.0:
        raise ValueError("k_loss == 0: the temperature variance is unbounded")
    if sigma_d < 0:
        raise ValueError(f"sigma_d must be >= 0, got {sigma_d}")
    rate_dt = cfg.dt * cfg.k_loss / cfg.mCp  # dt / tau, in (0, 2) by Euler guard
    one_minus_a_sq = rate_dt * (2.0 - rate_dt)  # = 1 - (1 - rate_dt) ** 2
    return float((cfg.dt / cfg.mCp) ** 2 * sigma_d**2 / one_minus_a_sq)
