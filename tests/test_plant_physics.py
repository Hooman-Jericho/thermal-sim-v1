"""Physics validation of ActuatedThermalPlant against closed-form solutions.

Three different claims are tested, because they fail for different reasons:

1. The *code* implements forward Euler exactly (matches ``euler_response`` to
   machine precision).
2. The *method* converges to the true ODE solution, first order in ``dt``.
3. The *noise model* has the variance theory predicts.
"""

import math

import numpy as np
import pytest

from src.analytical import (
    continuous_response,
    euler_response,
    euler_stationary_variance,
    steady_state,
    time_constant,
)
from src.plant import ActuatedThermalPlant, PlantConfig

MCP, K_LOSS, T_AMB = 500.0, 5.0, 25.0


def _cfg(dt: float = 1.0, k_loss: float = K_LOSS) -> PlantConfig:
    return PlantConfig(mCp=MCP, k_loss=k_loss, T_amb=T_AMB, dt=dt)


def _simulate(cfg, T0, u, d, n_steps):
    plant = ActuatedThermalPlant(cfg, initial_temp=T0, seed=0)
    out = [T0]
    for _ in range(n_steps):
        out.append(plant.step(u=u, d=d).temperatures["T"])
    return np.array(out)


# --- helper formulas ------------------------------------------------------
def test_time_constant_and_steady_state():
    cfg = _cfg()
    assert time_constant(cfg) == pytest.approx(100.0)
    assert steady_state(cfg, u=200.0, d=10.0) == pytest.approx(63.0)


def test_lossless_plant_has_no_steady_state():
    cfg = _cfg(k_loss=0.0)
    assert math.isinf(time_constant(cfg))
    with pytest.raises(ValueError):
        steady_state(cfg, u=10.0)
    with pytest.raises(ValueError):
        euler_stationary_variance(cfg, 1.0)


# --- 1. the code is exactly forward Euler ----------------------------------
@pytest.mark.parametrize("dt", [2.0, 1.0, 0.25])
@pytest.mark.parametrize("u,d,T0", [(200.0, 10.0, 25.0), (0.0, 0.0, 80.0)])
def test_simulator_equals_closed_form_euler(dt, u, d, T0):
    cfg = _cfg(dt=dt)
    n = int(round(300.0 / dt))
    sim = _simulate(cfg, T0, u, d, n)
    exact_euler = euler_response(cfg, np.arange(n + 1), T0, u, d)
    np.testing.assert_allclose(sim, exact_euler, rtol=0, atol=1e-9)


# --- 2. the method converges to the true solution --------------------------
def _max_error_vs_ode(dt, u=200.0, d=10.0, T0=25.0, horizon=300.0):
    cfg = _cfg(dt=dt)
    n = int(round(horizon / dt))
    sim = _simulate(cfg, T0, u, d, n)
    truth = continuous_response(cfg, np.arange(n + 1) * dt, T0, u, d)
    return float(np.max(np.abs(sim - truth)))


def test_transient_is_close_to_true_ode_solution():
    # Step response from 25 C towards 63 C: error well under 0.1 C at dt = 1 s.
    assert _max_error_vs_ode(1.0) < 0.1


def test_euler_error_is_first_order_in_dt():
    errs = [_max_error_vs_ode(dt) for dt in (2.0, 1.0, 0.5, 0.25)]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    # Halving dt must halve the error (ratio -> 2 for a first-order method).
    for r in ratios:
        assert r == pytest.approx(2.0, abs=0.05)


def test_response_reaches_steady_state_with_disturbance():
    cfg = _cfg()
    sim = _simulate(cfg, 25.0, u=200.0, d=10.0, n_steps=2000)
    assert sim[-1] == pytest.approx(steady_state(cfg, 200.0, 10.0), abs=1e-6)


def test_time_constant_is_63_percent_rise_time():
    cfg = _cfg(dt=0.1)
    T0, u = 25.0, 200.0
    T_ss = steady_state(cfg, u)
    n = int(round(time_constant(cfg) / cfg.dt))  # one tau
    sim = _simulate(cfg, T0, u, 0.0, n)
    expected = T0 + (1.0 - math.exp(-1.0)) * (T_ss - T0)
    assert sim[-1] == pytest.approx(expected, abs=0.1)


def test_lossless_plant_is_a_pure_integrator():
    cfg = _cfg(k_loss=0.0)
    sim = _simulate(cfg, 25.0, u=100.0, d=0.0, n_steps=500)
    # dT/dt = u / mCp = 0.2 K/s  ->  after 500 s: +100 K, exactly.
    assert sim[-1] == pytest.approx(25.0 + 100.0 * 500.0 / MCP * 1.0, abs=1e-9)
    np.testing.assert_allclose(
        sim, continuous_response(cfg, np.arange(501) * cfg.dt, 25.0, 100.0), atol=1e-9
    )


def test_energy_balance_in_joules():
    """Energy in = stored + lost, accumulated step by step (units check)."""
    cfg = _cfg()
    u, d = 150.0, 0.0
    sim = _simulate(cfg, 25.0, u, d, 400)
    heat_in = u * cfg.dt * 400
    stored = MCP * (sim[-1] - sim[0])
    lost = float(np.sum(K_LOSS * (sim[:-1] - T_AMB) * cfg.dt))
    assert heat_in == pytest.approx(stored + lost, rel=1e-9)


def test_cooling_from_hot_start_matches_newtonian_decay():
    cfg = _cfg(dt=0.1)
    sim = _simulate(cfg, 90.0, u=0.0, d=0.0, n_steps=1000)  # 100 s = 1 tau
    expected = T_AMB + (90.0 - T_AMB) * math.exp(-1.0)
    assert sim[-1] == pytest.approx(expected, abs=0.1)


# --- 3. the noise model has the predicted variance -------------------------
@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_stationary_variance_matches_ar1_theory(dt):
    cfg = _cfg(dt=dt)
    sigma = 5.0
    plant = ActuatedThermalPlant(cfg, initial_temp=T_AMB, seed=1)
    burn_in = int(10 * time_constant(cfg) / dt)
    samples = []
    for i in range(burn_in + 150_000):
        plant.step(u=0.0, d=plant.sample_disturbance(sigma))
        if i >= burn_in:
            samples.append(plant.state.temperatures["T"])
    theory = euler_stationary_variance(cfg, sigma)
    assert float(np.var(samples)) == pytest.approx(theory, rel=0.05)
    assert float(np.mean(samples)) == pytest.approx(T_AMB, abs=0.05)


def test_disturbance_std_scales_with_dt_not_physically_invariant():
    """Same sigma_d at a smaller dt gives a smaller T variance (~ dt)."""
    v1 = euler_stationary_variance(_cfg(dt=1.0), 5.0)
    v2 = euler_stationary_variance(_cfg(dt=0.5), 5.0)
    assert v2 / v1 == pytest.approx(0.5, rel=0.01)


# --- the safety band is actually reachable ----------------------------------
def test_full_actuation_steady_state_exceeds_t_max():
    """With the baseline config (u_max=300 W) the unsafe region is reachable."""
    cfg = _cfg()
    assert steady_state(cfg, u=300.0) == pytest.approx(85.0)
    assert steady_state(cfg, u=300.0) > 80.0  # T_max
    assert steady_state(cfg, u=0.0) == pytest.approx(T_AMB)
