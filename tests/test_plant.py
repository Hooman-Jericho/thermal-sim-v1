import numpy as np
import pytest

from src.plant import ActuatedThermalPlant, PlantConfig


@pytest.fixture
def cfg() -> PlantConfig:
    return PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=1.0)


def test_zero_input_holds_at_ambient(cfg):
    """u=0, d=0 from T=T_amb should stay at T_amb (already at equilibrium)."""
    plant = ActuatedThermalPlant(cfg, initial_temp=cfg.T_amb, seed=0)
    for _ in range(50):
        plant.step(u=0.0, d=0.0)
    assert plant.state.temperatures["T"] == pytest.approx(cfg.T_amb, abs=1e-9)


def test_converges_to_analytical_steady_state(cfg):
    """Constant u should drive T toward T_amb + u/k_loss (energy balance at equilibrium)."""
    u = 200.0
    expected_steady_state = cfg.T_amb + u / cfg.k_loss  # 65.0
    plant = ActuatedThermalPlant(cfg, initial_temp=cfg.T_amb, seed=0)
    for _ in range(2000):  # >> tau = mCp / k_loss = 100 s
        plant.step(u=u, d=0.0)
    assert plant.state.temperatures["T"] == pytest.approx(expected_steady_state, abs=1e-2)


def test_matches_forward_euler_by_hand(cfg):
    """One manual step should match the closed-form forward-Euler update."""
    plant = ActuatedThermalPlant(cfg, initial_temp=40.0, seed=0)
    u, d = 150.0, 10.0
    T0 = plant.state.temperatures["T"]
    expected_dT = (u - cfg.k_loss * (T0 - cfg.T_amb) - d) / cfg.mCp * cfg.dt
    plant.step(u=u, d=d)
    assert plant.state.temperatures["T"] == pytest.approx(T0 + expected_dT, abs=1e-9)


def test_disturbance_reproducible_given_seed(cfg):
    """Same seed -> identical disturbance sequence (private, seeded RNG)."""
    p1 = ActuatedThermalPlant(cfg, seed=123)
    p2 = ActuatedThermalPlant(cfg, seed=123)
    d1 = [p1.sample_disturbance(10.0) for _ in range(20)]
    d2 = [p2.sample_disturbance(10.0) for _ in range(20)]
    assert d1 == d2


def test_disturbance_differs_across_seeds(cfg):
    p1 = ActuatedThermalPlant(cfg, seed=1)
    p2 = ActuatedThermalPlant(cfg, seed=2)
    d1 = [p1.sample_disturbance(10.0) for _ in range(20)]
    d2 = [p2.sample_disturbance(10.0) for _ in range(20)]
    assert d1 != d2


def test_reset_returns_to_initial_condition(cfg):
    plant = ActuatedThermalPlant(cfg, initial_temp=33.0, seed=0)
    plant.step(u=500.0, d=0.0)
    assert plant.state.temperatures["T"] != pytest.approx(33.0)
    plant.reset()
    # NOTE: reset() alone (without reseed()) restores initial_temp but
    # the class stores initial_temp as an attribute, so re-calling reset()
    # directly must reproduce it exactly.
    fresh_state = plant.reset()
    assert fresh_state.temperatures["T"] == pytest.approx(33.0)


def test_max_physical_rate_matches_energy_balance(cfg):
    u_max = 300.0
    rate = plant_rate = ActuatedThermalPlant(cfg, seed=0).max_physical_rate(u_max)
    assert rate == pytest.approx(u_max / cfg.mCp)
