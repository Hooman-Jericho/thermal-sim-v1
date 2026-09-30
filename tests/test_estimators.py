import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from experiments.baseline_dynamics_learning.estimators import (
    MODEL_NAMES, PersistenceBaseline, PhysicsBaseline, ResidualRegressor,
    make_model,
)
from src.plant import PlantConfig

MODELS_CFG = {"random_forest": {"n_estimators": 20, "max_depth": 5},
             "svm": {"C": 10.0, "epsilon": 0.01}}


@pytest.fixture
def plant_cfg() -> PlantConfig:
    return PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=1.0)


@pytest.fixture
def linear_frame(plant_cfg):
    """A tiny dataset that EXACTLY follows the plant's own physics (no noise),
    so a perfect model of any target_mode must reach ~0 MSE and agree with
    every other target_mode's prediction of T_next."""
    rng = np.random.default_rng(0)
    T = rng.uniform(20, 60, size=200)
    u = rng.uniform(0, 300, size=200)
    T_next = T + plant_cfg.dt * (u - plant_cfg.k_loss * (T - plant_cfg.T_amb)) / plant_cfg.mCp
    X = pd.DataFrame({"T_current": T, "u": u})
    return X, pd.Series(T_next, name="T_next")


# --- baselines ---------------------------------------------------------------

def test_persistence_baseline_returns_T_current():
    X = pd.DataFrame({"T_current": [30.0, 40.0], "u": [0.0, 0.0]})
    assert np.allclose(PersistenceBaseline()(X), [30.0, 40.0])


def test_physics_baseline_matches_energy_balance_by_hand(plant_cfg):
    X = pd.DataFrame({"T_current": [40.0], "u": [150.0]})
    expected = 40.0 + plant_cfg.dt * (150.0 - plant_cfg.k_loss * (40.0 - plant_cfg.T_amb)) / plant_cfg.mCp
    assert PhysicsBaseline(plant_cfg)(X)[0] == pytest.approx(expected)


# --- make_model: every (model, target_mode) fits and predicts T_next-shaped output

@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("target_mode", [
    "absolute_unscaled", "absolute", "delta", "physics_residual",
])
def test_every_model_target_combination_fits_and_predicts(model_name, target_mode, plant_cfg, linear_frame):
    X, y = linear_frame
    model = make_model(model_name, target_mode, plant_cfg, MODELS_CFG, seed=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.all(np.isfinite(pred))


def test_unknown_target_mode_is_rejected(plant_cfg):
    with pytest.raises(ValueError):
        make_model("linear", "not_a_real_mode", plant_cfg, MODELS_CFG, seed=0)


def test_unknown_model_name_is_rejected(plant_cfg):
    with pytest.raises(ValueError):
        make_model("not_a_real_model", "delta", plant_cfg, MODELS_CFG, seed=0)


def test_linear_recovers_noiseless_physics_regardless_of_target_mode(plant_cfg, linear_frame):
    """On noise-free, physics-consistent data, every target_mode is just a different
    parametrisation of the SAME function -- linear regression should reach ~0 MSE in all four."""
    X, y = linear_frame
    for target_mode in ["absolute_unscaled", "absolute", "delta", "physics_residual"]:
        model = make_model("linear", target_mode, plant_cfg, MODELS_CFG, seed=0)
        model.fit(X, y)
        mse = np.mean((model.predict(X) - y.to_numpy()) ** 2)
        assert mse < 1e-6, f"{target_mode}: mse={mse}"


# --- ResidualRegressor: the mechanism behind delta/physics_residual modes ----

def test_residual_regressor_learns_zero_when_baseline_is_exact():
    """If the baseline already equals y exactly, the wrapped estimator has nothing
    left to learn -- residual target is all zeros."""
    X = pd.DataFrame({"T_current": [10.0, 20.0, 30.0], "u": [0.0, 0.0, 0.0]})
    y = pd.Series([10.0, 20.0, 30.0])  # exactly T_current
    reg = ResidualRegressor(LinearRegression(), PersistenceBaseline())
    reg.fit(X, y)
    assert np.allclose(reg.predict(X), y.to_numpy(), atol=1e-8)


def test_residual_regressor_rejects_non_dataframe_input():
    reg = ResidualRegressor(LinearRegression(), PersistenceBaseline())
    with pytest.raises(TypeError):
        reg.fit(np.array([[1.0, 2.0]]), np.array([1.0]))


def test_residual_regressor_adds_baseline_back_at_predict_time():
    X = pd.DataFrame({"T_current": [10.0, 20.0], "u": [0.0, 0.0]})
    y = pd.Series([12.0, 25.0])  # baseline (T_current) + a learnable linear residual
    reg = ResidualRegressor(LinearRegression(), PersistenceBaseline()).fit(X, y)
    assert np.allclose(reg.predict(X), y.to_numpy(), atol=1e-6)
