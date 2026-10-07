"""Input-validation tests for src/ (parameters, plant inputs, GD models)."""

import math

import numpy as np
import pytest

from src.ml_scratch.models import LinearRegressionScratch, LogisticRegressionScratch
from src.plant import ActuatedThermalPlant, PlantConfig
from src.systems import HeatExchangerSystem, NewtonianCoolingSystem

NAN = float("nan")
INF = float("inf")


def _cfg(**kw: float) -> PlantConfig:
    base = {"mCp": 500.0, "k_loss": 5.0, "T_amb": 25.0, "dt": 1.0}
    base.update(kw)
    return PlantConfig(**base)


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    return X, X @ np.array([1.0, -2.0, 0.5]) + 0.3


# --- passive systems ------------------------------------------------------
@pytest.mark.parametrize("k", [0.0, -0.1, NAN])
def test_newtonian_rejects_bad_k(k: float) -> None:
    with pytest.raises(ValueError, match="k must be > 0"):
        NewtonianCoolingSystem(k=k)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"k_exchange": -0.01}, "k_exchange"),
        ({"k_loss": -0.01}, "k_loss"),
        ({"k_exchange": NAN}, "k_exchange"),
    ],
)
def test_heat_exchanger_rejects_negative_coefficients(
    kw: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        HeatExchangerSystem(**kw)  # type: ignore[arg-type]


def test_heat_exchanger_allows_zero_coefficients() -> None:
    HeatExchangerSystem(k_exchange=0.0, k_loss=0.0).step()


def test_passive_systems_euler_guard() -> None:
    with pytest.raises(ValueError, match="unstable"):
        NewtonianCoolingSystem(k=2.0, dt=1.0)
    with pytest.raises(ValueError, match="unstable"):
        HeatExchangerSystem(k_exchange=0.5, k_loss=0.1, dt=2.0)
    NewtonianCoolingSystem(k=1.9, dt=1.0)  # just inside the stable region


# --- plant config / plant -------------------------------------------------
@pytest.mark.parametrize(
    "kw",
    [
        {"mCp": 0.0},
        {"mCp": -1.0},
        {"k_loss": -0.5},
        {"dt": 0.0},
        {"dt": -1.0},
        {"mCp": NAN},
        {"k_loss": INF},
        {"T_amb": NAN},
        {"dt": INF},
    ],
)
def test_plant_config_rejects_bad_values(kw: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _cfg(**kw)


def test_plant_config_euler_instability_guard() -> None:
    with pytest.raises(ValueError, match="forward Euler is unstable"):
        _cfg(mCp=1.0, k_loss=2.0, dt=1.0)  # dt * k_loss / mCp == 2
    _cfg(mCp=1.0, k_loss=1.99, dt=1.0)


def test_plant_rechecks_mutated_config() -> None:
    cfg = _cfg()
    cfg.dt = -1.0
    with pytest.raises(ValueError, match="dt"):
        ActuatedThermalPlant(cfg)


@pytest.mark.parametrize("name", ["u", "d"])
@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_plant_step_rejects_non_finite(name: str, bad: float) -> None:
    plant = ActuatedThermalPlant(_cfg())
    with pytest.raises(ValueError, match=name):
        plant.step(**{name: bad})
    assert plant.state.t == 0.0  # failed step must not advance the state


def test_plant_step_rejects_non_numeric() -> None:
    plant = ActuatedThermalPlant(_cfg())
    with pytest.raises(TypeError, match="u"):
        plant.step(u="hot")  # type: ignore[arg-type]


def test_valid_plant_still_works() -> None:
    plant = ActuatedThermalPlant(_cfg(), seed=1)
    for _ in range(5):
        plant.step(u=100.0, d=plant.sample_disturbance(5.0))
    assert math.isfinite(plant.state.temperatures["T"])
    assert plant.state.temperatures["T"] > 25.0


def test_repo_experiment_config_values_are_valid() -> None:
    cfg = PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=1.0)
    assert cfg.dt * cfg.k_loss / cfg.mCp < 2.0


# --- GD models ------------------------------------------------------------
@pytest.mark.parametrize(
    "model_cls", [LinearRegressionScratch, LogisticRegressionScratch]
)
@pytest.mark.parametrize("bad", [NAN, INF])
def test_fit_rejects_non_finite_inputs(model_cls: type, bad: float) -> None:
    X, _ = _data()
    y = (X[:, 0] > 0).astype(float)
    X_bad = X.copy()
    X_bad[3, 1] = bad
    with pytest.raises(ValueError, match="X contains"):
        model_cls().fit(X_bad, y)
    y_bad = y.copy()
    y_bad[0] = bad
    with pytest.raises(ValueError, match="y contains"):
        model_cls().fit(X, y_bad)


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_unfitted_predict_raises_runtime_error(method: str) -> None:
    X, _ = _data()
    with pytest.raises(RuntimeError, match="call fit\\(\\) first"):
        getattr(LogisticRegressionScratch(), method)(X)
    with pytest.raises(RuntimeError, match="call fit\\(\\) first"):
        LinearRegressionScratch().predict(X)


def test_predict_feature_count_mismatch() -> None:
    X, y = _data()
    lin = LinearRegressionScratch(epochs=3).fit(X, y)
    with pytest.raises(ValueError, match="2 features.*3 features"):
        lin.predict(X[:, :2])
    log = LogisticRegressionScratch(epochs=3).fit(X, (y > 0).astype(float))
    with pytest.raises(ValueError, match="features"):
        log.predict_proba(np.hstack([X, X]))
    with pytest.raises(ValueError, match="2-D"):
        lin.predict(X[0])


def test_divergence_raises_floating_point_error() -> None:
    X, y = _data()
    with pytest.raises(FloatingPointError, match="lower lr"):
        LinearRegressionScratch(lr=50.0, epochs=200, momentum=0.0).fit(X * 1e3, y)


def test_valid_fit_unchanged_and_deterministic() -> None:
    X, y = _data()
    a = LinearRegressionScratch(lr=0.05, epochs=200, random_state=3).fit(X, y)
    b = LinearRegressionScratch(lr=0.05, epochs=200, random_state=3).fit(X, y)
    assert np.array_equal(a.weights, b.weights)
    assert np.allclose(a.predict(X), y, atol=1e-2)
    assert len(a.loss_history) == 200
