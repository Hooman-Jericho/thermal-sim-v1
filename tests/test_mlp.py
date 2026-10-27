"""Tests for src/nn/mlp.py (Days 37 and 43)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from experiments.baseline_dynamics_learning.estimators import make_model
from src.nn.mlp import MLP, TorchMLPRegressor, TrainConfig, train_mlp
from src.plant import PlantConfig


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return X, 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + 3.0


# --- the module ------------------------------------------------------------
def test_forward_shape():
    import torch

    out = MLP(4, (8, 8))(torch.zeros(5, 4))
    assert out.shape == (5,)


@pytest.mark.parametrize(
    "kw",
    [
        {"n_in": 0},
        {"n_in": 2, "hidden": ()},
        {"n_in": 2, "hidden": (0,)},
        {"n_in": 2, "activation": "nope"},
        {"n_in": 2, "dropout": 1.0},
    ],
)
def test_bad_architecture_is_rejected(kw):
    with pytest.raises(ValueError):
        MLP(**kw)


# --- the training loop -------------------------------------------------------
def test_learns_a_linear_function_in_original_units():
    X, y = _toy()
    model, hist = train_mlp(X[:500], y[:500], TrainConfig(max_epochs=80, seed=1))
    err = np.mean((model.predict(X[500:]) - y[500:]) ** 2)
    assert err < 0.01 * y.var()
    assert hist.train_loss[-1] < hist.train_loss[0]


def test_same_seed_is_bit_identical_and_other_seed_differs():
    X, y = _toy()
    cfg = TrainConfig(max_epochs=5, seed=3)
    p1 = train_mlp(X, y, cfg)[0].predict(X)
    p2 = train_mlp(X, y, cfg)[0].predict(X)
    p3 = train_mlp(X, y, TrainConfig(max_epochs=5, seed=4))[0].predict(X)
    np.testing.assert_array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


def test_early_stopping_restores_best_weights():
    X, y = _toy(800)
    y_noisy = y + np.random.default_rng(1).normal(scale=3.0, size=len(y))
    cfg = TrainConfig(hidden=(64, 64), lr=5e-3, max_epochs=200, patience=5, seed=0)
    model, hist = train_mlp(X[:200], y_noisy[:200], cfg, X[200:500], y_noisy[200:500])
    assert hist.stopped_early and len(hist.train_loss) < 200
    # restored model reaches the recorded best validation loss
    z = (X[200:500] - model.x_mean) / model.x_std
    import torch

    model.module.eval()
    with torch.no_grad():
        pred = model.module(torch.as_tensor(z, dtype=torch.float32)).numpy()
    yv = (y_noisy[200:500] - model.y_mean) / model.y_std
    assert np.mean((pred - yv) ** 2) == pytest.approx(min(hist.val_loss), rel=1e-4)


def test_on_epoch_callback_is_called_every_epoch_and_may_abort():
    X, y = _toy(200)
    seen = []
    train_mlp(X, y, TrainConfig(max_epochs=4), on_epoch=lambda e, t, v: seen.append(e))
    assert seen == [1, 2, 3, 4]

    def stop(epoch, tr, va):
        if epoch == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train_mlp(X, y, TrainConfig(max_epochs=9), on_epoch=stop)


def test_nonfinite_data_and_shape_errors():
    X, y = _toy(50)
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        train_mlp(bad, y)
    with pytest.raises(ValueError, match="rows"):
        train_mlp(X, y[:-1])
    with pytest.raises(ValueError, match="2-D"):
        train_mlp(X[:, 0], y)
    with pytest.raises(ValueError, match="validation"):
        train_mlp(X, y, TrainConfig(patience=3))
    with pytest.raises(ValueError, match="both"):
        train_mlp(X, y, X_val=X)


def test_wrong_feature_count_at_predict():
    X, y = _toy(100)
    model, _ = train_mlp(X, y, TrainConfig(max_epochs=2))
    with pytest.raises(ValueError, match="features"):
        model.predict(X[:, :2])


def test_divergence_raises_clear_error():
    X, y = _toy(100)
    with pytest.raises(FloatingPointError):
        train_mlp(X, y * 1e30, TrainConfig(lr=1e30, max_epochs=3))


@pytest.mark.parametrize("kw", [{"lr": 0.0}, {"batch_size": 0}, {"max_epochs": 0}])
def test_bad_train_config(kw):
    X, y = _toy(50)
    with pytest.raises(ValueError):
        train_mlp(X, y, TrainConfig(**kw))


def test_constant_feature_does_not_produce_nan():
    X, y = _toy(100)
    X[:, 1] = 5.0
    pred = train_mlp(X, y, TrainConfig(max_epochs=3))[0].predict(X)
    assert np.all(np.isfinite(pred))


# --- the sklearn wrapper -----------------------------------------------------
def test_wrapper_clone_and_unfitted_error():
    est = TorchMLPRegressor(hidden=(8,), max_epochs=2)
    assert clone(est).get_params() == est.get_params()
    with pytest.raises(RuntimeError, match="fit"):
        est.predict(np.zeros((1, 3)))


def test_make_model_mlp_physics_residual_matches_physics_on_noiseless_data():
    cfg = PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=1.0)
    rng = np.random.default_rng(0)
    T, u = rng.uniform(20, 60, 300), rng.uniform(0, 300, 300)
    y = T + cfg.dt * (u - cfg.k_loss * (T - cfg.T_amb)) / cfg.mCp
    X = pd.DataFrame({"T_current": T, "u": u})
    model = make_model("mlp", "physics_residual", cfg, {"mlp": {"max_epochs": 100}}, 0)
    model.fit(X, y)
    # The residual is exactly zero, so the MLP only has to learn "add nothing".
    # Measured: max error 0.147 K after 5 epochs, 0.012 K after 50, so 100 epochs
    # must stay well under 0.05 K (a trained network is never exactly 0).
    assert np.max(np.abs(model.predict(X) - y)) < 0.05
