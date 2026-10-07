"""
test_lr_schedule_and_stopping.py
----------------------------------
- lr_decay: proves mini-batch GD actually converges to the EXACT
  minimiser (weights, not just a quality metric) once the learning
  rate decays -- a constant lr stalls on a gradient-noise floor.
- tol: early stopping behavior and its interaction with accuracy.
"""

import numpy as np
from sklearn.metrics import r2_score

from src.ml_scratch.models import LinearRegressionScratch
from tests._typing import fitted_weights


def test_constant_lr_stalls_on_noise_floor(regression_data):
    """Across several seeds, constant lr should NOT get close to sklearn's weights."""
    d = regression_data
    errs = []
    for seed in range(4):
        c = LinearRegressionScratch(
            lr=0.02, epochs=1000, batch_size=32, momentum=0.9, random_state=seed
        ).fit(d.X_tr, d.y_tr)
        errs.append(np.max(np.abs(c.weights - d.sk.coef_)))
    assert min(errs) > 0.3, f"max|dw| per seed={np.round(errs, 3)}"


def test_lr_decay_converges_close_to_minimiser(regression_data):
    d = regression_data
    errs = []
    for seed in range(4):
        m = LinearRegressionScratch(
            lr=0.02,
            epochs=1000,
            batch_size=32,
            momentum=0.9,
            lr_decay=1.0,
            random_state=seed,
        ).fit(d.X_tr, d.y_tr)
        errs.append(np.max(np.abs(m.weights - d.sk.coef_)))
    assert max(errs) < 0.03, f"max|dw| per seed={np.round(errs, 4)}"


def test_lr_decay_is_at_least_30x_closer_than_constant_lr(regression_data):
    d = regression_data
    errs_const, errs_decay = [], []
    for seed in range(4):
        c = LinearRegressionScratch(
            lr=0.02, epochs=1000, batch_size=32, momentum=0.9, random_state=seed
        ).fit(d.X_tr, d.y_tr)
        m = LinearRegressionScratch(
            lr=0.02,
            epochs=1000,
            batch_size=32,
            momentum=0.9,
            lr_decay=1.0,
            random_state=seed,
        ).fit(d.X_tr, d.y_tr)
        errs_const.append(np.max(np.abs(c.weights - d.sk.coef_)))
        errs_decay.append(np.max(np.abs(m.weights - d.sk.coef_)))
    ratio = np.mean(errs_const) / np.mean(errs_decay)
    assert ratio >= 30, f"ratio={ratio:.0f}x"


def test_pure_sgd_with_lr_decay_matches_sklearn_closely(regression_data):
    d = regression_data
    sgd_d = LinearRegressionScratch(
        lr=0.01, epochs=300, batch_size=1, momentum=0.0, lr_decay=1.0
    ).fit(d.X_tr, d.y_tr)
    assert np.max(np.abs(sgd_d.weights - d.sk.coef_)) < 0.02


def test_early_stopping_triggers_and_flags_converged(regression_data):
    d = regression_data
    es = LinearRegressionScratch(
        lr=0.02, epochs=2000, batch_size=32, momentum=0.9, lr_decay=1.0, tol=1e-6
    ).fit(d.X_tr, d.y_tr)
    assert es.converged_ and es.n_iter_ < 2000


def test_loss_history_length_matches_n_iter(regression_data):
    d = regression_data
    es = LinearRegressionScratch(
        lr=0.02, epochs=2000, batch_size=32, momentum=0.9, lr_decay=1.0, tol=1e-6
    ).fit(d.X_tr, d.y_tr)
    assert len(es.loss_history) == es.n_iter_


def test_early_stopped_model_still_accurate(regression_data):
    d = regression_data
    es = LinearRegressionScratch(
        lr=0.02, epochs=2000, batch_size=32, momentum=0.9, lr_decay=1.0, tol=1e-6
    ).fit(d.X_tr, d.y_tr)
    r2_scratch = r2_score(d.y_te, es.predict(d.X_te))
    r2_sklearn = r2_score(d.y_te, d.sk.predict(d.X_te))
    assert abs(r2_scratch - r2_sklearn) < 0.005


def test_tol_none_runs_every_epoch(regression_data):
    d = regression_data
    no_es = LinearRegressionScratch(
        lr=0.02, epochs=50, batch_size=32, momentum=0.9
    ).fit(d.X_tr, d.y_tr)
    assert no_es.n_iter_ == 50 and not no_es.converged_


def test_full_batch_with_tight_tol_stops_at_machine_precision_and_stays_exact(
    regression_data,
):
    d = regression_data
    tight = LinearRegressionScratch(
        lr=0.1, epochs=5000, batch_size=None, momentum=0.0, tol=1e-14
    ).fit(d.X_tr, d.y_tr)
    assert tight.converged_
    assert np.allclose(fitted_weights(tight), d.sk.coef_, atol=1e-6)
