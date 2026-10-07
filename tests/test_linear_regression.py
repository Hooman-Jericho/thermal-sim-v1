"""
test_linear_regression.py
--------------------------
Two kinds of checks, throughout this suite:

- EXACT       -> full-batch GD, no momentum (deterministic) => weights must
                 match sklearn to ~1e-6.
- STATISTICAL -> mini-batch/SGD + momentum never converges to the exact
                 minimiser (gradient-noise floor), so we compare quality
                 metrics on a held-out test set instead of raw weights.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from src.ml_scratch.models import LinearRegressionScratch


def test_full_batch_weights_match_sklearn_exactly(regression_data):
    d = regression_data
    lin = LinearRegressionScratch(
        lr=0.1, epochs=500, batch_size=None, momentum=0.0
    ).fit(d.X_tr, d.y_tr)
    assert np.allclose(lin.weights, d.sk.coef_, atol=1e-6)


def test_full_batch_bias_matches_sklearn_exactly(regression_data):
    d = regression_data
    lin = LinearRegressionScratch(
        lr=0.1, epochs=500, batch_size=None, momentum=0.0
    ).fit(d.X_tr, d.y_tr)
    assert abs(lin.bias - d.sk.intercept_) < 1e-6


def test_full_batch_loss_is_monotonically_non_increasing(regression_data):
    d = regression_data
    lin = LinearRegressionScratch(
        lr=0.1, epochs=500, batch_size=None, momentum=0.0
    ).fit(d.X_tr, d.y_tr)
    history = lin.loss_history
    assert all(a >= b - 1e-12 for a, b in zip(history, history[1:], strict=False))


def test_minibatch_momentum_test_r2_close_to_sklearn(regression_data):
    d = regression_data
    mb = LinearRegressionScratch(lr=0.005, epochs=200, batch_size=32, momentum=0.9).fit(
        d.X_tr, d.y_tr
    )
    r2_scratch = r2_score(d.y_te, mb.predict(d.X_te))
    r2_sklearn = r2_score(d.y_te, d.sk.predict(d.X_te))
    assert abs(r2_scratch - r2_sklearn) < 0.01


def test_minibatch_momentum_test_mse_within_3pct_of_sklearn(regression_data):
    d = regression_data
    mb = LinearRegressionScratch(lr=0.005, epochs=200, batch_size=32, momentum=0.9).fit(
        d.X_tr, d.y_tr
    )
    mse_scratch = mean_squared_error(d.y_te, mb.predict(d.X_te))
    mse_sklearn = mean_squared_error(d.y_te, d.sk.predict(d.X_te))
    assert mse_scratch <= 1.03 * mse_sklearn


def test_pure_sgd_test_r2_close_to_sklearn(regression_data):
    d = regression_data
    sgd = LinearRegressionScratch(lr=0.001, epochs=100, batch_size=1, momentum=0.0).fit(
        d.X_tr, d.y_tr
    )
    r2_scratch = r2_score(d.y_te, sgd.predict(d.X_te))
    r2_sklearn = r2_score(d.y_te, d.sk.predict(d.X_te))
    assert abs(r2_scratch - r2_sklearn) < 0.01
