"""
test_engineering.py
----------------------
Correctness properties that have nothing to do with the math being
right, but matter just as much for code you'd reuse elsewhere in the
thesis (e.g. as a PyTorch-optimizer baseline comparison):
reproducibility, no hidden global state, refit() actually resets, and
constructor arguments are validated instead of failing confusingly
deep inside fit().
"""

import numpy as np
import pytest

from src.ml_scratch.models import LinearRegressionScratch, LogisticRegressionScratch


def test_same_seed_gives_bit_identical_weights(regression_data):
    d = regression_data
    m1 = LinearRegressionScratch(lr=0.01, epochs=50, batch_size=32, random_state=7).fit(
        d.X_tr, d.y_tr
    )
    m2 = LinearRegressionScratch(lr=0.01, epochs=50, batch_size=32, random_state=7).fit(
        d.X_tr, d.y_tr
    )
    assert np.array_equal(m1.weights, m2.weights)
    assert m1.bias == m2.bias


def test_different_seeds_give_different_weights(regression_data):
    d = regression_data
    m1 = LinearRegressionScratch(lr=0.01, epochs=50, batch_size=32, random_state=1).fit(
        d.X_tr, d.y_tr
    )
    m2 = LinearRegressionScratch(lr=0.01, epochs=50, batch_size=32, random_state=2).fit(
        d.X_tr, d.y_tr
    )
    assert not np.array_equal(m1.weights, m2.weights)


def test_fit_does_not_touch_global_numpy_random_state(regression_data):
    d = regression_data
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    LinearRegressionScratch(lr=0.01, epochs=50, batch_size=32, random_state=7).fit(
        d.X_tr, d.y_tr
    )
    after = np.random.get_state()[1]
    assert np.array_equal(before, after), (
        "fit() must use its own RNG, not np.random's global state"
    )


def test_refit_resets_loss_history_and_iteration_count(regression_data):
    d = regression_data
    m = LinearRegressionScratch(lr=0.01, epochs=30, batch_size=32).fit(d.X_tr, d.y_tr)
    first_len = len(m.loss_history)
    m.fit(d.X_tr, d.y_tr)  # refit on the same object
    assert len(m.loss_history) == first_len  # not doubled / appended


def test_refit_with_fewer_epochs_shrinks_history(regression_data):
    d = regression_data
    m = LinearRegressionScratch(lr=0.01, epochs=30, batch_size=32)
    m.fit(d.X_tr, d.y_tr)
    m.epochs = 5
    m.fit(d.X_tr, d.y_tr)
    assert len(m.loss_history) == 5


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        dict(lr=0.0),
        dict(lr=-1.0),
        dict(epochs=0),
        dict(momentum=1.0),
        dict(momentum=-0.1),
        dict(l1_ratio=-0.1),
        dict(l2_ratio=-0.1),
        dict(batch_size=0),
        dict(lr_decay=-1.0),
        dict(tol=-1e-6),
        dict(n_iter_no_change=0),
    ],
)
def test_constructor_rejects_invalid_hyperparameters(bad_kwargs):
    with pytest.raises(ValueError):
        LinearRegressionScratch(**bad_kwargs)


def test_fit_rejects_mismatched_X_y_lengths(regression_data):
    d = regression_data
    with pytest.raises(ValueError):
        LinearRegressionScratch(epochs=5).fit(d.X_tr, d.y_tr[:-1])


def test_fit_rejects_1d_X(regression_data):
    d = regression_data
    with pytest.raises(ValueError):
        LinearRegressionScratch(epochs=5).fit(d.X_tr[:, 0], d.y_tr)


def test_logistic_rejects_non_binary_targets_before_any_fitting_work(
    classification_data,
):
    """Validation should happen before the (potentially long) training loop starts."""
    d = classification_data
    y_bad = d.y_tr.astype(float).copy()
    y_bad[:5] = 3.0
    with pytest.raises(ValueError):
        LogisticRegressionScratch(epochs=100000).fit(d.X_tr, y_bad)
