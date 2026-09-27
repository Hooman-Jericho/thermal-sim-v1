"""
test_logistic_regression.py
------------------------------
Same EXACT vs STATISTICAL split as test_linear_regression.py.
Unregularized reference uses sklearn's LogisticRegression(C=np.inf)
(sklearn's max_iter-based solver, not gradient descent -- so "exact"
here means "both converge to the same unregularized MLE", verified
to a looser tolerance than the linear-regression case since sklearn's
solver (LBFGS) and ours (GD) approach that MLE differently).
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from src.ml_scratch.models import LogisticRegressionScratch


def test_full_batch_weights_close_to_sklearn_mle(classification_data):
    d = classification_data
    lg = LogisticRegressionScratch(lr=0.5, epochs=3000, batch_size=None, momentum=0.0).fit(d.X_tr, d.y_tr)
    assert np.allclose(lg.weights, d.sk.coef_.ravel(), atol=0.05)
    assert abs(lg.bias - d.sk.intercept_[0]) < 0.05


def test_full_batch_test_accuracy_matches_sklearn(classification_data):
    d = classification_data
    lg = LogisticRegressionScratch(lr=0.5, epochs=3000, batch_size=None, momentum=0.0).fit(d.X_tr, d.y_tr)
    acc_scratch = accuracy_score(d.y_te, lg.predict(d.X_te))
    acc_sklearn = accuracy_score(d.y_te, d.sk.predict(d.X_te))
    assert abs(acc_scratch - acc_sklearn) <= 0.02


def test_full_batch_log_loss_matches_sklearn(classification_data):
    d = classification_data
    lg = LogisticRegressionScratch(lr=0.5, epochs=3000, batch_size=None, momentum=0.0).fit(d.X_tr, d.y_tr)
    ll_scratch = log_loss(d.y_te, lg.predict_proba(d.X_te))
    ll_sklearn = log_loss(d.y_te, d.sk.predict_proba(d.X_te)[:, 1])
    assert ll_scratch <= ll_sklearn + 0.02


def test_minibatch_momentum_accuracy_close_to_sklearn(classification_data):
    d = classification_data
    mb = LogisticRegressionScratch(lr=0.05, epochs=300, batch_size=32, momentum=0.9).fit(d.X_tr, d.y_tr)
    acc_scratch = accuracy_score(d.y_te, mb.predict(d.X_te))
    acc_sklearn = accuracy_score(d.y_te, d.sk.predict(d.X_te))
    assert abs(acc_scratch - acc_sklearn) <= 0.03


def test_predict_proba_is_between_zero_and_one(classification_data):
    d = classification_data
    lg = LogisticRegressionScratch(lr=0.5, epochs=200, batch_size=None, momentum=0.0).fit(d.X_tr, d.y_tr)
    p = lg.predict_proba(d.X_te)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_predict_threshold_is_configurable(classification_data):
    d = classification_data
    lg = LogisticRegressionScratch(lr=0.5, epochs=500, batch_size=None, momentum=0.0).fit(d.X_tr, d.y_tr)
    strict = lg.predict(d.X_te, threshold=0.9)
    lenient = lg.predict(d.X_te, threshold=0.1)
    assert lenient.sum() >= strict.sum()


def test_sigmoid_is_numerically_stable_for_extreme_inputs():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    p = LogisticRegressionScratch._sigmoid(z)
    assert np.all(np.isfinite(p))
    assert p[0] == 0.0 and p[-1] == 1.0
    assert p[2] == 0.5


def test_rejects_non_binary_targets(classification_data):
    d = classification_data
    y_bad = d.y_tr.copy().astype(float)
    y_bad[0] = 2.0
    import pytest
    with pytest.raises(ValueError):
        LogisticRegressionScratch(epochs=5).fit(d.X_tr, y_bad)
