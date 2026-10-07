"""
test_regularization_logistic.py
----------------------------------
sklearn's LogisticRegression parameterizes regularization strength as
C = 1 / (2 * n * l2_ratio) for an L2 penalty on the MEAN cross-entropy
(the "2" comes from sklearn scaling its penalty by 1/2 internally).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.ml_scratch.models import LogisticRegressionScratch
from tests._typing import fitted_weights


def test_l2_weights_match_sklearn_under_correct_C_mapping(classification_data):
    d = classification_data
    l2_ratio = 0.3
    C = 1.0 / (2.0 * d.n * l2_ratio)
    ours = LogisticRegressionScratch(
        lr=0.5, epochs=3000, batch_size=None, momentum=0.0, l2_ratio=l2_ratio
    ).fit(d.X_tr, d.y_tr)
    sk_l2 = LogisticRegression(l1_ratio=0, C=C, max_iter=10000, tol=1e-12).fit(
        d.X_tr, d.y_tr
    )
    assert np.allclose(fitted_weights(ours), sk_l2.coef_.ravel(), atol=0.05)


def test_l2_shrinks_weight_norm_as_ratio_increases(classification_data):
    # NOTE on l2_ratio=0.5, not e.g. 2.0: plain gradient descent on an L2
    # penalty is only a CONTRACTION when lr * 2 * l2_ratio < 1 (the L2
    # term's own update is w *= 1 - 2*lr*l2_ratio). At lr=0.5, that bound
    # is l2_ratio < 1.0 -- l2_ratio=2.0 sits exactly on/past the marginal
    # stability boundary (factor = -1) and, combined with the data
    # gradient over 1000 epochs, diverges instead of shrinking. This is
    # standard GD step-size/strong-convexity behavior, not a modeling
    # bug -- verified by test_l2_diverges_when_step_size_too_large below.
    d = classification_data
    weak = LogisticRegressionScratch(
        lr=0.5, epochs=1000, batch_size=None, momentum=0.0, l2_ratio=0.01
    ).fit(d.X_tr, d.y_tr)
    strong = LogisticRegressionScratch(
        lr=0.5, epochs=1000, batch_size=None, momentum=0.0, l2_ratio=0.5
    ).fit(d.X_tr, d.y_tr)
    assert np.linalg.norm(fitted_weights(strong)) < np.linalg.norm(fitted_weights(weak))


def test_l2_diverges_when_step_size_exceeds_stability_bound(classification_data):
    """Documents the instability from the note above, so it's a known,
    tested property instead of a silent trap for the next person who
    picks a large lr and l2_ratio together."""
    d = classification_data
    lr, l2_ratio = 0.5, 2.0  # lr * 2 * l2_ratio = 2.0 >= 1 -> marginal/unstable
    unstable = LogisticRegressionScratch(
        lr=lr, epochs=1000, batch_size=None, momentum=0.0, l2_ratio=l2_ratio
    ).fit(d.X_tr, d.y_tr)
    assert (
        np.linalg.norm(fitted_weights(unstable)) > 50.0
    )  # confirms it did NOT shrink toward 0


def test_l1_produces_exact_zero_coefficients(classification_data):
    d = classification_data
    ours = LogisticRegressionScratch(
        lr=0.5, epochs=3000, batch_size=None, momentum=0.0, l1_ratio=0.3
    ).fit(d.X_tr, d.y_tr)
    assert np.sum(ours.weights == 0.0) >= 1


def test_l1_sparsity_direction_matches_sklearn_l1(classification_data):
    """Not an exact-weight match (different solvers), just: both agree on WHICH
    features survive."""
    d = classification_data
    l1_ratio = 0.3
    C = 1.0 / (
        d.n * l1_ratio
    )  # sklearn's l1_ratio-to-C mapping (no factor of 2 for L1's own convention)
    ours = LogisticRegressionScratch(
        lr=0.5, epochs=3000, batch_size=None, momentum=0.0, l1_ratio=l1_ratio
    ).fit(d.X_tr, d.y_tr)
    sk_l1 = LogisticRegression(
        l1_ratio=1, solver="liblinear", C=C, max_iter=10000, tol=1e-12
    ).fit(d.X_tr, d.y_tr)
    ours_survivors = set(np.flatnonzero(ours.weights != 0.0))
    sk_survivors = set(np.flatnonzero(sk_l1.coef_.ravel() != 0.0))
    overlap = len(ours_survivors & sk_survivors)
    assert overlap >= min(len(ours_survivors), len(sk_survivors)) - 1
