"""
test_regularization_linear.py
-------------------------------
Ridge/Lasso are verified against sklearn under the EXACT parameter
mapping between this module's convention and sklearn's, not just
"roughly similar regularization":

    This module minimises:  mean_loss(w,b) + l2_ratio*||w||^2 + l1_ratio*||w||_1
    sklearn Ridge minimises: sum_loss(w,b)  + alpha*||w||^2
        => alpha = n_train * l2_ratio   (mean -> sum rescales the penalty by n)
    sklearn Lasso minimises: (1/(2n))*sum_loss(w,b) + alpha*||w||_1
        => alpha = l1_ratio / 2         (mean/(2) vs mean rescales by 1/2)

Getting this mapping wrong is the single most common bug in "from
scratch" regularization implementations -- these tests exist
specifically to catch it.
"""

import numpy as np
from sklearn.linear_model import Lasso, Ridge

from src.ml_scratch.models import LinearRegressionScratch


def test_ridge_weights_match_sklearn_under_correct_alpha_mapping(regression_data):
    d = regression_data
    l2_ratio = 0.5
    alpha = d.n * l2_ratio
    ours = LinearRegressionScratch(lr=0.1, epochs=1000, batch_size=None, momentum=0.0,
                                    l2_ratio=l2_ratio).fit(d.X_tr, d.y_tr)
    sk_ridge = Ridge(alpha=alpha).fit(d.X_tr, d.y_tr)
    assert np.allclose(ours.weights, sk_ridge.coef_, atol=1e-4)
    assert abs(ours.bias - sk_ridge.intercept_) < 1e-4


def test_ridge_bias_is_never_penalised(regression_data):
    """Increasing l2_ratio should shrink ||w|| but bias should stay close to y.mean()."""
    d = regression_data
    small = LinearRegressionScratch(lr=0.1, epochs=500, batch_size=None, momentum=0.0,
                                     l2_ratio=0.01).fit(d.X_tr, d.y_tr)
    large = LinearRegressionScratch(lr=0.1, epochs=500, batch_size=None, momentum=0.0,
                                     l2_ratio=5.0).fit(d.X_tr, d.y_tr)
    assert np.linalg.norm(large.weights) < np.linalg.norm(small.weights)
    assert abs(large.bias - small.bias) < 0.5


def test_lasso_weights_match_sklearn_under_correct_alpha_mapping(sparse_regression_data):
    X, y = sparse_regression_data
    l1_ratio = 0.1
    alpha = l1_ratio / 2.0
    ours = LinearRegressionScratch(lr=0.05, epochs=3000, batch_size=None, momentum=0.0,
                                    l1_ratio=l1_ratio).fit(X, y)
    sk_lasso = Lasso(alpha=alpha, max_iter=100000, tol=1e-10).fit(X, y)
    assert np.allclose(ours.weights, sk_lasso.coef_, atol=1e-2)


def test_lasso_produces_exact_zero_coefficients(sparse_regression_data):
    """Proximal soft-thresholding (not plain sub-gradient) should zero out weak features exactly."""
    X, y = sparse_regression_data  # only 3 of 10 features are informative
    ours = LinearRegressionScratch(lr=0.05, epochs=3000, batch_size=None, momentum=0.0,
                                    l1_ratio=0.5).fit(X, y)
    n_exact_zero = np.sum(ours.weights == 0.0)
    assert n_exact_zero >= 3, f"only {n_exact_zero} exact zeros out of {len(ours.weights)} weights"


def test_stronger_l1_yields_more_sparsity(sparse_regression_data):
    X, y = sparse_regression_data
    weak = LinearRegressionScratch(lr=0.05, epochs=3000, batch_size=None, momentum=0.0,
                                    l1_ratio=0.05).fit(X, y)
    strong = LinearRegressionScratch(lr=0.05, epochs=3000, batch_size=None, momentum=0.0,
                                      l1_ratio=1.0).fit(X, y)
    assert np.sum(strong.weights == 0.0) >= np.sum(weak.weights == 0.0)
