"""
conftest.py
-----------
Shared fixtures for the ml_scratch test suite.

Two reference datasets are used throughout, each fit once per test
session so every test file compares against the SAME sklearn
reference (no per-file duplication, no risk of two files silently
using different random data):

- ``regression_data`` -- ``make_regression`` + train/test split +
  ``StandardScaler`` fit on TRAIN only (no leakage), plus a fitted
  ``sklearn.LinearRegression`` (the "ground truth" weights every
  linear test compares against).
- ``classification_data`` -- ``make_classification`` (stratified
  split), plus a helper to build an unregularized reference
  ``sklearn.LogisticRegression`` (``sk_logreg_factory``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionData:
    X_tr: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    sk: LinearRegression
    n: int  # number of training samples


@dataclass
class ClassificationData:
    X_tr: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    sk: LogisticRegression  # unregularized reference (C=inf)
    n: int  # number of training samples


def _sk_logreg(**kw) -> LogisticRegression:
    """sklearn >= 1.8 API: only C / l1_ratio (the `penalty` argument is deprecated)."""
    return LogisticRegression(max_iter=10000, tol=1e-12, **kw)


@pytest.fixture(scope="session")
def sk_logreg_factory() -> Callable[..., LogisticRegression]:
    """Factory so tests can build a differently-regularized reference model."""
    return _sk_logreg


@pytest.fixture(scope="session")
def regression_data() -> RegressionData:
    X, y = make_regression(n_samples=1000, n_features=5, noise=10.0, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler().fit(X_tr)  # fit on TRAIN only (no leakage)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    sk = LinearRegression().fit(X_tr, y_tr)
    return RegressionData(
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, sk=sk, n=X_tr.shape[0]
    )


@pytest.fixture(scope="session")
def sparse_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """A separate, higher-dimensional dataset with known-sparse structure (Lasso)."""
    X, y = make_regression(
        n_samples=500, n_features=10, n_informative=3, noise=1.0, random_state=0
    )
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.fixture(scope="session")
def classification_data() -> ClassificationData:
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    sk = _sk_logreg(C=1e10, l1_ratio=0).fit(
        X_tr, y_tr
    )  # effectively unpenalized, no penalty=None warning
    return ClassificationData(
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, sk=sk, n=X_tr.shape[0]
    )
