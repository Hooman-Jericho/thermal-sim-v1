"""Gradient-descent Linear and Logistic Regression implemented from scratch.

A shared gradient-descent engine plus Linear/Logistic Regression built on it.

This is a straight extraction of the model code from the original
``final_ml_scratch_project_v3.py`` (Sep 19 submission) into its own
module -- the training/inference logic is UNCHANGED. What moved is
everything else: the ad-hoc ``run_verification_suite()`` / ``_ok()``
assert script that used to share this file is now a proper pytest
suite under ``tests/`` (see ``src/ml_scratch/README.md`` for the file-by-file
mapping), so this module can be imported anywhere (e.g. as an
optimizer baseline elsewhere in the thesis) without dragging a
verification script behind it.

Objective minimised by both models (bias is NEVER regularised)::

    J(w, b) = data_loss(w, b) + l2_ratio * ||w||_2^2 + l1_ratio * ||w||_1

- data_loss is the MEAN loss (MSE or binary cross-entropy).
- L2 enters through its exact gradient: 2 * l2_ratio * w
- L1 is handled with a PROXIMAL step (soft-thresholding) instead of the
  plain sub-gradient sign(w): the sub-gradient never produces exact
  zeros, soft-thresholding does (real Lasso sparsity).
- batch_size=None (or >= n) -> Batch GD, 1 -> SGD, otherwise Mini-Batch.

Convergence controls:

- lr_decay : inverse-time schedule  lr_t = lr / (1 + lr_decay * epoch).
  With a constant lr, mini-batch/SGD stalls on a noise floor around the
  minimiser; a decaying lr shrinks that noise so the iterate can settle
  on the exact minimiser (Robbins-Monro). lr_decay=0 -> constant lr.
- tol : early stopping. Training stops when the full objective J changes
  by less than tol * max(1, |J|) for ``n_iter_no_change`` consecutive
  epochs. tol=None disables it. Fitted attributes: n_iter_ (epochs
  actually run), converged_.

Input checks: ``fit`` rejects non-finite data and raises
``FloatingPointError`` if the loss diverges; ``predict`` before ``fit``
raises ``RuntimeError``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
_M = TypeVar("_M", bound="_GDModelBase")


class _GDModelBase:
    """Shared fit() loop (batching, momentum, L1/L2, lr_decay, early stopping).

    Subclasses implement only the three things that differ between a
    regression and a classification loss: ``_error``, ``_grad_scale``,
    ``_data_loss`` (and ``_check_targets`` when the targets need
    validating, e.g. binary-only for logistic regression).

    Parameters
    ----------
    lr : float
        Base learning rate, must be > 0.
    epochs : int
        Maximum number of passes over the data, must be >= 1.
    batch_size : int or None
        Mini-batch size; None (or >= n_samples) means full-batch GD.
    momentum : float
        Heavy-ball momentum coefficient in [0, 1).
    l1_ratio, l2_ratio : float
        L1 (proximal) and L2 penalty strengths, both >= 0.
    random_state : int
        Seed of the local RNG that shuffles samples each epoch.
    lr_decay : float
        Inverse-time decay rate of the learning rate, >= 0.
    tol : float or None
        Early-stopping tolerance on the objective; None disables it.
    n_iter_no_change : int
        Consecutive stalled epochs required to stop early, >= 1.
    """

    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 50,
        batch_size: int | None = 32,
        momentum: float = 0.9,
        l1_ratio: float = 0.0,
        l2_ratio: float = 0.0,
        random_state: int = 42,
        lr_decay: float = 0.0,
        tol: float | None = None,
        n_iter_no_change: int = 5,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if l1_ratio < 0 or l2_ratio < 0:
            raise ValueError("l1_ratio and l2_ratio must be >= 0")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be None or >= 1")
        if lr_decay < 0:
            raise ValueError("lr_decay must be >= 0")
        if tol is not None and tol < 0:
            raise ValueError("tol must be None or >= 0")
        if n_iter_no_change < 1:
            raise ValueError("n_iter_no_change must be >= 1")
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.random_state = random_state
        self.lr_decay = lr_decay
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.weights: FloatArray | None = None
        self.bias: float | None = None
        self.loss_history: list[float] = []
        self.n_iter_ = 0
        self.converged_ = False

    # --- hooks implemented by subclasses --------------------------------
    def _check_targets(self, y: FloatArray) -> None:
        """Validate the targets before training (default: accept anything)."""

    def _error(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Return (prediction - target) used by the gradient."""
        raise NotImplementedError

    def _grad_scale(self) -> float:
        """Constant in front of X^T @ error / m in the data-loss gradient."""
        raise NotImplementedError

    def _data_loss(self, X: FloatArray, y: FloatArray) -> float:
        """Return the mean data loss (without the penalty term)."""
        raise NotImplementedError

    # --- shared machinery ------------------------------------------------
    def _params(self) -> tuple[FloatArray, float]:
        """Return ``(weights, bias)``, raising if the model is not fitted."""
        if self.weights is None or self.bias is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit() first."
            )
        return self.weights, self.bias

    def _linear(self, X: ArrayLike) -> FloatArray:
        """Return ``X @ w + b`` after checking fit state and feature count."""
        w, b = self._params()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if X_arr.shape[1] != w.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but the model was fitted "
                f"with {w.shape[0]} features."
            )
        out: FloatArray = X_arr @ w + b
        return out

    def _penalty(self) -> float:
        """Return the L1 + L2 penalty of the current weights."""
        w, _ = self._params()
        return float(self.l2_ratio * np.sum(w**2) + self.l1_ratio * np.sum(np.abs(w)))

    def _iter_batches(
        self, X: FloatArray, y: FloatArray, rng: np.random.Generator
    ) -> Iterator[tuple[FloatArray, FloatArray]]:
        """Yield shuffled ``(X_batch, y_batch)`` pairs covering one epoch."""
        n = X.shape[0]
        idx = rng.permutation(n)  # shuffle every epoch
        if self.batch_size is None or self.batch_size >= n:
            yield X[idx], y[idx]
        else:
            for i in range(0, n, self.batch_size):
                sel = idx[i : i + self.batch_size]
                yield X[sel], y[sel]

    def fit(self: _M, X: ArrayLike, y: ArrayLike) -> _M:
        """Train the model with (mini-)batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features; must be finite.
        y : array-like of shape (n_samples,)
            Training targets; must be finite.

        Returns
        -------
        The fitted model (``self``).

        Raises
        ------
        ValueError
            If shapes are inconsistent or X / y contain NaN or inf.
        FloatingPointError
            If the loss becomes non-finite (divergence).
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("y must be 1-D with length n_samples")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains NaN or infinite values")
        self._check_targets(y_arr)

        rng = np.random.default_rng(self.random_state)  # local RNG, no global state
        n_features = X_arr.shape[1]
        w = np.zeros(n_features)
        self.weights = w
        self.bias = 0.0
        self.loss_history = []  # reset on every fit()
        self.n_iter_ = 0
        self.converged_ = False
        stall = 0

        v_w = np.zeros(n_features)
        v_b = 0.0

        # Divergence is reported via FloatingPointError below, so silence the
        # raw numpy overflow / invalid-value warnings it would otherwise emit.
        with np.errstate(all="ignore"):
            for epoch in range(self.epochs):
                lr_t = self.lr / (1.0 + self.lr_decay * epoch)  # inverse-time decay
                for X_b, y_b in self._iter_batches(X_arr, y_arr, rng):
                    m = X_b.shape[0]
                    err = self._error(X_b, y_b)
                    dw = (self._grad_scale() / m) * (X_b.T @ err)
                    db = (self._grad_scale() / m) * np.sum(err)

                    if self.l2_ratio > 0:
                        dw = dw + 2.0 * self.l2_ratio * w

                    v_w = self.momentum * v_w + lr_t * dw
                    v_b = self.momentum * v_b + lr_t * db
                    w -= v_w
                    self.bias -= v_b

                    if self.l1_ratio > 0:  # proximal (soft-threshold) step
                        thr = lr_t * self.l1_ratio
                        w[:] = np.sign(w) * np.maximum(np.abs(w) - thr, 0.0)

                # full objective (data loss + penalty) so the curve is comparable
                # to J
                loss = self._data_loss(X_arr, y_arr) + self._penalty()
                if not np.isfinite(loss):
                    raise FloatingPointError(
                        f"Training diverged: loss became {loss} at epoch "
                        f"{epoch + 1}. Scale the features (e.g. standardise "
                        "X) or lower lr."
                    )
                self.loss_history.append(loss)
                self.n_iter_ = epoch + 1

                if self.tol is not None and len(self.loss_history) >= 2:
                    prev, cur = self.loss_history[-2], self.loss_history[-1]
                    stall = (
                        stall + 1
                        if abs(prev - cur) < self.tol * max(1.0, abs(prev))
                        else 0
                    )
                    if stall >= self.n_iter_no_change:
                        self.converged_ = True
                        break
        return self


class LinearRegressionScratch(_GDModelBase):
    """MSE loss: J = (1/n)||y - Xw - b||^2 + l2*||w||^2 + l1*||w||_1."""

    def _error(self, X: FloatArray, y: FloatArray) -> FloatArray:
        w, b = self._params()
        out: FloatArray = X @ w + b - y
        return out

    def _grad_scale(self) -> float:
        return 2.0

    def _data_loss(self, X: FloatArray, y: FloatArray) -> float:
        w, b = self._params()
        return float(np.mean((X @ w + b - y) ** 2))

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict continuous targets for ``X`` of shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If ``X`` has a different number of features than at fit time.
        """
        return self._linear(X)


class LogisticRegressionScratch(_GDModelBase):
    """Binary cross-entropy: J = mean BCE + l2*||w||^2 + l1*||w||_1."""

    @staticmethod
    def _sigmoid(z: FloatArray) -> FloatArray:
        # numerically stable sigmoid (no overflow for large |z|)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        e = np.exp(z[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def _check_targets(self, y: FloatArray) -> None:
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError(
                "LogisticRegressionScratch expects binary targets in {0, 1}"
            )

    def _error(self, X: FloatArray, y: FloatArray) -> FloatArray:
        w, b = self._params()
        return self._sigmoid(X @ w + b) - y

    def _grad_scale(self) -> float:
        return 1.0

    def _data_loss(self, X: FloatArray, y: FloatArray) -> float:
        p = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return P(y = 1) for ``X`` of shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If ``X`` has a different number of features than at fit time.
        """
        return self._sigmoid(self._linear(X))

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> NDArray[np.int_]:
        """Predict class labels in {0, 1} by thresholding ``predict_proba``."""
        return (self.predict_proba(X) >= threshold).astype(int)
