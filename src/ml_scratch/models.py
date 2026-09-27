"""
models.py
---------
Shared gradient-descent engine + Linear/Logistic Regression built on it.

This is a straight extraction of the model code from the original
``final_ml_scratch_project_v3.py`` (Sep 19 submission) into its own
module -- the training/inference logic is UNCHANGED. What moved is
everything else: the ad-hoc ``run_verification_suite()`` / ``_ok()``
assert script that used to share this file is now a proper pytest
suite under ``tests/`` (see ``tests/README.md`` for the file-by-file
mapping), so this module can be imported anywhere (e.g. as an
optimizer baseline elsewhere in the thesis) without dragging a
verification script behind it.

Objective minimised by both models (bias is NEVER regularised):

    J(w, b) = data_loss(w, b) + l2_ratio * ||w||_2^2 + l1_ratio * ||w||_1

- data_loss is the MEAN loss (MSE or binary cross-entropy).
- L2 enters through its exact gradient: 2 * l2_ratio * w
- L1 is handled with a PROXIMAL step (soft-thresholding) instead of the
  plain sub-gradient sign(w): the sub-gradient never produces exact
  zeros, soft-thresholding does (real Lasso sparsity).
- batch_size=None (or >= n) -> Batch GD, 1 -> SGD, otherwise Mini-Batch.

Convergence controls:
- lr_decay : inverse-time schedule  lr_t = lr / (1 + lr_decay * epoch).
             With a constant lr, mini-batch/SGD stalls on a noise floor
             around the minimiser; a decaying lr shrinks that noise so
             the iterate can settle on the exact minimiser
             (Robbins-Monro). lr_decay=0 -> constant lr.
- tol      : early stopping. Training stops when the full objective J
             changes by less than tol * max(1, |J|) for
             ``n_iter_no_change`` consecutive epochs. tol=None disables it.
             Fitted attributes: n_iter_ (epochs actually run), converged_.
"""

from __future__ import annotations

import numpy as np


class _GDModelBase:
    """Shared fit() loop (batching, momentum, L1/L2, lr_decay, early stopping).

    Subclasses implement only the three things that differ between a
    regression and a classification loss: ``_error``, ``_grad_scale``,
    ``_data_loss`` (and ``_check_targets`` when the targets need
    validating, e.g. binary-only for logistic regression).
    """

    def __init__(self, lr=0.05, epochs=50, batch_size=32, momentum=0.9,
                 l1_ratio=0.0, l2_ratio=0.0, random_state=42,
                 lr_decay=0.0, tol=None, n_iter_no_change=5):
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
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.n_iter_ = 0
        self.converged_ = False

    # --- hooks implemented by subclasses --------------------------------
    def _check_targets(self, y):
        pass

    def _error(self, X, y):
        """Return (prediction - target) used by the gradient."""
        raise NotImplementedError

    def _grad_scale(self):
        """Constant in front of X^T @ error / m in the data-loss gradient."""
        raise NotImplementedError

    def _data_loss(self, X, y):
        raise NotImplementedError

    # --- shared machinery ------------------------------------------------
    def _penalty(self):
        return (self.l2_ratio * np.sum(self.weights ** 2)
                + self.l1_ratio * np.sum(np.abs(self.weights)))

    def _iter_batches(self, X, y, rng):
        n = X.shape[0]
        idx = rng.permutation(n)                      # shuffle every epoch
        if self.batch_size is None or self.batch_size >= n:
            yield X[idx], y[idx]
        else:
            for i in range(0, n, self.batch_size):
                sel = idx[i:i + self.batch_size]
                yield X[sel], y[sel]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1-D with length n_samples")
        self._check_targets(y)

        rng = np.random.default_rng(self.random_state)   # local RNG, no global state
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []                            # reset on every fit()
        self.n_iter_ = 0
        self.converged_ = False
        stall = 0

        v_w = np.zeros(n_features)
        v_b = 0.0

        for epoch in range(self.epochs):
            lr_t = self.lr / (1.0 + self.lr_decay * epoch)   # inverse-time decay
            for X_b, y_b in self._iter_batches(X, y, rng):
                m = X_b.shape[0]
                err = self._error(X_b, y_b)
                dw = (self._grad_scale() / m) * (X_b.T @ err)
                db = (self._grad_scale() / m) * np.sum(err)

                if self.l2_ratio > 0:
                    dw = dw + 2.0 * self.l2_ratio * self.weights

                v_w = self.momentum * v_w + lr_t * dw
                v_b = self.momentum * v_b + lr_t * db
                self.weights -= v_w
                self.bias -= v_b

                if self.l1_ratio > 0:                     # proximal (soft-threshold) step
                    thr = lr_t * self.l1_ratio
                    self.weights = np.sign(self.weights) * np.maximum(np.abs(self.weights) - thr, 0.0)

            # full objective (data loss + penalty) so the curve is comparable to J
            self.loss_history.append(self._data_loss(X, y) + self._penalty())
            self.n_iter_ = epoch + 1

            if self.tol is not None and len(self.loss_history) >= 2:
                prev, cur = self.loss_history[-2], self.loss_history[-1]
                stall = stall + 1 if abs(prev - cur) < self.tol * max(1.0, abs(prev)) else 0
                if stall >= self.n_iter_no_change:
                    self.converged_ = True
                    break
        return self


class LinearRegressionScratch(_GDModelBase):
    """MSE loss: J = (1/n)||y - Xw - b||^2 + l2*||w||^2 + l1*||w||_1"""

    def _error(self, X, y):
        return X @ self.weights + self.bias - y

    def _grad_scale(self):
        return 2.0

    def _data_loss(self, X, y):
        return float(np.mean((X @ self.weights + self.bias - y) ** 2))

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias


class LogisticRegressionScratch(_GDModelBase):
    """Binary cross-entropy: J = mean BCE + l2*||w||^2 + l1*||w||_1"""

    @staticmethod
    def _sigmoid(z):
        # numerically stable sigmoid (no overflow for large |z|)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        e = np.exp(z[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def _check_targets(self, y):
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("LogisticRegressionScratch expects binary targets in {0, 1}")

    def _error(self, X, y):
        return self._sigmoid(X @ self.weights + self.bias) - y

    def _grad_scale(self):
        return 1.0

    def _data_loss(self, X, y):
        p = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X, dtype=float) @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
