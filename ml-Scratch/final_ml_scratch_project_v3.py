import warnings

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 0. SHARED GRADIENT-DESCENT ENGINE
#
#    Objective minimised by both models (bias is NEVER regularised):
#        J(w, b) = data_loss(w, b) + l2_ratio * ||w||_2^2 + l1_ratio * ||w||_1
#
#    - data_loss is the MEAN loss (MSE or binary cross-entropy).
#    - L2 enters through its exact gradient:  2 * l2_ratio * w
#    - L1 is handled with a PROXIMAL step (soft-thresholding) instead of the
#      plain sub-gradient sign(w): the sub-gradient never produces exact
#      zeros, soft-thresholding does (real Lasso sparsity).
#    - batch_size=None (or >= n) -> Batch GD, 1 -> SGD, otherwise Mini-Batch.
#
#    Convergence controls (v3):
#    - lr_decay : inverse-time schedule  lr_t = lr / (1 + lr_decay * epoch).
#                 With a constant lr, mini-batch/SGD stalls on a noise floor
#                 around the minimiser; a decaying lr shrinks that noise so the
#                 iterate can settle on the exact minimiser (Robbins-Monro).
#                 lr_decay=0 -> constant lr (old behaviour).
#    - tol      : early stopping. Training stops when the full objective J
#                 changes by less than  tol * max(1, |J|)  for
#                 `n_iter_no_change` consecutive epochs. tol=None disables it.
#                 Fitted attributes: n_iter_ (epochs actually run), converged_.
# =====================================================================
class _GDModelBase:
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


# =====================================================================
# 1. LINEAR REGRESSION FROM SCRATCH (NumPy)
# =====================================================================
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


# =====================================================================
# 2. LOGISTIC REGRESSION FROM SCRATCH (NumPy)
# =====================================================================
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


# =====================================================================
# 3. VERIFICATION SUITE AGAINST SCIKIT-LEARN  (every check is an assert)
#
#    Parameter mapping scratch -> sklearn (n = number of training samples):
#      Ridge  : sklearn minimises ||y-Xw||^2 + a*||w||^2     -> a = n * l2_ratio
#      Lasso  : sklearn minimises (1/2n)||y-Xw||^2 + a*||w||_1 -> a = l1_ratio / 2
#      LogReg : sklearn minimises C*sum(loss) + 0.5||w||^2
#                 L2: C = 1 / (2 * n * l2_ratio)
#                 L1: C = 1 / (n * l1_ratio)   (penalty term = ||w||_1)
#
#    Two kinds of checks:
#      EXACT       -> full-batch GD, no momentum (deterministic) => weights must
#                     match sklearn to ~1e-6.
#      STATISTICAL -> mini-batch/SGD + momentum never converges to the exact
#                     minimiser (gradient-noise floor), so we compare quality
#                     metrics on a held-out test set instead of raw weights.
# =====================================================================
# sklearn 1.8 emits this harmless notice for C=np.inf (= unpenalised); silence only that message
warnings.filterwarnings("ignore", message="Setting penalty=None", category=UserWarning)


def _sk_logreg(**kw):
    """sklearn >= 1.8 API: only C / l1_ratio (the `penalty` argument is deprecated)."""
    return LogisticRegression(max_iter=10000, tol=1e-12, **kw)


def _ok(name, cond, detail=""):
    assert cond, f"FAILED: {name}  {detail}"
    print(f"  [PASS] {name}  {detail}")


def run_verification_suite():
    print("=" * 69)
    print("   SCRATCH IMPLEMENTATION VERIFICATION AGAINST SCIKIT-LEARN (v3)")
    print("=" * 69)

    # ---------------- data: regression (with train/test split) ----------
    X, y = make_regression(n_samples=1000, n_features=5, noise=10.0, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    sc = StandardScaler().fit(X_tr)                    # fit on TRAIN only (no leakage)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    n = X_tr.shape[0]

    # ---- Test 1a: Linear, EXACT (full-batch) ----
    print("\n--- 1a. LINEAR REGRESSION (full-batch GD, exact) ---")
    lin = LinearRegressionScratch(lr=0.1, epochs=500, batch_size=None, momentum=0.0).fit(X_tr, y_tr)
    sk = LinearRegression().fit(X_tr, y_tr)
    _ok("weights match", np.allclose(lin.weights, sk.coef_, atol=1e-6),
        f"max|dw|={np.max(np.abs(lin.weights - sk.coef_)):.2e}")
    _ok("bias matches", abs(lin.bias - sk.intercept_) < 1e-6, f"|db|={abs(lin.bias - sk.intercept_):.2e}")
    _ok("loss monotonically non-increasing",
        all(a >= b - 1e-12 for a, b in zip(lin.loss_history, lin.loss_history[1:])))

    # ---- Test 1b: Linear, STATISTICAL (mini-batch + momentum) ----
    print("\n--- 1b. LINEAR REGRESSION (mini-batch GD + momentum, statistical) ---")
    mb = LinearRegressionScratch(lr=0.005, epochs=200, batch_size=32, momentum=0.9).fit(X_tr, y_tr)
    r2_mb, r2_sk = r2_score(y_te, mb.predict(X_te)), r2_score(y_te, sk.predict(X_te))
    _ok("test R2 within 0.01 of sklearn", abs(r2_mb - r2_sk) < 0.01,
        f"scratch={r2_mb:.5f} sklearn={r2_sk:.5f}")
    _ok("test MSE within 3% of sklearn",
        mean_squared_error(y_te, mb.predict(X_te)) <= 1.03 * mean_squared_error(y_te, sk.predict(X_te)))
    sgd = LinearRegressionScratch(lr=0.001, epochs=100, batch_size=1, momentum=0.0).fit(X_tr, y_tr)
    _ok("pure SGD (batch_size=1) test R2 within 0.01", abs(r2_score(y_te, sgd.predict(X_te)) - r2_sk) < 0.01,
        f"R2={r2_score(y_te, sgd.predict(X_te)):.5f}")

    # ---- Test 1c: lr_decay makes mini-batch converge to the MINIMISER (weights, not just R2) ----
    print("\n--- 1c. lr_decay: mini-batch converges to the minimiser ---")
    errs_const, errs_decay = [], []
    for seed in range(4):
        c = LinearRegressionScratch(lr=0.02, epochs=1000, batch_size=32, momentum=0.9,
                                    random_state=seed).fit(X_tr, y_tr)
        d = LinearRegressionScratch(lr=0.02, epochs=1000, batch_size=32, momentum=0.9,
                                    lr_decay=1.0, random_state=seed).fit(X_tr, y_tr)
        errs_const.append(np.max(np.abs(c.weights - sk.coef_)))
        errs_decay.append(np.max(np.abs(d.weights - sk.coef_)))
    _ok("constant lr stalls on a noise floor (>0.3 in every seed)", min(errs_const) > 0.3,
        f"max|dw| per seed={np.round(errs_const, 3)}")
    _ok("lr_decay gets < 0.03 in every seed", max(errs_decay) < 0.03,
        f"max|dw| per seed={np.round(errs_decay, 4)}")
    _ok("lr_decay is >= 30x closer than constant lr on average",
        np.mean(errs_const) / np.mean(errs_decay) >= 30,
        f"ratio={np.mean(errs_const) / np.mean(errs_decay):.0f}x")
    sgd_d = LinearRegressionScratch(lr=0.01, epochs=300, batch_size=1, momentum=0.0, lr_decay=1.0).fit(X_tr, y_tr)
    _ok("pure SGD + lr_decay weights within 0.02", np.max(np.abs(sgd_d.weights - sk.coef_)) < 0.02,
        f"max|dw|={np.max(np.abs(sgd_d.weights - sk.coef_)):.2e}")

    # ---- Test 1d: tol early stopping ----
    print("\n--- 1d. tol early stopping ---")
    es = LinearRegressionScratch(lr=0.02, epochs=2000, batch_size=32, momentum=0.9,
                                 lr_decay=1.0, tol=1e-6).fit(X_tr, y_tr)
    _ok("stops early and flags converged_", es.converged_ and es.n_iter_ < 2000,
        f"n_iter_={es.n_iter_}/2000")
    _ok("len(loss_history) == n_iter_", len(es.loss_history) == es.n_iter_)
    _ok("early-stopped model still accurate (test R2 within 0.005)",
        abs(r2_score(y_te, es.predict(X_te)) - r2_score(y_te, sk.predict(X_te))) < 0.005)
    no_es = LinearRegressionScratch(lr=0.02, epochs=50, batch_size=32, momentum=0.9).fit(X_tr, y_tr)
    _ok("tol=None runs all epochs", no_es.n_iter_ == 50 and not no_es.converged_)
    tight = LinearRegressionScratch(lr=0.1, epochs=5000, batch_size=None, momentum=0.0, tol=1e-14).fit(X_tr, y_tr)
    _ok("full-batch + tol stops at machine-precision plateau and stays exact",
        tight.converged_ and np.allclose(tight.weights, sk.coef_, atol=1e-6),
        f"n_iter_={tight.n_iter_}")

    # ---- Test 2: Ridge (correct mapping a = n * l2) ----
    print("\n--- 2. RIDGE (L2) ---")
    lam = 1.0
    rid = LinearRegressionScratch(lr=0.1, epochs=3000, batch_size=None, momentum=0.0, l2_ratio=lam).fit(X_tr, y_tr)
    sk_r = Ridge(alpha=lam * n).fit(X_tr, y_tr)
    _ok("Ridge weights match (alpha = n*lambda)", np.allclose(rid.weights, sk_r.coef_, atol=1e-6),
        f"max|dw|={np.max(np.abs(rid.weights - sk_r.coef_)):.2e}")
    _ok("Ridge bias matches (bias not regularised)", abs(rid.bias - sk_r.intercept_) < 1e-6)
    _ok("Ridge shrinks weights vs OLS", np.linalg.norm(rid.weights) < np.linalg.norm(lin.weights))
    rid_mb = LinearRegressionScratch(lr=0.005, epochs=300, batch_size=32, momentum=0.9, l2_ratio=0.05).fit(X_tr, y_tr)
    sk_r2 = Ridge(alpha=0.05 * n).fit(X_tr, y_tr)
    _ok("Ridge mini-batch test R2 within 0.01",
        abs(r2_score(y_te, rid_mb.predict(X_te)) - r2_score(y_te, sk_r2.predict(X_te))) < 0.01)
    rid_d = LinearRegressionScratch(lr=0.02, epochs=1000, batch_size=32, momentum=0.9,
                                    l2_ratio=0.05, lr_decay=1.0).fit(X_tr, y_tr)
    _ok("Ridge mini-batch + lr_decay: WEIGHTS within 0.03",
        np.max(np.abs(rid_d.weights - sk_r2.coef_)) < 0.03,
        f"max|dw|={np.max(np.abs(rid_d.weights - sk_r2.coef_)):.2e}")

    # ---- Test 3: Lasso (proximal L1, real sparsity) ----
    print("\n--- 3. LASSO (L1, proximal soft-threshold) ---")
    Xs, ys = make_regression(n_samples=500, n_features=10, n_informative=3, noise=1.0, random_state=0)
    Xs = StandardScaler().fit_transform(Xs)
    l1 = 5.0
    las = LinearRegressionScratch(lr=0.05, epochs=5000, batch_size=None, momentum=0.0, l1_ratio=l1).fit(Xs, ys)
    sk_l = Lasso(alpha=l1 / 2.0, tol=1e-12, max_iter=100000).fit(Xs, ys)
    _ok("Lasso weights match (alpha = l1/2)", np.allclose(las.weights, sk_l.coef_, atol=1e-4),
        f"max|dw|={np.max(np.abs(las.weights - sk_l.coef_)):.2e}")
    _ok("exact zeros produced (sparsity)", int(np.sum(las.weights == 0)) >= 1,
        f"zeros scratch={int(np.sum(las.weights == 0))} sklearn={int(np.sum(sk_l.coef_ == 0))}")
    _ok("same sparsity pattern as sklearn", np.array_equal(las.weights == 0, sk_l.coef_ == 0))
    las_mb = LinearRegressionScratch(lr=0.02, epochs=1000, batch_size=32, momentum=0.0,
                                     l1_ratio=l1, lr_decay=1.0).fit(Xs, ys)
    _ok("Lasso mini-batch + lr_decay finds the same support (non-zero set)",
        np.array_equal(las_mb.weights == 0, sk_l.coef_ == 0),
        f"zeros scratch={int(np.sum(las_mb.weights == 0))} sklearn={int(np.sum(sk_l.coef_ == 0))}")

    # ---------------- data: classification ------------------------------
    Xc, yc = make_classification(n_samples=1000, n_features=5, random_state=42)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25, random_state=42, stratify=yc)
    scc = StandardScaler().fit(Xc_tr)
    Xc_tr, Xc_te = scc.transform(Xc_tr), scc.transform(Xc_te)
    nc = Xc_tr.shape[0]

    # ---- Test 4a: Logistic, EXACT ----
    print("\n--- 4a. LOGISTIC REGRESSION (full-batch GD, exact) ---")
    lg = LogisticRegressionScratch(lr=0.5, epochs=5000, batch_size=None, momentum=0.0).fit(Xc_tr, yc_tr)
    sk_lg = _sk_logreg(C=np.inf).fit(Xc_tr, yc_tr)
    _ok("weights match", np.allclose(lg.weights, sk_lg.coef_[0], atol=1e-4),
        f"max|dw|={np.max(np.abs(lg.weights - sk_lg.coef_[0])):.2e}")
    _ok("bias matches", abs(lg.bias - sk_lg.intercept_[0]) < 1e-4)
    _ok("predictions identical on test set", np.array_equal(lg.predict(Xc_te), sk_lg.predict(Xc_te)))

    # ---- Test 4b: Logistic, STATISTICAL ----
    print("\n--- 4b. LOGISTIC REGRESSION (mini-batch GD + momentum, statistical) ---")
    lg_mb = LogisticRegressionScratch(lr=0.01, epochs=200, batch_size=32, momentum=0.9).fit(Xc_tr, yc_tr)
    ll_s, ll_k = log_loss(yc_te, lg_mb.predict_proba(Xc_te)), log_loss(yc_te, sk_lg.predict_proba(Xc_te)[:, 1])
    acc_s, acc_k = accuracy_score(yc_te, lg_mb.predict(Xc_te)), accuracy_score(yc_te, sk_lg.predict(Xc_te))
    _ok("test LogLoss within 0.01", abs(ll_s - ll_k) < 0.01, f"scratch={ll_s:.5f} sklearn={ll_k:.5f}")
    _ok("test accuracy within 1.5%", abs(acc_s - acc_k) <= 0.015, f"scratch={acc_s:.4f} sklearn={acc_k:.4f}")
    errs_c = np.max(np.abs(lg_mb.weights - sk_lg.coef_[0]))
    lg_d = LogisticRegressionScratch(lr=0.05, epochs=1000, batch_size=32, momentum=0.9, lr_decay=1.0).fit(Xc_tr, yc_tr)
    errs_d = np.max(np.abs(lg_d.weights - sk_lg.coef_[0]))
    _ok("Logistic mini-batch + lr_decay: WEIGHTS within 0.01", errs_d < 0.01,
        f"max|dw|={errs_d:.2e} (constant lr: {errs_c:.2e})")
    _ok("Logistic mini-batch + lr_decay: bias within 0.01", abs(lg_d.bias - sk_lg.intercept_[0]) < 0.01)

    # ---- Test 5: Logistic + L2 (C = 1 / (2 n lambda)) ----
    print("\n--- 5. LOGISTIC + L2 ---")
    lam_c = 0.01
    lg_l2 = LogisticRegressionScratch(lr=0.5, epochs=5000, batch_size=None, momentum=0.0, l2_ratio=lam_c).fit(Xc_tr, yc_tr)
    sk_l2 = _sk_logreg(C=1.0 / (2.0 * nc * lam_c)).fit(Xc_tr, yc_tr)
    _ok("weights match (C = 1/(2 n lambda))", np.allclose(lg_l2.weights, sk_l2.coef_[0], atol=1e-4),
        f"max|dw|={np.max(np.abs(lg_l2.weights - sk_l2.coef_[0])):.2e}")

    # ---- Test 6: Logistic + L1 (C = 1 / (n l1)) ----
    print("\n--- 6. LOGISTIC + L1 ---")
    l1_c = 0.05
    lg_l1 = LogisticRegressionScratch(lr=0.5, epochs=5000, batch_size=None, momentum=0.0, l1_ratio=l1_c).fit(Xc_tr, yc_tr)
    sk_l1 = _sk_logreg(C=1.0 / (nc * l1_c), l1_ratio=1.0, solver="saga").fit(Xc_tr, yc_tr)
    _ok("L1 weights match (C = 1/(n l1))", np.allclose(lg_l1.weights, sk_l1.coef_[0], atol=5e-3),
        f"max|dw|={np.max(np.abs(lg_l1.weights - sk_l1.coef_[0])):.2e}")
    _ok("L1 produces exact zeros", int(np.sum(lg_l1.weights == 0)) >= 1,
        f"zeros scratch={int(np.sum(lg_l1.weights == 0))} sklearn={int(np.sum(sk_l1.coef_[0] == 0))}")

    # ---- Test 7: engineering checks ----
    print("\n--- 7. ENGINEERING CHECKS ---")
    m = LinearRegressionScratch(epochs=5).fit(X_tr, y_tr)
    m.fit(X_tr, y_tr)
    _ok("loss_history resets on refit (len == epochs)", len(m.loss_history) == 5)
    a = LinearRegressionScratch(epochs=10, random_state=7).fit(X_tr, y_tr)
    b = LinearRegressionScratch(epochs=10, random_state=7).fit(X_tr, y_tr)
    _ok("reproducible with same seed", np.array_equal(a.weights, b.weights))
    st = np.random.get_state()[1].copy()
    LinearRegressionScratch(epochs=3).fit(X_tr, y_tr)
    _ok("global NumPy RNG state untouched", np.array_equal(st, np.random.get_state()[1]))
    try:
        LogisticRegressionScratch().fit(Xc_tr, yc_tr + 2)
        _ok("non-binary targets rejected", False)
    except ValueError:
        _ok("non-binary targets rejected", True)
    for kw in ({"lr_decay": -1.0}, {"tol": -1.0}, {"n_iter_no_change": 0}):
        try:
            LinearRegressionScratch(**kw)
            _ok(f"invalid {list(kw)[0]} rejected", False)
        except ValueError:
            _ok(f"invalid {list(kw)[0]} rejected", True)
    big = LogisticRegressionScratch._sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    _ok("sigmoid stable for extreme inputs", np.allclose(big, [0.0, 0.5, 1.0]) and np.all(np.isfinite(big)))

    print("\n" + "=" * 69)
    print("   ALL CHECKS PASSED")
    print("=" * 69)


if __name__ == "__main__":
    run_verification_suite()
