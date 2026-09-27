# ml-scratch v3.1 — restructured

Gradient-descent Linear/Logistic Regression implemented from scratch
(NumPy only), verified against scikit-learn.

This is a restructuring of the Sep 19 submission
(`ml-Scratch/final_ml_scratch_project_v3.py`) — **the math is
unchanged**; what moved is everything around it: implementation,
verification, and plotting are now three separate, independently
runnable things instead of one 429-line script with an
`assert`-based verification block glued to the bottom.

This package lives under `src/ml_scratch/` -- a SIBLING of
`src/core.py`/`src/systems.py`, not a replacement for them -- and
follows the exact same import/run convention already used by the rest
of `thermal-sim-v1` (`from src.<module> import ...`, run with
`PYTHONPATH=.`). The old `ml-Scratch/` folder at the repo root is
superseded by this and can be removed (see the repo-root
`DEPLOY_NOTES.md` this ships alongside).

```
src/ml_scratch/
    models.py      # _GDModelBase, LinearRegressionScratch, LogisticRegressionScratch
    plotting.py    # convergence_plot.png  (python -m src.ml_scratch.plotting)
tests/
    conftest.py                       # shared fixtures: regression/classification data, sklearn refs
    test_linear_regression.py         # exact (full-batch) + statistical (mini-batch/SGD) checks
    test_lr_schedule_and_stopping.py  # lr_decay convergence proof, early-stopping (tol)
    test_regularization_linear.py     # Ridge/Lasso vs sklearn, exact alpha mapping, exact zeros
    test_logistic_regression.py       # exact/statistical checks, sigmoid stability, input validation
    test_regularization_logistic.py   # L2/L1 vs sklearn's C mapping
    test_engineering.py               # reproducibility, RNG isolation, refit reset, param validation
```

## Run (from the repo root, after merging -- see DEPLOY_NOTES.md)

```bash
pip install -r requirements.txt   # pandas/wandb/joblib/pyyaml already added for experiments/; nothing new needed here
PYTHONPATH=. pytest tests/ -v -k "linear_regression or logistic_regression or regularization or engineering or lr_schedule"
PYTHONPATH=. python -m src.ml_scratch.plotting   # -> convergence_plot.png
```

## What changed vs. the Sep 19 submission, and why

1. **Implementation moved to an importable package** (`src.ml_scratch.models`).
   The original file couldn't be imported without also running (or
   skipping) its verification block. Now `from src.ml_scratch.models import
   LinearRegressionScratch` works anywhere — e.g. as an optimizer
   baseline elsewhere in the thesis — with nothing else attached.

2. **Verification rewritten as pytest, not `assert` + `print("[PASS]")`.**
   Same rigor (exact-vs-statistical split, exact sklearn parameter
   mappings for regularization, RNG-isolation checks) — now you can run
   one test, get a real stack trace on failure, and put it in CI.

3. **One real bug found and fixed *in the test design itself* while
   porting:** `test_l2_shrinks_weight_norm_as_ratio_increases` originally
   used `l2_ratio=2.0` at `lr=0.5`. Plain gradient descent on an L2 term
   is only a contraction when `lr * 2 * l2_ratio < 1`; at those values
   the factor is exactly `-1` (marginal stability), so the "more
   regularized" run diverged to a *larger* norm than the "less
   regularized" one — the opposite of what the test meant to check.
   This is standard GD step-size theory, not a bug in `models.py` itself,
   but it was silently wrong as a *test*. Fixed by picking `l2_ratio=0.5`
   for the "strong" case (comfortably inside the stable region) and
   added `test_l2_diverges_when_step_size_exceeds_stability_bound` so the
   instability itself is now a documented, tested property instead of a
   trap for whoever tunes these hyperparameters next.

4. **sklearn 1.8 API cleanup:** the old `penalty="l1"/"l2"` kwargs are
   deprecated in favor of `l1_ratio`; tests use the new API so the suite
   runs with zero warnings.

Nothing about the training math (batching, momentum, proximal L1,
lr_decay, early stopping, numerically-stable sigmoid) was touched —
see `models.py`'s module docstring for the full spec, carried over
verbatim from the original.
