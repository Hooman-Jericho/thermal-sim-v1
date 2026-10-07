"""Plotting helpers for the consolidated Random Forest pipeline.

Plotting helpers for the consolidated RF pipeline (Day 20). Kept separate
from ``run_consolidated_pipeline.py`` on purpose: a plotting function that
can only run inside a 140-line ``main()`` can't be unit-tested or reused by
``run_feature_study.py``'s own parity plots (which duplicated a smaller
version of this before). All three take/return plain arrays or DataFrames,
not a fitted pipeline object, so they don't care which model produced the
numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Headless backend: scripts here only save figures, never open a window.
plt.switch_backend("Agg")


def plot_feature_importance(
    feature_names: list[str], importances: NDArray[np.float64], out_path: Path
) -> Path:
    """Save a horizontal bar chart of Random Forest feature importances.

    Parameters
    ----------
    feature_names : list of str
        Feature names, one per importance.
    importances : numpy.ndarray
        Importance value per feature.
    out_path : Path
        Destination PNG path.

    Returns
    -------
    Path
        ``out_path``.
    """
    if len(feature_names) != len(importances):
        raise ValueError(
            f"{len(feature_names)} names but {len(importances)} importances"
        )
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(feature_names) + 1.5))
    ax.barh(np.array(feature_names)[order], np.array(importances)[order])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_residual_diagnostics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], out_path: Path
) -> Path:
    """Residual distribution + predicted-vs-actual parity, side by side.

    Residuals are NOT expected to be centered at 0 here: the plant has an
    unmeasured disturbance ``d``, so even a perfect deployable model has an
    irreducible, non-degenerate residual (see ``oracle_prediction`` in
    ``metrics.py``). A residual plot exists to catch *bias* (a mean far from
    0, or a trend against T_current) and heavy tails, not to demand a
    residual of exactly 0.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(residuals, bins=40, color="#d62728", alpha=0.85)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(
        f"Residuals (mean={residuals.mean():.3f}, std={residuals.std():.3f})"
    )
    axes[0].set_xlabel("y_true - y_pred")

    axes[1].scatter(y_true, y_pred, s=6, alpha=0.35)
    lims = [
        float(min(np.min(y_true), np.min(y_pred))),
        float(max(np.max(y_true), np.max(y_pred))),
    ]
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_xlabel("Actual T_next")
    axes[1].set_ylabel("Predicted T_next")
    axes[1].set_title("Predicted vs. actual")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_cv_scores(
    fold_scores: NDArray[np.float64], metric_name: str, out_path: Path
) -> Path:
    """Plot one dot per CV fold plus a mean +/- std band.

    This makes CV VARIANCE visible, not just its mean (a single averaged
    number hides a fold that failed badly).
    """
    fold_scores = np.asarray(fold_scores, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(1, len(fold_scores) + 1)
    ax.scatter(x, fold_scores, zorder=3)
    mean, std = fold_scores.mean(), fold_scores.std()
    ax.axhline(mean, color="gray", linestyle="--", linewidth=1)
    ax.fill_between(
        [0.5, len(fold_scores) + 0.5], mean - std, mean + std, alpha=0.15, color="gray"
    )
    ax.set_xticks(x)
    ax.set_xlabel("Fold (grouped by episode_id)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} across folds: {mean:.4f} +/- {std:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
