import numpy as np
import pytest

from experiments.baseline_dynamics_learning.diagnostics import (
    plot_cv_scores, plot_feature_importance, plot_residual_diagnostics,
)


def test_plot_feature_importance_writes_a_file(tmp_path):
    out = plot_feature_importance(["a", "b", "c"], np.array([0.1, 0.6, 0.3]), tmp_path / "fi.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_feature_importance_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        plot_feature_importance(["a", "b"], np.array([0.1, 0.6, 0.3]), tmp_path / "fi.png")


def test_plot_residual_diagnostics_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.uniform(20, 80, size=200)
    y_pred = y_true + rng.normal(0, 1, size=200)
    out = plot_residual_diagnostics(y_true, y_pred, tmp_path / "resid.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_cv_scores_writes_a_file(tmp_path):
    out = plot_cv_scores(np.array([0.01, 0.012, 0.009, 0.011, 0.0105]), "CV MSE", tmp_path / "cv.png")
    assert out.exists() and out.stat().st_size > 0
