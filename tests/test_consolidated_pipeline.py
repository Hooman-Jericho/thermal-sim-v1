import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.baseline_dynamics_learning.run_consolidated_pipeline import (
    FEATURE_SET,
    MODEL_NAME,
    TARGET_MODE,
    _unwrap_to_sklearn_rf,
    run,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

FAST_OVERRIDES = {
    "data": {
        "n_episodes": 6,
        "steps_per_episode": 150,
        "u_hold_steps": 50,
        "n_test_episodes": 2,
    },
    "feature_study": {"n_folds": 3},
}


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("consolidated_run")
    cwd = os.getcwd()
    os.environ.setdefault("WANDB_MODE", "offline")
    os.chdir(workdir)
    try:
        manifest = run(extra_overrides=FAST_OVERRIDES)
    finally:
        os.chdir(cwd)
    return {"manifest": manifest, "workdir": workdir}


# --- the literal Day 20 Definition of Done, as a regression test ------------


def test_zero_manual_intervention_on_a_fresh_machine(tmp_path):
    """Runs the ACTUAL script in a subprocess with no WANDB_API_KEY, no cached
    login, and no stdin -- the exact scenario the Sep-28 submission crashed on
    (wandb.errors.errors.UsageError: No API key configured). Must exit 0.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("WANDB_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.baseline_dynamics_learning.run_consolidated_pipeline",
        ],
        cwd=str(tmp_path),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Pipeline did not survive a fresh, unauthenticated machine.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (tmp_path / "outputs" / "consolidated_pipeline" / "manifest.json").exists()


# --- structural checks on a fast run -----------------------------------------


def test_manifest_has_expected_keys(pipeline_result):
    m = pipeline_result["manifest"]
    for key in [
        "recipe",
        "config_fingerprint",
        "n_train_episodes",
        "n_test_episodes",
        "cv_mean_mse",
        "cv_std_mse",
        "test_scores",
        "safety_agreement",
        "wandb_mode",
    ]:
        assert key in m


def test_recipe_matches_day19_finding(pipeline_result):
    """Regression test for a specific claim in the module docstring: the
    recipe is the one Day 19 measured as best for Random Forest, not an
    arbitrary or re-tuned choice."""
    assert (MODEL_NAME, FEATURE_SET, TARGET_MODE) == (
        "random_forest",
        "full",
        "physics_residual",
    )
    assert pipeline_result["manifest"]["recipe"] == {
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET,
        "target_mode": TARGET_MODE,
    }


def test_all_expected_output_files_exist(pipeline_result):
    out_dir = pipeline_result["workdir"] / "outputs" / "consolidated_pipeline"
    for name in [
        "thermal_dataset_raw.csv",
        "thermal_dataset_engineered.csv",
        "random_forest_consolidated.pkl",
        "manifest.json",
        "feature_importance.png",
        "residuals.png",
        "cv_scores.png",
    ]:
        assert (out_dir / name).exists(), name


def test_cv_score_is_not_worse_than_a_generous_sanity_bound(pipeline_result):
    """Loose bound (not a tight regression target): catches a totally broken
    pipeline without being sensitive to exact numbers on a 6-episode smoke config."""
    assert 0 <= pipeline_result["manifest"]["cv_mean_mse"] < 1.0


def test_safety_agreement_reports_a_missed_violation_rate(pipeline_result):
    safety = pipeline_result["manifest"]["safety_agreement"]
    assert "missed_violation_pct" in safety
    assert safety["true_violation_pct"] >= 0


def test_wandb_mode_in_manifest_matches_config_default():
    """Documents WHY 'zero manual intervention' holds: the default is offline,
    not a mode chosen ad hoc by whoever runs it."""
    import yaml

    cfg = yaml.safe_load(
        open(REPO_ROOT / "experiments" / "baseline_dynamics_learning" / "config.yaml")
    )
    assert cfg["ml_pipeline"]["wandb_mode"] == "offline"


# --- _unwrap_to_sklearn_rf ----------------------------------------------------


def test_unwrap_to_sklearn_rf_finds_feature_importances(pipeline_result):
    """Re-fit a tiny model through the real make_model() wrapping chain and
    confirm the unwrap helper reaches an object with feature_importances_."""
    import pandas as pd

    from experiments.baseline_dynamics_learning.estimators import make_model
    from src.plant import PlantConfig

    plant = PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=1.0)
    X = pd.DataFrame(
        {"T_current": [30.0, 40.0, 50.0, 35.0], "u": [10.0, 20.0, 30.0, 15.0]}
    )
    y = pd.Series([31.0, 41.0, 49.0, 36.0])
    models_cfg = {"random_forest": {"n_estimators": 5, "max_depth": 3}}
    model = make_model(
        "random_forest", "physics_residual", plant, models_cfg, seed=0
    ).fit(X, y)
    rf = _unwrap_to_sklearn_rf(model)
    assert hasattr(rf, "feature_importances_")
    assert len(rf.feature_importances_) == 2


def test_unwrap_to_sklearn_rf_raises_for_a_model_without_one():
    class NotATree:
        pass

    with pytest.raises(AttributeError):
        _unwrap_to_sklearn_rf(NotATree())
