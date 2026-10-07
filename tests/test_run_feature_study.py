import os

import pandas as pd
import pytest

from experiments.baseline_dynamics_learning.features import (
    FEATURES,
    find_redundant_pairs,
)
from experiments.baseline_dynamics_learning.run_feature_study import run

# Offline W&B: the study must run without a login (read at wandb.init time).
os.environ.setdefault("WANDB_MODE", "offline")

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
def study_results(tmp_path_factory):
    """Run the whole study ONCE for this module (it's the expensive part);
    individual tests below only check properties of the shared result."""
    workdir = tmp_path_factory.mktemp("feature_study_run")
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        results = run(extra_overrides=FAST_OVERRIDES)
    finally:
        os.chdir(cwd)
    results["_workdir"] = workdir
    return results


def test_run_produces_all_four_tables(study_results):
    for key in [
        "feature_ablation",
        "target_ablation",
        "reference_floors",
        "safety_agreement",
    ]:
        assert key in study_results
        assert len(study_results[key]) > 0


def test_run_writes_manifest_and_csvs_to_disk(study_results):
    out_dir = study_results["_workdir"] / "outputs" / "feature_study"
    for name in [
        "manifest.json",
        "feature_ablation.csv",
        "target_ablation.csv",
        "reference_floors.csv",
        "safety_agreement.csv",
        "redundancy_report.json",
    ]:
        assert (out_dir / name).exists(), name


def test_feature_ablation_covers_every_configured_model_and_set(study_results):
    fa = study_results["feature_ablation"]
    assert set(fa["model"]) == {"linear", "random_forest", "svm"}
    assert set(fa["feature_set"]) == {"raw", "physics", "history", "observer", "full"}
    assert len(fa) == 3 * 5


def test_target_ablation_covers_every_configured_model_and_mode(study_results):
    ta = study_results["target_ablation"]
    assert set(ta["target_mode"]) == {
        "absolute_unscaled",
        "absolute",
        "delta",
        "physics_residual",
    }
    assert len(ta) == 3 * 4


def test_no_mse_is_negative_or_nan(study_results):
    for key in ["feature_ablation", "target_ablation"]:
        assert (study_results[key]["MSE_mean"] >= 0).all()
        assert not study_results[key]["MSE_mean"].isna().any()


def test_engineered_feature_sets_beat_raw_for_every_model(study_results):
    """The actual Day-19 claim: history/observer/full must reduce MSE below
    'raw' -- the Sep-27 submission's features did NOT do this (measured: MSE
    got slightly worse). This is a regression test for that finding."""
    fa = study_results["feature_ablation"].set_index(["model", "feature_set"])[
        "MSE_mean"
    ]
    for model in ["linear", "random_forest", "svm"]:
        raw_mse = fa[(model, "raw")]
        for feat_set in ["history", "observer", "full"]:
            assert fa[(model, feat_set)] < raw_mse, (
                f"{model}/{feat_set} did not beat raw"
            )


def test_scaled_targets_beat_unscaled_for_tree_and_kernel_models(study_results):
    """Regression test for the Sep-27 SVM scaling artifact (R2=0.44 on raw target)."""
    ta = study_results["target_ablation"].set_index(["model", "target_mode"])[
        "MSE_mean"
    ]
    for model in ["random_forest", "svm"]:
        unscaled = ta[(model, "absolute_unscaled")]
        for mode in ["delta", "physics_residual"]:
            assert ta[(model, mode)] < unscaled, (
                f"{model}/{mode} did not beat absolute_unscaled"
            )


def test_reference_floors_are_ordered_persistence_worst_physics_middle_oracle_best(
    study_results,
):
    floors = study_results["reference_floors"].set_index("reference")["MSE"]
    persistence = floors["persistence (no model)"]
    physics = floors["physics, d=0 (no model)"]
    oracle = floors["oracle disturbance observer (uses true d; NOT deployable)"]
    assert oracle <= physics <= persistence


def test_safety_agreement_has_valid_percentages(study_results):
    sa = study_results["safety_agreement"]
    for col in ["true_violation_pct", "pred_violation_pct"]:
        assert ((sa[col] >= 0) & (sa[col] <= 100)).all()


def test_committed_feature_sets_stay_redundancy_free(study_results):
    """Whatever the study just generated, the shipped feature set must not
    regress into the Sep-27 bug (two engineered columns being affine copies)."""
    df = pd.read_csv(
        study_results["_workdir"]
        / "outputs"
        / "feature_study"
        / "thermal_dataset_features.csv"
    )
    assert find_redundant_pairs(df[list(FEATURES)], threshold=0.999) == []
