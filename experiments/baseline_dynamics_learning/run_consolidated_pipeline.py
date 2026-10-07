"""Run the consolidated end-to-end Random Forest pipeline (Day 20).

Day 20 deliverable. Definition of Done (Super Planner): "pipeline runs
end-to-end (raw data -> trained RF -> W&B log) with zero manual
intervention, on a fresh run."

"Consolidation" here means exactly that and nothing more: wire the
already-fixed, already-tested pieces (``data_generation.py``,
``features.py``, ``estimators.py``, ``config.py``, ``metrics.py``,
``diagnostics.py``) into ONE clean run for a single model (Random Forest),
with the diagnostics and robustness checks a "production" pipeline needs.
It deliberately does NOT re-implement data generation, feature engineering,
or the target-mode machinery -- a from-scratch reimplementation is exactly
what caused a submission on this same day to reintroduce four bugs that
were already found and fixed earlier (unmeasured-disturbance leakage,
affine-copy features, a dead 100%-constant safety metric, and physically
unrealistic actuator limits). See DAY20_CONSOLIDATION.md for the full
comparison against that submission.

Recipe (feature_set, target_mode) is not re-tuned here -- it is read
straight from Day 19's measured results: for Random Forest, ``full``
features + ``physics_residual`` target gave the lowest CV MSE
(``DAY19_FEATURE_STUDY.md``). Re-deriving it would just be Day 19 again.

"Zero manual intervention on a fresh run" is not a claim printed at the
end of a log -- it is enforced by construction: ``ml_pipeline.wandb_mode``
defaults to "offline" in config.yaml, so nothing here ever calls
``wandb.init()`` in its default, network/login-requiring "online" mode
unless a human explicitly opts in. ``tests/test_consolidated_pipeline.py``
has a dedicated regression test that runs this exact script with no
WANDB_API_KEY and no cached login and asserts it does not raise.

Run::

    PYTHONPATH=. python -m \
        experiments.baseline_dynamics_learning.run_consolidated_pipeline
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import wandb
from sklearn.model_selection import GroupKFold

from experiments.baseline_dynamics_learning.config import (
    DEFAULT_BASE,
    DEFAULT_OVERRIDES,
    config_fingerprint,
    datagen_from_cfg,
    load_config,
    plant_from_cfg,
)
from experiments.baseline_dynamics_learning.data_generation import (
    TARGET_COLUMN,
    episode_train_test_split,
    generate_dataset,
    generate_stress_test_episodes,
)
from experiments.baseline_dynamics_learning.diagnostics import (
    plot_cv_scores,
    plot_feature_importance,
    plot_residual_diagnostics,
)
from experiments.baseline_dynamics_learning.estimators import make_model
from experiments.baseline_dynamics_learning.features import (
    build_features,
    resolve_feature_set,
)
from experiments.baseline_dynamics_learning.metrics import (
    persistence_prediction,
    physics_prediction,
    regression_scores,
    safety_agreement_metrics,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# The Day-19-measured best recipe for Random Forest. See module docstring.
MODEL_NAME = "random_forest"
FEATURE_SET = "full"
TARGET_MODE = "physics_residual"


def _unwrap_to_sklearn_rf(fitted_model: Any) -> Any:
    """Return the fitted ``RandomForestRegressor`` inside a wrapped model.

    Walks ResidualRegressor -> TransformedTargetRegressor -> Pipeline down
    to the actual fitted RandomForestRegressor, wherever make_model() put it.

    A hasattr-based walk (rather than hardcoding one nesting depth) so this
    keeps working if estimators.py's wrapping order ever changes.
    """
    m = fitted_model
    for _ in range(6):
        if hasattr(m, "feature_importances_"):
            return m
        if hasattr(m, "estimator_"):  # ResidualRegressor
            m = m.estimator_
        elif hasattr(m, "regressor_"):  # TransformedTargetRegressor (fitted)
            m = m.regressor_
        elif hasattr(m, "named_steps"):  # Pipeline
            m = m.named_steps["model"]
        else:
            break
    raise AttributeError(
        f"Could not find a fitted RandomForestRegressor inside {type(fitted_model)}"
    )


def run(
    overrides_path: str | Path | None = None,
    extra_overrides: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the consolidated pipeline end to end.

    Parameters
    ----------
    overrides_path : str or Path or None
        Overrides YAML merged over the base config; defaults to
        ``DEFAULT_OVERRIDES``.
    extra_overrides : dict or None
        In-memory config overrides applied last.
    cfg : dict or None
        A fully resolved config; if given, no file is loaded.

    Returns
    -------
    dict[str, Any]
        Summary of the run (metrics and output paths).
    """
    if cfg is None:
        cfg = load_config(
            DEFAULT_BASE,
            overrides_path or DEFAULT_OVERRIDES,
            extra_overrides=extra_overrides,
        )
    plant = plant_from_cfg(cfg)
    gen_cfg = datagen_from_cfg(cfg)
    n_folds = cfg["feature_study"]["n_folds"]

    out_dir = Path("outputs/consolidated_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw data -----------------------------------------------------------
    logger.info("Generating raw thermal dataset...")
    raw_df = generate_dataset(
        plant, gen_cfg, output_path=str(out_dir / "thermal_dataset_raw.csv")
    )

    # 2. Features (leakage-guarded, redundancy-audited -- see features.py) --
    logger.info("Engineering features (recipe: %s + %s)...", FEATURE_SET, TARGET_MODE)
    df = build_features(raw_df, plant)
    df.to_csv(out_dir / "thermal_dataset_engineered.csv", index=False)
    cols = resolve_feature_set(FEATURE_SET)

    train_df, test_df = episode_train_test_split(
        df, n_test_episodes=cfg["data"]["n_test_episodes"], seed=cfg["random_seed"]
    )

    run_ctx = wandb.init(
        project=cfg["ml_pipeline"]["wandb_project"],
        name="RF_consolidated_day20",
        config={
            **cfg,
            "recipe": {
                "model": MODEL_NAME,
                "feature_set": FEATURE_SET,
                "target_mode": TARGET_MODE,
            },
        },
        mode=cfg["ml_pipeline"]["wandb_mode"],
        reinit="finish_previous",
    )
    dataset_artifact = wandb.Artifact("thermal_dataset_consolidated", type="dataset")
    dataset_artifact.add_file(str(out_dir / "thermal_dataset_engineered.csv"))
    run_ctx.log_artifact(dataset_artifact)

    # 3. Robustness check: GroupKFold CV on the TRAIN partition only --------
    # Grouped by episode_id -- a random KFold here would leak the same way a
    # random train/test split does (adjacent rows are autocorrelated).
    logger.info(
        "Cross-validating (GroupKFold, %d folds, grouped by episode_id)...", n_folds
    )
    folds = list(
        GroupKFold(n_splits=n_folds).split(train_df, groups=train_df["episode_id"])
    )
    fold_r2, fold_mse = [], []
    for tr_idx, val_idx in folds:
        m = make_model(
            MODEL_NAME, TARGET_MODE, plant, cfg["models"], cfg["random_seed"]
        )
        m.fit(train_df.iloc[tr_idx][cols], train_df.iloc[tr_idx][TARGET_COLUMN])
        val = train_df.iloc[val_idx]
        pred = m.predict(val[cols])
        scores = regression_scores(
            val[TARGET_COLUMN].to_numpy(),
            pred,
            val["T_current"].to_numpy(),
            persistence_prediction(val["T_current"].to_numpy()),
            physics_prediction(
                val["T_current"].to_numpy(),
                val["u"].to_numpy(),
                plant.dt,
                plant.mCp,
                plant.k_loss,
                plant.T_amb,
            ),
        )
        fold_r2.append(scores["R2_T_next"])
        fold_mse.append(scores["MSE"])
    fold_r2_arr = np.array(fold_r2)
    fold_mse_arr = np.array(fold_mse)
    logger.info("CV R2 per fold: %s", np.round(fold_r2_arr, 5))
    logger.info("CV MSE: %.5f +/- %.5f", fold_mse_arr.mean(), fold_mse_arr.std())
    wandb.log(
        {
            "CV_Mean_R2": float(fold_r2_arr.mean()),
            "CV_Std_R2": float(fold_r2_arr.std()),
            "CV_Mean_MSE": float(fold_mse_arr.mean()),
            "CV_Std_MSE": float(fold_mse_arr.std()),
        }
    )
    cv_plot_path = plot_cv_scores(fold_mse_arr, "CV MSE", out_dir / "cv_scores.png")
    wandb.log({"CV_scores": wandb.Image(str(cv_plot_path))})

    # 4. Final fit on the full training partition ----------------------------
    logger.info("Training final Random Forest on the full training partition...")
    model = make_model(
        MODEL_NAME, TARGET_MODE, plant, cfg["models"], cfg["random_seed"]
    )
    model.fit(train_df[cols], train_df[TARGET_COLUMN])

    # 5. Held-out evaluation: honest scores + safety agreement ---------------
    y_pred = model.predict(test_df[cols])
    scores = regression_scores(
        test_df[TARGET_COLUMN].to_numpy(),
        y_pred,
        test_df["T_current"].to_numpy(),
        persistence_prediction(test_df["T_current"].to_numpy()),
        physics_prediction(
            test_df["T_current"].to_numpy(),
            test_df["u"].to_numpy(),
            plant.dt,
            plant.mCp,
            plant.k_loss,
            plant.T_amb,
        ),
    )
    logger.info("Held-out test scores: %s", {k: round(v, 5) for k, v in scores.items()})
    wandb.log(scores)

    logger.info("Building safety stress-test set...")
    stress_raw = generate_stress_test_episodes(
        plant,
        n_episodes=cfg["data"]["n_test_episodes"],
        duration_steps=cfg["data"]["steps_per_episode"],
        u_level=cfg["data"]["u_max"],
        disturbance_std=gen_cfg.disturbance_std,
        base_seed=cfg["random_seed"],
        disturbance_autocorr=gen_cfg.disturbance_autocorr,
    )
    stress_df = build_features(stress_raw, plant)
    stress_pred = model.predict(stress_df[cols])
    safety = safety_agreement_metrics(
        stress_df[TARGET_COLUMN].to_numpy(),
        stress_pred,
        cfg["physics"]["T_min"],
        cfg["physics"]["T_max"],
    )
    logger.info(
        "Safety agreement (stress test): %s",
        {k: round(v, 3) for k, v in safety.items()},
    )
    wandb.log({f"safety_{k}": v for k, v in safety.items()})

    # 6. Diagnostics -----------------------------------------------------------
    rf = _unwrap_to_sklearn_rf(model)
    imp_path = plot_feature_importance(
        cols, rf.feature_importances_, out_dir / "feature_importance.png"
    )
    resid_path = plot_residual_diagnostics(
        test_df[TARGET_COLUMN].to_numpy(), y_pred, out_dir / "residuals.png"
    )
    wandb.log(
        {
            "Feature_Importance": wandb.Image(str(imp_path)),
            "Residual_Analysis": wandb.Image(str(resid_path)),
        }
    )

    # 7. Persist + manifest -----------------------------------------------------
    model_path = out_dir / "random_forest_consolidated.pkl"
    joblib.dump(model, model_path)
    model_artifact = wandb.Artifact("RF_model_consolidated", type="model")
    model_artifact.add_file(str(model_path))
    run_ctx.log_artifact(model_artifact)

    manifest = {
        "recipe": {
            "model": MODEL_NAME,
            "feature_set": FEATURE_SET,
            "target_mode": TARGET_MODE,
        },
        "config_fingerprint": config_fingerprint(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "n_train_episodes": int(train_df["episode_id"].nunique()),
        "n_test_episodes": int(test_df["episode_id"].nunique()),
        "cv_mean_mse": float(fold_mse_arr.mean()),
        "cv_std_mse": float(fold_mse_arr.std()),
        "test_scores": scores,
        "safety_agreement": safety,
        "wandb_mode": cfg["ml_pipeline"]["wandb_mode"],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    run_ctx.finish()

    logger.info(
        "Consolidated RF pipeline completed end-to-end. Artifacts in %s", out_dir
    )
    return manifest


if __name__ == "__main__":
    run()
