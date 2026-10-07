"""Run the Day 19 feature-engineering ablation study.

Day 19 deliverable: does feature engineering help predict the thermal
plant's next temperature, and if so, which part of it -- the inputs or what
the model is asked to predict?

This is deliberately a *study*, not a single "best model" script: every
result below is a controlled ablation against a stated reference, because
the Sep-27 submission's single R2 = 0.9997 number was high only because
predicting "nothing changed" already scores R2 = 0.998 on this data.

Reads config.yaml + feature_study.yaml (see config.py). Writes, under
``outputs/feature_study/``:
    manifest.json              config fingerprint, package versions, row counts
    feature_ablation.csv       CV MSE per (model, feature_set), target held fixed
    target_ablation.csv        CV MSE per (model, target_mode), features held fixed
    safety_agreement.csv       missed/false-alarm rates on the dedicated stress test
    redundancy_report.json     feature pairs flagged by find_redundant_pairs
    parity_<model>.png         predicted vs. actual, best configuration per model

Run:  PYTHONPATH=. python -m experiments.baseline_dynamics_learning.run_feature_study
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import wandb
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from experiments.baseline_dynamics_learning.config import (
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
from experiments.baseline_dynamics_learning.estimators import make_model
from experiments.baseline_dynamics_learning.features import (
    build_features,
    find_redundant_pairs,
    resolve_feature_set,
)
from experiments.baseline_dynamics_learning.metrics import (
    oracle_prediction,
    persistence_prediction,
    physics_prediction,
    regression_scores,
    safety_agreement_metrics,
)
from src.plant import PlantConfig

# Headless backend: scripts here only save figures, never open a window.
plt.switch_backend("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _cv_score(
    df: pd.DataFrame,
    cols: list[str],
    model_name: str,
    target_mode: str,
    cfg: dict[str, Any],
    plant: PlantConfig,
    folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
) -> dict[str, float]:
    """Return mean and std of ``regression_scores`` across the CV folds.

    One (feature columns, target mode) pair is scored per call.
    """
    per_fold = []
    for train_idx, test_idx in folds:
        model = make_model(
            model_name, target_mode, plant, cfg["models"], cfg["random_seed"]
        )
        model.fit(df.iloc[train_idx][cols], df.iloc[train_idx][TARGET_COLUMN])
        y_pred = model.predict(df.iloc[test_idx][cols])
        te = df.iloc[test_idx]
        per_fold.append(
            regression_scores(
                te[TARGET_COLUMN].to_numpy(),
                y_pred,
                te["T_current"].to_numpy(),
                persistence_prediction(te["T_current"].to_numpy()),
                physics_prediction(
                    te["T_current"].to_numpy(),
                    te["u"].to_numpy(),
                    plant.dt,
                    plant.mCp,
                    plant.k_loss,
                    plant.T_amb,
                ),
            )
        )
    keys = per_fold[0].keys()
    out = {f"{k}_mean": float(np.mean([f[k] for f in per_fold])) for k in keys}
    out.update({f"{k}_std": float(np.std([f[k] for f in per_fold])) for k in keys})
    return out


def run(
    config_path: str | Path | None = None,
    overrides_path: str | Path | None = None,
    extra_overrides: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the feature-engineering ablation study.

    Parameters
    ----------
    config_path : str or Path or None
        Base config YAML; defaults to ``DEFAULT_BASE``.
    overrides_path : str or Path or None
        Overrides YAML; defaults to ``DEFAULT_OVERRIDES``.
    extra_overrides : dict or None
        In-memory config overrides applied last.
    cfg : dict or None
        A fully resolved config; if given, no file is loaded.

    Returns
    -------
    dict[str, pandas.DataFrame]
        The result tables of the study, keyed by name.
    """
    from experiments.baseline_dynamics_learning.config import (
        DEFAULT_BASE,
        DEFAULT_OVERRIDES,
    )

    if cfg is None:
        cfg = load_config(
            config_path or DEFAULT_BASE,
            overrides_path or DEFAULT_OVERRIDES,
            extra_overrides=extra_overrides,
        )
    fs_cfg = cfg["feature_study"]
    plant = plant_from_cfg(cfg)
    gen_cfg = datagen_from_cfg(cfg)

    out_dir = Path("outputs/feature_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating episodic dataset (rho=%.2f) and building features...",
        gen_cfg.disturbance_autocorr,
    )
    raw_df = generate_dataset(
        plant, gen_cfg, output_path=str(out_dir / "thermal_dataset_raw.csv")
    )
    df = build_features(raw_df, plant)
    df.to_csv(out_dir / "thermal_dataset_features.csv", index=False)
    logger.info("Rows after warm-up trim: %d (from %d)", len(df), len(raw_df))

    train_df, test_df = episode_train_test_split(
        df, n_test_episodes=cfg["data"]["n_test_episodes"], seed=cfg["random_seed"]
    )
    folds = list(
        GroupKFold(n_splits=fs_cfg["n_folds"]).split(
            train_df, groups=train_df["episode_id"]
        )
    )

    # ---- redundancy audit -------------------------------------------------
    full_cols = resolve_feature_set(fs_cfg["target_ablation_feature_set"])
    redundant = find_redundant_pairs(
        df[full_cols], threshold=fs_cfg["redundancy_threshold"]
    )
    with open(out_dir / "redundancy_report.json", "w") as f:
        json.dump(
            {"threshold": fs_cfg["redundancy_threshold"], "pairs": redundant},
            f,
            indent=2,
        )
    if redundant:
        logger.warning(
            "Redundant feature pairs (|corr| >= %.3f): %s",
            fs_cfg["redundancy_threshold"],
            redundant,
        )
    else:
        logger.info("No redundant feature pairs found among %s.", full_cols)

    run_ctx = wandb.init(
        project=cfg["ml_pipeline"]["wandb_project"],
        name="feature_study",
        config=cfg,
        mode=cfg["ml_pipeline"]["wandb_mode"],
        reinit="finish_previous",
    )

    # ---- feature ablation: target fixed, inputs vary -----------------------
    logger.info("Feature ablation (target_mode=%s)...", fs_cfg["ablation_target_mode"])
    rows = []
    for model_name in fs_cfg["models"]:
        for feat_set in fs_cfg["feature_sets"]:
            cols = resolve_feature_set(feat_set)
            scores = _cv_score(
                train_df,
                cols,
                model_name,
                fs_cfg["ablation_target_mode"],
                cfg,
                plant,
                folds,
            )
            rows.append({"model": model_name, "feature_set": feat_set, **scores})
    feature_ablation = pd.DataFrame(rows)
    feature_ablation.to_csv(out_dir / "feature_ablation.csv", index=False)
    wandb.log({"feature_ablation": wandb.Table(dataframe=feature_ablation)})

    # ---- target ablation: features fixed, target formulation varies --------
    logger.info(
        "Target ablation (feature_set=%s)...", fs_cfg["target_ablation_feature_set"]
    )
    rows = []
    for model_name in fs_cfg["models"]:
        for target_mode in fs_cfg["target_modes"]:
            scores = _cv_score(
                train_df, full_cols, model_name, target_mode, cfg, plant, folds
            )
            rows.append({"model": model_name, "target_mode": target_mode, **scores})
    target_ablation = pd.DataFrame(rows)
    target_ablation.to_csv(out_dir / "target_ablation.csv", index=False)
    wandb.log({"target_ablation": wandb.Table(dataframe=target_ablation)})

    # ---- reference floors (no learning) on the SAME held-out test set ------
    te = test_df
    pers_mse = float(
        np.mean((te[TARGET_COLUMN] - persistence_prediction(te["T_current"])) ** 2)
    )
    phys_pred = physics_prediction(
        te["T_current"], te["u"], plant.dt, plant.mCp, plant.k_loss, plant.T_amb
    )
    phys_mse = float(np.mean((te[TARGET_COLUMN] - phys_pred) ** 2))
    d_prev = te.groupby("episode_id")["debug_d"].shift(1).fillna(0.0)
    oracle_pred = oracle_prediction(
        phys_pred, d_prev, plant.dt, plant.mCp, gen_cfg.disturbance_autocorr
    )
    oracle_mse = float(np.mean((te[TARGET_COLUMN] - oracle_pred) ** 2))
    reference = pd.DataFrame(
        [
            {"reference": "persistence (no model)", "MSE": pers_mse},
            {"reference": "physics, d=0 (no model)", "MSE": phys_mse},
            {
                "reference": (
                    "oracle disturbance observer (uses true d; NOT deployable)"
                ),
                "MSE": oracle_mse,
            },
        ]
    )
    reference.to_csv(out_dir / "reference_floors.csv", index=False)
    logger.info("\n%s", reference.to_string(index=False))

    # ---- safety agreement: sustained-high-actuation stress test ------------
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
    rows = []
    for name, recipe in fs_cfg["safety_recipes"].items():
        cols = resolve_feature_set(recipe["feature_set"])
        model = make_model(
            fs_cfg["models"][0],
            recipe["target_mode"],
            plant,
            cfg["models"],
            cfg["random_seed"],
        )
        model.fit(train_df[cols], train_df[TARGET_COLUMN])
        y_pred = model.predict(stress_df[cols])
        m = safety_agreement_metrics(
            stress_df[TARGET_COLUMN].to_numpy(),
            y_pred,
            cfg["physics"]["T_min"],
            cfg["physics"]["T_max"],
        )
        rows.append({"recipe": name, "model": fs_cfg["models"][0], **recipe, **m})
    safety = pd.DataFrame(rows)
    safety.to_csv(out_dir / "safety_agreement.csv", index=False)
    wandb.log({"safety_agreement": wandb.Table(dataframe=safety)})
    logger.info("\n%s", safety.to_string(index=False))

    # ---- one parity plot per model, best (feature_set, target_mode) found --
    best_per_model = feature_ablation.assign(
        target_mode=fs_cfg["ablation_target_mode"]
    ).loc[feature_ablation.groupby("model")["MSE_mean"].idxmin()]
    for _, row in best_per_model.iterrows():
        cols = resolve_feature_set(row["feature_set"])
        model = make_model(
            row["model"], row["target_mode"], plant, cfg["models"], cfg["random_seed"]
        )
        model.fit(train_df[cols], train_df[TARGET_COLUMN])
        y_pred = model.predict(test_df[cols])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(test_df[TARGET_COLUMN], y_pred, s=6, alpha=0.4)
        lims = [test_df[TARGET_COLUMN].min(), test_df[TARGET_COLUMN].max()]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual T_next")
        ax.set_ylabel("Predicted T_next")
        ax.set_title(f"{row['model']} | {row['feature_set']} + {row['target_mode']}")
        fig.tight_layout()
        path = out_dir / f"parity_{row['model']}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        wandb.log({f"parity_{row['model']}": wandb.Image(str(path))})

    manifest = {
        "config_fingerprint": config_fingerprint(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "n_rows_raw": len(raw_df),
        "n_rows_after_warmup": len(df),
        "n_train_episodes": int(train_df["episode_id"].nunique()),
        "n_test_episodes": int(test_df["episode_id"].nunique()),
        "redundant_feature_pairs": redundant,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    run_ctx.finish()

    logger.info(
        "\n%s\nFEATURE ABLATION (target=%s), best row per model\n%s",
        "=" * 70,
        fs_cfg["ablation_target_mode"],
        "=" * 70,
    )
    logger.info(
        "\n%s",
        feature_ablation.loc[
            feature_ablation.groupby("model")["MSE_mean"].idxmin(),
            ["model", "feature_set", "MSE_mean", "MSE_std", "Skill_vs_physics_mean"],
        ].to_string(index=False),
    )
    logger.info(
        "\n%s\nTARGET ABLATION (features=%s), best row per model\n%s",
        "=" * 70,
        fs_cfg["target_ablation_feature_set"],
        "=" * 70,
    )
    logger.info(
        "\n%s",
        target_ablation.loc[
            target_ablation.groupby("model")["MSE_mean"].idxmin(),
            ["model", "target_mode", "MSE_mean", "MSE_std"],
        ].to_string(index=False),
    )
    return {
        "feature_ablation": feature_ablation,
        "target_ablation": target_ablation,
        "reference_floors": reference,
        "safety_agreement": safety,
    }


if __name__ == "__main__":
    run()
