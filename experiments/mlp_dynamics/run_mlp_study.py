"""MLP vs Random Forest vs Linear on the thermal dynamics (Days 43 and 44).

Day 43: an MLP (``src/nn/mlp.py``) trained on the thermal data.
Day 44: the comparison table (R2, MAE, MSE ...) logged to W&B.

Run::

    PYTHONPATH=. python -m experiments.mlp_dynamics.run_mlp_study

What the comparison guards against (the recurring bug patterns of this repo):

1. ``d`` is never a feature (``resolve_feature_set`` enforces it).
2. Whole-episode splits everywhere: test, validation, CV and Optuna folds.
3. Safety is measured on a separate stress set, not on nominal data.
4. The target is standardised for every model (``make_model``).
5. Features are the audited sets from Day 19.
6. **No headline number without its no-learning reference.** ``T_next`` is
   almost ``T_current``, so *predicting no change* already scores
   R2 ~ 0.998. The table therefore always carries the persistence and
   physics references, ``R2_delta`` and the skill scores, plus the theoretical
   noise floor, so a reader can see what each model adds.

Fairness: the MLP is tuned with Optuna on the TRAIN episodes, so the Random
Forest gets the same treatment (a grouped grid over its configured
``param_grid``). The Linear model has nothing to tune. The test set is used
once, after all tuning.
"""

from __future__ import annotations

import itertools
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from experiments.baseline_dynamics_learning.config import (
    config_fingerprint,
)
from experiments.baseline_dynamics_learning.data_generation import (
    GROUP_COLUMN,
    TARGET_COLUMN,
)
from experiments.baseline_dynamics_learning.estimators import make_model
from experiments.baseline_dynamics_learning.metrics import (
    persistence_prediction,
    physics_prediction,
    regression_scores,
    safety_agreement_metrics,
)
from experiments.mlp_dynamics import wandb_utils
from experiments.mlp_dynamics.data import (
    StudyData,
    build_study_data,
    feature_columns,
    load_study_config,
    split_validation,
)
from experiments.mlp_dynamics.tuning import baseline_for, tune_mlp
from src.nn.mlp import TrainConfig, train_mlp

plt.switch_backend("Agg")  # headless: figures are only saved

logger = logging.getLogger(__name__)

SCORE_COLUMNS = [
    "R2_T_next",
    "R2_delta",
    "MAE",
    "MSE",
    "Skill_vs_persistence",
    "Skill_vs_physics",
    "stress_missed_violation_pct",
]


def noise_floor_mse(cfg: dict[str, Any]) -> float:
    """Return the MSE no deployable model can beat for the configured load.

    For an AR(1) load the best guess of the next load is ``rho * d_prev``;
    even a model that knew ``d_prev`` exactly keeps the innovation:
    ``(dt / mCp)^2 * (1 - rho^2) * std^2``.
    """
    ph, da = cfg["physics"], cfg["data"]
    rho, std = da.get("disturbance_autocorr", 0.0), da["disturbance_std"]
    return float((ph["dt"] / ph["mCp"]) ** 2 * (1.0 - rho**2) * std**2)


def longest_decreasing_streak(values: list[float]) -> int:
    """Return the longest run of consecutive epochs whose loss strictly decreased."""
    best = run = 0
    for prev, cur in zip(values[:-1], values[1:], strict=True):
        run = run + 1 if cur < prev else 0
        best = max(best, run)
    return best


def moving_average(values: list[float], window: int) -> list[float]:
    """Return the trailing moving average (shorter windows at the start)."""
    return [
        float(np.mean(values[max(0, i - window + 1) : i + 1]))
        for i in range(len(values))
    ]


def tune_random_forest(
    data: StudyData, recipe: dict[str, str], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Grouped-CV grid over the Random Forest's configured ``param_grid``."""
    grid = cfg["models"]["random_forest"]["param_grid"]
    cols = feature_columns(recipe["feature_set"])
    train = data.train_df
    folds = list(
        GroupKFold(n_splits=cfg["optuna"]["n_folds"]).split(
            train, groups=train[GROUP_COLUMN]
        )
    )
    best: tuple[float, dict[str, Any]] | None = None
    for n_est, depth in itertools.product(grid["n_estimators"], grid["max_depth"]):
        models_cfg = {"random_forest": {"n_estimators": n_est, "max_depth": depth}}
        mses = []
        for tr, va in folds:
            m = make_model(
                "random_forest",
                recipe["target_mode"],
                data.plant,
                models_cfg,
                cfg["random_seed"],
            )
            m.fit(train.iloc[tr][cols], train.iloc[tr][TARGET_COLUMN])
            pred = m.predict(train.iloc[va][cols])
            mses.append(float(np.mean((pred - train.iloc[va][TARGET_COLUMN]) ** 2)))
        score = float(np.mean(mses))
        if best is None or score < best[0]:
            best = (score, models_cfg["random_forest"])
    assert best is not None
    return best[1]


def _evaluate(
    model: Any, data: StudyData, cols: list[str], cfg: dict[str, Any]
) -> dict[str, float]:
    """Score a fitted model on the held-out test episodes and the stress set."""
    p, test = data.plant, data.test_df
    pred = model.predict(test[cols])
    scores = regression_scores(
        test[TARGET_COLUMN].to_numpy(),
        pred,
        test["T_current"].to_numpy(),
        persistence_prediction(test["T_current"].to_numpy()),
        physics_prediction(
            test["T_current"].to_numpy(),
            test["u"].to_numpy(),
            p.dt,
            p.mCp,
            p.k_loss,
            p.T_amb,
        ),
    )
    stress = data.stress_df
    safety = safety_agreement_metrics(
        stress[TARGET_COLUMN].to_numpy(),
        model.predict(stress[cols]),
        cfg["physics"]["T_min"],
        cfg["physics"]["T_max"],
    )
    scores["stress_missed_violation_pct"] = safety["missed_violation_pct"]
    return scores


def _reference_rows(data: StudyData, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """No-learning references, scored exactly like the models."""
    p, test, stress = data.plant, data.test_df, data.stress_df
    T, u = test["T_current"].to_numpy(), test["u"].to_numpy()

    def persist(t: NDArray[np.float64], uu: NDArray[np.float64]) -> NDArray[np.float64]:
        return persistence_prediction(t)

    def physics(t: NDArray[np.float64], uu: NDArray[np.float64]) -> NDArray[np.float64]:
        return physics_prediction(t, uu, p.dt, p.mCp, p.k_loss, p.T_amb)

    rows = []
    for name, fn in (
        ("persistence (no learning)", persist),
        ("physics, d = 0 (no learning)", physics),
    ):
        pred = fn(T, u)
        s = regression_scores(
            test[TARGET_COLUMN].to_numpy(),
            pred,
            T,
            persistence_prediction(T),
            physics_prediction(T, u, p.dt, p.mCp, p.k_loss, p.T_amb),
        )
        safety = safety_agreement_metrics(
            stress[TARGET_COLUMN].to_numpy(),
            fn(stress["T_current"].to_numpy(), stress["u"].to_numpy()),
            cfg["physics"]["T_min"],
            cfg["physics"]["T_max"],
        )
        s["stress_missed_violation_pct"] = safety["missed_violation_pct"]
        rows.append(
            {
                "model": name,
                "recipe": "-",
                "seeds": 1,
                **s,
                **{f"{k}_std": 0.0 for k in SCORE_COLUMNS},
            }
        )
    return rows


def _plot_curve(train_loss: list[float], val_loss: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(1, len(train_loss) + 1), train_loss, label="train")
    ax.plot(range(1, len(val_loss) + 1), val_loss, label="validation (whole episodes)")
    ax.set(
        xlabel="epoch", ylabel="MSE (standardised target)", title="MLP training curve"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_markdown(table: pd.DataFrame) -> str:
    """Render the comparison table as GitHub markdown (mean, with std for the MLP)."""

    def cell(row: pd.Series, col: str) -> str:
        v, sd = row[col], row.get(f"{col}_std", 0.0)
        if pd.isna(v):
            return "n/a"
        return f"{v:.5g}" if not sd else f"{v:.5g} ± {sd:.2g}"

    head = ["model", "recipe", "seeds", *SCORE_COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["model"]),
                    str(r["recipe"]),
                    str(r["seeds"]),
                    *[cell(r, c) for c in SCORE_COLUMNS],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run(
    cfg: dict[str, Any] | None = None,
    out_dir: str | Path = "outputs/mlp_study",
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Run the whole study and return the manifest.

    Parameters
    ----------
    cfg : dict, optional
        Resolved config; defaults to :func:`load_study_config`.
    out_dir : str or Path
        Where tables, figures, checkpoints and the manifest go.
    n_trials : int, optional
        Overrides ``optuna.n_trials`` (used by the tests).
    """
    cfg = cfg or load_study_config()
    if n_trials is not None:
        cfg = {**cfg, "optuna": {**cfg["optuna"], "n_trials": n_trials}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = cfg["mlp_study"]
    seed = cfg["random_seed"]

    data = build_study_data(cfg, out)
    run_ctx = wandb_utils.start_run(
        cfg, "mlp_vs_rf_study", {"fingerprint": config_fingerprint(cfg)}
    )

    # ---- Day 37: the training curve (train vs whole-episode validation) ----
    best_recipe = study["recipes"]["best"]
    best_cols = feature_columns(best_recipe["feature_set"])
    mlp_cfgs: dict[str, dict[str, Any]] = {}
    tuning_info: dict[str, Any] = {}
    for rname, recipe in study["recipes"].items():
        cols = feature_columns(recipe["feature_set"])

        def on_trial(
            num: int, val: float, params: dict[str, Any], r: str = rname
        ) -> None:
            wandb_utils.safe_log(
                run_ctx, {f"optuna/{r}/trial_value": val, f"optuna/{r}/trial": num}
            )

        res = tune_mlp(
            data.train_df,
            cols,
            recipe["target_mode"],
            data.plant,
            cfg["optuna"],
            on_trial,
        )
        mlp_cfgs[rname] = res.model_cfg()
        res.trials.to_csv(out / f"optuna_trials_{rname}.csv", index=False)
        tuning_info[rname] = {
            "best_cv_val_loss": res.best_value,
            "mlp_cfg": res.model_cfg(),
        }

    curve_cfg = mlp_cfgs["best"]
    tr_df, va_df = split_validation(data.train_df, study["n_val_episodes"], seed)
    base_tr = baseline_for(tr_df, best_recipe["target_mode"], data.plant)
    base_va = baseline_for(va_df, best_recipe["target_mode"], data.plant)
    _, hist = train_mlp(
        tr_df[best_cols].to_numpy(dtype=float),
        tr_df[TARGET_COLUMN].to_numpy(dtype=float) - base_tr,
        TrainConfig(
            hidden=tuple(curve_cfg["hidden"]),
            activation=curve_cfg["activation"],
            lr=curve_cfg["lr"],
            weight_decay=curve_cfg["weight_decay"],
            batch_size=curve_cfg["batch_size"],
            max_epochs=study["curve_epochs"],
            seed=seed,
        ),
        va_df[best_cols].to_numpy(dtype=float),
        va_df[TARGET_COLUMN].to_numpy(dtype=float) - base_va,
        on_epoch=wandb_utils.epoch_logger(run_ctx),
    )
    _plot_curve(hist.train_loss, hist.val_loss, out / "training_curve.png")
    curve = {
        "epochs": len(hist.train_loss),
        "first_train_loss": hist.train_loss[0],
        "last_train_loss": hist.train_loss[-1],
        "best_val_loss": min(hist.val_loss),
        "best_val_epoch": hist.best_epoch,
        "longest_decreasing_train_streak": longest_decreasing_streak(hist.train_loss),
        "longest_decreasing_val_streak": longest_decreasing_streak(hist.val_loss),
        # Mini-batch loss is noisy once it nears the plateau, so the strict
        # per-epoch streak is short. The 5-epoch moving average is reported
        # next to it (and labelled as smoothed) rather than hiding either.
        "longest_decreasing_train_streak_ma5": longest_decreasing_streak(
            moving_average(hist.train_loss, 5)
        ),
        "longest_decreasing_val_streak_ma5": longest_decreasing_streak(
            moving_average(hist.val_loss, 5)
        ),
    }

    # ---- Day 43/44: final models, all on the same recipe grid ----
    rows = _reference_rows(data, cfg)
    floor = noise_floor_mse(cfg)
    rf_cfgs = {
        r: tune_random_forest(data, rec, cfg) for r, rec in study["recipes"].items()
    }
    checkpoint: Any = None
    for rname, recipe in study["recipes"].items():
        cols = feature_columns(recipe["feature_set"])
        for mname in study["models"]:
            seeds = study["mlp_seeds"] if mname == "mlp" else [seed]
            per_seed = []
            for s in seeds:
                models_cfg = {
                    "random_forest": rf_cfgs[rname],
                    "mlp": mlp_cfgs[rname],
                    "svm": cfg["models"].get("svm", {}),
                }
                model = make_model(
                    mname, recipe["target_mode"], data.plant, models_cfg, s
                )
                model.fit(data.train_df[cols], data.train_df[TARGET_COLUMN])
                per_seed.append(_evaluate(model, data, cols, cfg))
                if mname == "mlp" and rname == "best" and s == seeds[0]:
                    checkpoint = model
            frame = pd.DataFrame(per_seed)
            row: dict[str, Any] = {"model": mname, "recipe": rname, "seeds": len(seeds)}
            for c in SCORE_COLUMNS:
                row[c] = float(frame[c].mean())
                row[f"{c}_std"] = float(frame[c].std(ddof=0))
            rows.append(row)
            logger.info(
                "%s/%s: MSE %.5f  R2_delta %.4f",
                rname,
                mname,
                row["MSE"],
                row["R2_delta"],
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    md = (
        to_markdown(table)
        + f"\n\nNoise floor MSE (best any model can reach): {floor:.5g}\n"
    )
    (out / "comparison.md").write_text(md, encoding="utf-8")

    # ---- Day 39: everything to W&B (a no-op transport when offline/disabled) ----
    wandb_utils.safe_log(run_ctx, {f"curve/{k}": v for k, v in curve.items()})
    wandb_utils.safe_log(run_ctx, {"noise_floor_mse": floor})
    try:
        import wandb

        wandb_utils.safe_log(
            run_ctx,
            {
                "comparison": wandb.Table(
                    dataframe=table[["model", "recipe", "seeds", *SCORE_COLUMNS]]
                )
            },
        )
        wandb_utils.safe_log(
            run_ctx, {"training_curve": wandb.Image(str(out / "training_curve.png"))}
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not log table/image to W&B: %s", exc)

    if checkpoint is not None:
        joblib.dump(checkpoint, out / "mlp_best.pkl")

    manifest = {
        "config_fingerprint": config_fingerprint(cfg),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "optuna": optuna.__version__,
        "sklearn": sklearn.__version__,
        "wandb_mode": wandb_utils.resolve_mode(cfg["ml_pipeline"]["wandb_mode"]),
        "seed": seed,
        "mlp_seeds": study["mlp_seeds"],
        "n_train_episodes": int(data.train_df[GROUP_COLUMN].nunique()),
        "n_test_episodes": int(data.test_df[GROUP_COLUMN].nunique()),
        "noise_floor_mse": floor,
        "tuning": tuning_info,
        "rf_tuned": rf_cfgs,
        "training_curve": curve,
        "comparison": table[["model", "recipe", "seeds", *SCORE_COLUMNS]].to_dict(
            orient="records"
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )
    wandb_utils.finish(run_ctx)
    logger.info("MLP study finished; artifacts in %s", out)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run()
