"""Train classical ML baselines as thermal dynamics models.

Trains and compares classical ML models (Linear Regression, Random
Forest, SVR) as dynamics-model baselines for the thermal plant defined
in ``src/plant.py``.

Role in the thesis defense line
--------------------------------
This experiment does NOT compete with the thesis's RL contribution
(SAC + physics-informed reward + CBF-QP safety layer) -- it MOTIVATES
it. It answers a question the defense needs answered before the RL
results mean anything: *if you throw generic, physics-agnostic ML at
this plant's dynamics, how often does it predict a physically unsafe
or inconsistent state?* The ``Constraint_Violation_Rate`` and
``Energy_Residual`` metrics logged here are exactly what Paper 2's
CBF-QP safety layer is designed to eliminate -- this experiment
produces the "before" numbers that the RL policy's "after" numbers
get compared against in the defense.

It is also the natural first user of ``ActuatedThermalPlant``
(``src/plant.py``), ahead of ``thermal_env.py`` -- both this script
and the future Gym environment train/act on identical physics, so
nothing here needs to be redone when the RL environment is built.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from experiments.baseline_dynamics_learning.data_generation import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    TARGET_COLUMN,
    DataGenConfig,
    episode_train_test_split,
    generate_dataset,
    generate_stress_test_episodes,
)
from experiments.baseline_dynamics_learning.metrics import physics_informed_metrics
from experiments.baseline_dynamics_learning.utils import set_seed
from src.plant import PlantConfig

# Headless backend: scripts here only save figures, never open a window.
plt.switch_backend("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.
    """
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def build_model(name: str, cfg: dict[str, Any]) -> Any:
    """Build an unfitted sklearn regressor from its report name.

    Parameters
    ----------
    name : str
        One of ``"Linear_Regression"``, ``"Random_Forest"``, ``"SVM"``.
    cfg : dict[str, Any]
        Config with ``models`` hyper-parameters and ``random_seed``.

    Returns
    -------
    Any
        The unfitted estimator.
    """
    if name == "Linear_Regression":
        return LinearRegression()
    if name == "Random_Forest":
        return RandomForestRegressor(
            n_estimators=cfg["models"]["random_forest"]["n_estimators"],
            max_depth=cfg["models"]["random_forest"]["max_depth"],
            random_state=cfg["random_seed"],
        )
    if name == "SVM":
        return SVR(C=cfg["models"]["svm"]["C"], epsilon=cfg["models"]["svm"]["epsilon"])
    raise ValueError(f"Unknown model name: {name}")


def get_param_grid(name: str, cfg: dict[str, Any]) -> dict[str, list[Any]] | None:
    """Return the grid-search parameter grid for a model, if it has one.

    Parameters
    ----------
    name : str
        Model report name.
    cfg : dict[str, Any]
        Config holding ``models.<key>.param_grid``.

    Returns
    -------
    dict[str, list[Any]] or None
        Grid keyed ``model__<param>`` (pipeline step prefix), or ``None`` for
        models without a grid.
    """
    key = {"Random_Forest": "random_forest", "SVM": "svm"}.get(name)
    if key is None:
        return None
    grid = cfg["models"][key].get("param_grid")
    if grid is None:
        return None
    return {f"model__{k}": v for k, v in grid.items()}


def make_parity_plot(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    model_name: str,
    out_path: Path,
) -> Path:
    """Save a one-step-ahead parity plot of predicted vs. actual ``T_next``.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Actual and predicted next temperature (deg C).
    model_name : str
        Name shown in the plot title.
    out_path : Path
        Destination PNG path.

    Returns
    -------
    Path
        ``out_path``.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    lims = [
        float(min(np.min(y_true), np.min(y_pred))),
        float(max(np.max(y_true), np.max(y_pred))),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual T_next (deg C)")
    ax.set_ylabel("Predicted T_next (deg C)")
    ax.set_title(f"{model_name}: predicted vs. actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(config_path: str = "config.yaml") -> pd.DataFrame:
    """Train and compare all baseline models.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.

    Returns
    -------
    pandas.DataFrame
        One row of metrics per model, also written to ``outputs/``.
    """
    cfg = load_config(config_path)
    set_seed(cfg["random_seed"])

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # 1. Generate physics-based episodic data via the actuated plant.
    logger.info("Generating episodic thermal dataset via ActuatedThermalPlant...")
    plant_cfg = PlantConfig(
        mCp=cfg["physics"]["mCp"],
        k_loss=cfg["physics"]["k_loss"],
        T_amb=cfg["physics"]["T_amb"],
        dt=cfg["physics"]["dt"],
    )
    gen_cfg = DataGenConfig(
        n_episodes=cfg["data"]["n_episodes"],
        steps_per_episode=cfg["data"]["steps_per_episode"],
        u_max=cfg["data"]["u_max"],
        disturbance_std=cfg["data"]["disturbance_std"],
        base_seed=cfg["random_seed"],
    )
    dataset_path = str(out_dir / "thermal_dataset.csv")
    df = generate_dataset(plant_cfg, gen_cfg, output_path=dataset_path)

    # 2. Episode-level split -- see data_generation.episode_train_test_split
    #    for why this replaces the v1 submission's row-level shuffled split.
    train_df, test_df = episode_train_test_split(
        df, n_test_episodes=cfg["data"]["n_test_episodes"], seed=cfg["random_seed"]
    )
    logger.info(
        "Train: %d episodes / %d rows | Test: %d episodes / %d rows",
        train_df[GROUP_COLUMN].nunique(),
        len(train_df),
        test_df[GROUP_COLUMN].nunique(),
        len(test_df),
    )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]
    groups_train = train_df[GROUP_COLUMN]

    # Targeted safety stress-test set -- sustained near-max actuation,
    # held out of BOTH training and the nominal test set. See
    # generate_stress_test_episodes() docstring for why nominal
    # random-excitation data alone makes Constraint_Violation_Rate
    # uninformative (it rarely visits the unsafe region).
    stress_df = generate_stress_test_episodes(
        plant_cfg,
        n_episodes=cfg["data"]["n_test_episodes"],
        duration_steps=cfg["data"]["steps_per_episode"],
        u_level=cfg["data"]["u_max"],
        disturbance_std=cfg["data"]["disturbance_std"],
        base_seed=cfg["random_seed"],
    )
    X_stress, y_stress = stress_df[FEATURE_COLUMNS], stress_df[TARGET_COLUMN]
    logger.info(
        "Stress test: %d episodes / %d rows, %.1f%% of GROUND-TRUTH labels "
        "already outside [%s, %s]",
        stress_df[GROUP_COLUMN].nunique(),
        len(stress_df),
        (
            y_stress.gt(cfg["physics"]["T_max"]) | y_stress.lt(cfg["physics"]["T_min"])
        ).mean()
        * 100,
        cfg["physics"]["T_min"],
        cfg["physics"]["T_max"],
    )

    wandb_mode = cfg["ml_pipeline"]["wandb_mode"]
    results = []

    for model_name in ["Linear_Regression", "Random_Forest", "SVM"]:
        logger.info("Evaluating %s...", model_name)

        run_ctx = wandb.init(
            project=cfg["ml_pipeline"]["wandb_project"],
            name=model_name,
            config=cfg,
            mode=wandb_mode,
            reinit="finish_previous",
        )

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", build_model(model_name, cfg))]
        )

        param_grid = (
            get_param_grid(model_name, cfg)
            if cfg["ml_pipeline"]["tune_hyperparameters"]
            else None
        )
        if param_grid:
            # GroupKFold: CV folds never split an episode across train/val
            # either, keeping the leakage fix consistent through tuning.
            n_groups = groups_train.nunique()
            n_splits = min(cfg["ml_pipeline"]["cv_folds"], n_groups)
            cv = GroupKFold(n_splits=n_splits)
            search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring="neg_mean_squared_error"
            )
            search.fit(X_train, y_train, groups=groups_train)
            pipeline = search.best_estimator_
            # wandb ships no type stubs, hence the targeted ignore.
            wandb.config.update(  # type: ignore[no-untyped-call]
                {f"best_{model_name}_params": search.best_params_}
            )
            logger.info("%s best params: %s", model_name, search.best_params_)
        else:
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        metrics = physics_informed_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            T_current=X_test["T_current"].values,
            u=X_test["u"].values,
            dt=cfg["physics"]["dt"],
            mCp=cfg["physics"]["mCp"],
            k_loss=cfg["physics"]["k_loss"],
            T_amb=cfg["physics"]["T_amb"],
            t_min=cfg["physics"]["T_min"],
            t_max=cfg["physics"]["T_max"],
        )

        # Safety stress-test evaluation -- same fitted pipeline,
        # sustained-high-actuation data. This is the number that
        # matters for the "why we need the CBF-QP layer" defense
        # argument, not the nominal-test CVR above.
        y_pred_stress = pipeline.predict(X_stress)
        stress_violations = (y_pred_stress < cfg["physics"]["T_min"]) | (
            y_pred_stress > cfg["physics"]["T_max"]
        )
        metrics["Stress_Test_CVR_pct"] = float(stress_violations.mean() * 100.0)

        wandb.log(metrics)

        plot_path = make_parity_plot(
            y_test.values, y_pred, model_name, out_dir / f"{model_name}_parity.png"
        )
        wandb.log({"parity_plot": wandb.Image(str(plot_path))})

        model_path = out_dir / f"{model_name}.pkl"
        joblib.dump(pipeline, model_path)
        model_artifact = wandb.Artifact(f"{model_name}_model", type="model")
        model_artifact.add_file(str(model_path))
        run_ctx.log_artifact(model_artifact)

        metrics["Model"] = model_name
        results.append(metrics)
        run_ctx.finish()

    results_df = pd.DataFrame(results)[
        [
            "Model",
            "MSE",
            "MAE",
            "R2_Score",
            "Constraint_Violation_Rate_pct",
            "Stress_Test_CVR_pct",
            "Energy_Residual_Mean",
            "Energy_Residual_Std",
        ]
    ]
    results_df.to_csv(out_dir / "results_summary.csv", index=False)

    logger.info(
        "\n%s\n RIGOROUS MODEL COMPARISON "
        "(episode-level holdout, physics-informed)\n%s",
        "=" * 70,
        "=" * 70,
    )
    logger.info("\n%s", results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()
    run(args.config)
