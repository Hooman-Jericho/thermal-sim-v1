"""Build the train / validation / test / stress data for the MLP study.

Everything here reuses ``baseline_dynamics_learning`` (data generation,
features, split). Nothing is re-implemented, so the MLP is trained and judged
on exactly the data the Random Forest of Day 20 saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiments.baseline_dynamics_learning.config import (
    DEFAULT_BASE,
    DEFAULT_OVERRIDES,
    datagen_from_cfg,
    load_config,
    plant_from_cfg,
)
from experiments.baseline_dynamics_learning.data_generation import (
    GROUP_COLUMN,
    TARGET_COLUMN,
    DataGenConfig,
    episode_train_test_split,
    generate_dataset,
    generate_stress_test_episodes,
)
from experiments.baseline_dynamics_learning.features import (
    build_features,
    resolve_feature_set,
)
from src.plant import PlantConfig

MLP_STUDY_YAML = Path(__file__).resolve().parent / "mlp_study.yaml"


def load_study_config(extra_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load base + feature-study + MLP-study YAML (+ optional in-memory overrides)."""
    with open(MLP_STUDY_YAML, encoding="utf-8") as f:
        mlp_cfg = yaml.safe_load(f)
    from experiments.baseline_dynamics_learning.config import deep_merge

    merged = deep_merge(mlp_cfg, extra_overrides or {})
    return load_config(DEFAULT_BASE, DEFAULT_OVERRIDES, extra_overrides=merged)


@dataclass
class StudyData:
    """Engineered, episode-split data shared by every model in the study."""

    cfg: dict[str, Any]
    plant: PlantConfig
    gen_cfg: DataGenConfig
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    stress_df: pd.DataFrame


def build_study_data(cfg: dict[str, Any], out_dir: Path | str) -> StudyData:
    """Generate the episodes, engineer features, split by episode, add a stress set."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plant, gen_cfg = plant_from_cfg(cfg), datagen_from_cfg(cfg)
    raw = generate_dataset(plant, gen_cfg, output_path=str(out / "thermal_raw.csv"))
    df = build_features(raw, plant)
    train_df, test_df = episode_train_test_split(
        df, n_test_episodes=cfg["data"]["n_test_episodes"], seed=cfg["random_seed"]
    )
    stress_raw = generate_stress_test_episodes(
        plant,
        n_episodes=cfg["data"]["n_test_episodes"],
        duration_steps=cfg["data"]["steps_per_episode"],
        u_level=cfg["data"]["u_max"],
        disturbance_std=gen_cfg.disturbance_std,
        base_seed=cfg["random_seed"],
        disturbance_autocorr=gen_cfg.disturbance_autocorr,
    )
    return StudyData(
        cfg, plant, gen_cfg, train_df, test_df, build_features(stress_raw, plant)
    )


def split_validation(
    train_df: pd.DataFrame, n_val_episodes: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut ``n_val_episodes`` WHOLE episodes out of the training table."""
    return episode_train_test_split(train_df, n_test_episodes=n_val_episodes, seed=seed)


def feature_columns(feature_set: str) -> list[str]:
    """Return the (leakage-checked) column list of a feature set."""
    return resolve_feature_set(feature_set)


__all__ = [
    "GROUP_COLUMN",
    "TARGET_COLUMN",
    "StudyData",
    "build_study_data",
    "feature_columns",
    "load_study_config",
    "split_validation",
]
