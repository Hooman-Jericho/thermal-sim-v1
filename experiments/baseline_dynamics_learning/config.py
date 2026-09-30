"""
config.py
---------
Layered configuration: a base file (``config.yaml``, the single source of
truth for the physics) plus an optional overrides file (``feature_study.yaml``)
that states *only what the study changes*.

Why layered: the physics parameters used to be copied into every new
experiment config, and copies drift. Here the study cannot silently disagree
with the plant it evaluates. Configs are validated before any computation
starts, so a typo fails in milliseconds instead of after a two-minute run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.baseline_dynamics_learning.data_generation import DataGenConfig
from experiments.baseline_dynamics_learning.estimators import MODEL_NAMES, TARGET_MODES
from experiments.baseline_dynamics_learning.features import FEATURE_SETS
from src.plant import PlantConfig

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "config.yaml"
DEFAULT_OVERRIDES = HERE / "feature_study.yaml"


def deep_merge(base: dict, overrides: dict) -> dict:
    """Return ``base`` updated recursively by ``overrides`` (inputs are not mutated)."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(base_path: Path | str = DEFAULT_BASE,
                overrides_path: Path | str | None = DEFAULT_OVERRIDES,
                extra_overrides: dict | None = None) -> dict[str, Any]:
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides_path is not None:
        with open(overrides_path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if extra_overrides:
        cfg = deep_merge(cfg, extra_overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail fast, with a message that names the offending key."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"Invalid config: {msg}")

    for section in ("physics", "data", "models", "ml_pipeline", "feature_study"):
        need(section in cfg, f"missing section '{section}'")
    ph, da, fs = cfg["physics"], cfg["data"], cfg["feature_study"]
    need(ph["T_min"] < ph["T_max"], "physics.T_min must be < physics.T_max")
    for k in ("mCp", "k_loss", "dt"):
        need(ph[k] > 0, f"physics.{k} must be > 0")
    need(0.0 <= da.get("disturbance_autocorr", 0.0) < 1.0, "data.disturbance_autocorr must be in [0, 1)")
    need(da["n_test_episodes"] < da["n_episodes"], "data.n_test_episodes must be < data.n_episodes")
    n_train = da["n_episodes"] - da["n_test_episodes"]
    need(2 <= fs["n_folds"] <= da["n_episodes"], "feature_study.n_folds must be in [2, data.n_episodes]")
    need(n_train >= 2, "need at least 2 training episodes")
    for m in fs["models"]:
        need(m in MODEL_NAMES, f"unknown model '{m}' (known: {MODEL_NAMES})")
    for s in fs["feature_sets"]:
        need(s in FEATURE_SETS, f"unknown feature set '{s}' (known: {sorted(FEATURE_SETS)})")
    for t in fs["target_modes"]:
        need(t in TARGET_MODES, f"unknown target mode '{t}' (known: {TARGET_MODES})")
    need("raw" in fs["feature_sets"], "feature_sets must include 'raw' (the ablation baseline)")
    need(fs["ablation_target_mode"] in fs["target_modes"], "ablation_target_mode must be in target_modes")
    need(fs["target_ablation_feature_set"] in fs["feature_sets"], "target_ablation_feature_set must be in feature_sets")
    need("absolute_unscaled" in fs["target_modes"], "target_modes must include 'absolute_unscaled' (the target-ablation baseline)")
    for name, recipe in fs["safety_recipes"].items():
        need(recipe["feature_set"] in FEATURE_SETS, f"safety recipe '{name}': unknown feature_set")
        need(recipe["target_mode"] in TARGET_MODES, f"safety recipe '{name}': unknown target_mode")


def plant_from_cfg(cfg: dict) -> PlantConfig:
    p = cfg["physics"]
    return PlantConfig(mCp=p["mCp"], k_loss=p["k_loss"], T_amb=p["T_amb"], dt=p["dt"])


def datagen_from_cfg(cfg: dict) -> DataGenConfig:
    d = cfg["data"]
    return DataGenConfig(
        n_episodes=d["n_episodes"], steps_per_episode=d["steps_per_episode"],
        u_max=d["u_max"], disturbance_std=d["disturbance_std"],
        base_seed=cfg["random_seed"], u_hold_steps=d.get("u_hold_steps", 1),
        disturbance_autocorr=d.get("disturbance_autocorr", 0.0),
    )


def config_fingerprint(cfg: dict) -> str:
    """Stable short hash of the resolved config, stored in the run manifest."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
