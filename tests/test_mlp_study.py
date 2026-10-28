"""Tests for the MLP study: tuning, W&B helpers, hardening, the runner (Days 38-44)."""

import json
import logging
import os

import numpy as np
import pandas as pd
import pytest

from experiments.baseline_dynamics_learning.config import plant_from_cfg
from experiments.mlp_dynamics import hardening, wandb_utils
from experiments.mlp_dynamics.data import build_study_data, load_study_config
from experiments.mlp_dynamics.run_mlp_study import (
    longest_decreasing_streak,
    moving_average,
    noise_floor_mse,
    run,
    to_markdown,
)
from experiments.mlp_dynamics.tuning import baseline_for, tune_mlp

SMALL = {
    "random_seed": 5,
    "data": {"n_episodes": 8, "steps_per_episode": 200, "n_test_episodes": 2},
    "ml_pipeline": {"wandb_mode": "disabled"},
    "models": {
        "random_forest": {"param_grid": {"n_estimators": [10], "max_depth": [3]}}
    },
    "optuna": {
        "n_trials": 2,
        "n_folds": 2,
        "max_epochs_cap": 6,
        "patience": 3,
        "pruner_warmup_epochs": 2,
    },
    "mlp_study": {
        "mlp_seeds": [0, 1],
        "n_val_episodes": 2,
        "curve_epochs": 6,
        "models": ["linear", "random_forest", "mlp"],
    },
}


@pytest.fixture(scope="module")
def small_cfg():
    return load_study_config(SMALL)


# --- config / data -----------------------------------------------------------
def test_study_config_inherits_physics_from_the_base_file(small_cfg):
    assert small_cfg["physics"]["mCp"] == 500.0
    assert small_cfg["data"]["disturbance_autocorr"] == 0.9  # from feature_study.yaml


def test_no_leak_columns_in_study_feature_sets(small_cfg, tmp_path):
    from experiments.mlp_dynamics.data import feature_columns

    for fs in ("raw", "full"):
        assert "debug_d" not in feature_columns(fs)
        assert "T_next" not in feature_columns(fs)
    data = build_study_data(small_cfg, tmp_path)
    assert set(data.train_df.episode_id).isdisjoint(set(data.test_df.episode_id))


# --- tuning --------------------------------------------------------------------
def test_baseline_for_modes_and_error(small_cfg, tmp_path):
    data = build_study_data(small_cfg, tmp_path)
    df, p = data.train_df, data.plant
    assert np.all(baseline_for(df, "absolute", p) == 0)
    np.testing.assert_array_equal(baseline_for(df, "delta", p), df["T_current"])
    with pytest.raises(ValueError):
        baseline_for(df, "absolute_unscaled", p)


def test_tune_mlp_runs_is_seeded_and_uses_only_train_rows(small_cfg, tmp_path):
    from experiments.mlp_dynamics.data import feature_columns

    data = build_study_data(small_cfg, tmp_path)
    cols = feature_columns("full")
    r1 = tune_mlp(
        data.train_df, cols, "physics_residual", data.plant, small_cfg["optuna"]
    )
    r2 = tune_mlp(
        data.train_df, cols, "physics_residual", data.plant, small_cfg["optuna"]
    )
    assert r1.best_params == r2.best_params and r1.best_value == r2.best_value
    assert 1 <= r1.n_epochs <= small_cfg["optuna"]["max_epochs_cap"]
    assert len(r1.trials) == 2
    assert r1.model_cfg()["max_epochs"] == r1.n_epochs


# --- W&B helpers (Day 39) ---------------------------------------------------------
def test_online_without_credentials_downgrades_to_offline(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc
    with caplog.at_level(logging.WARNING):
        assert wandb_utils.resolve_mode("online") == "offline"
    assert "no credentials" in caplog.text


def test_online_with_key_stays_online(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert wandb_utils.resolve_mode("online") == "online"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        wandb_utils.resolve_mode("sometimes")


def test_safe_log_swallows_errors():
    class Broken:
        def log(self, *a, **k):
            raise RuntimeError("network down")

    wandb_utils.safe_log(Broken(), {"x": 1})  # must not raise
    wandb_utils.finish(Broken())  # must not raise either


def test_start_run_disabled_and_epoch_logger(small_cfg):
    run_ctx = wandb_utils.start_run(small_cfg, "t")
    cb = wandb_utils.epoch_logger(run_ctx)
    cb(1, 0.5, 0.4)
    cb(2, 0.4, None)
    wandb_utils.finish(run_ctx)


# --- hardening (Day 40) ------------------------------------------------------------
def test_one_hundred_consecutive_episodes_do_not_crash(small_cfg):
    out = hardening.run_consecutive_episodes(
        plant_from_cfg(small_cfg), n_episodes=100, steps=120
    )
    assert out["episodes"] == 100 and out["rows"] == 100 * 120
    assert np.isfinite(out["t_min"]) and np.isfinite(out["t_max"])


def test_hardening_detects_a_nan(small_cfg, monkeypatch):
    from src.plant import ActuatedThermalPlant

    real = ActuatedThermalPlant.sample_disturbance
    monkeypatch.setattr(
        ActuatedThermalPlant,
        "sample_disturbance",
        lambda self, std: 1e308 if std > 0 else real(self, std),
    )
    with pytest.raises((AssertionError, ValueError, OverflowError)):
        hardening.run_consecutive_episodes(
            plant_from_cfg(small_cfg), n_episodes=3, steps=20
        )


def test_reproducibility_report_all_true(small_cfg):
    rep = hardening.check_reproducibility(small_cfg)
    assert all(rep.values()), rep


def test_frame_hash_changes_with_one_value():
    a = pd.DataFrame({"x": [1.0, 2.0]})
    b = pd.DataFrame({"x": [1.0, 2.0000001]})
    assert hardening.frame_hash(a) == hardening.frame_hash(a.copy())
    assert hardening.frame_hash(a) != hardening.frame_hash(b)


# --- runner helpers ---
def test_streak_and_moving_average():
    assert longest_decreasing_streak([5, 4, 3, 3.5, 3, 2, 1]) == 3
    assert longest_decreasing_streak([1.0]) == 0
    assert moving_average([2.0, 4.0, 6.0], 2) == [2.0, 3.0, 5.0]


def test_noise_floor_matches_formula(small_cfg):
    expected = (1.0 / 500.0) ** 2 * (1 - 0.9**2) * 30.0**2
    assert noise_floor_mse(small_cfg) == pytest.approx(expected)


def test_to_markdown_marks_nan_as_na():
    t = pd.DataFrame(
        [
            {
                "model": "m",
                "recipe": "r",
                "seeds": 1,
                **{
                    c: float("nan")
                    for c in [
                        "R2_T_next",
                        "R2_delta",
                        "MAE",
                        "MSE",
                        "Skill_vs_persistence",
                        "Skill_vs_physics",
                        "stress_missed_violation_pct",
                    ]
                },
            }
        ]
    )
    assert "n/a" in to_markdown(t)


# --- the whole study, tiny ---
def test_run_study_end_to_end_without_wandb_login(small_cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    manifest = run(small_cfg, out_dir=tmp_path, n_trials=2)
    for name in (
        "comparison.csv",
        "comparison.md",
        "manifest.json",
        "training_curve.png",
        "mlp_best.pkl",
        "optuna_trials_best.csv",
    ):
        assert (tmp_path / name).exists(), name
    table = pd.read_csv(tmp_path / "comparison.csv")
    models = set(table["model"])
    assert {
        "persistence (no learning)",
        "physics, d = 0 (no learning)",
        "linear",
        "random_forest",
        "mlp",
    } <= models
    mlp = table[table["model"] == "mlp"]
    assert (mlp["seeds"] == 2).all()
    assert json.loads((tmp_path / "manifest.json").read_text())["noise_floor_mse"] > 0
    assert manifest["training_curve"]["epochs"] == 6
    assert os.environ.get("WANDB_API_KEY") is None
