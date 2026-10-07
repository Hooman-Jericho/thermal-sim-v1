import numpy as np
import pandas as pd
import pytest

from experiments.baseline_dynamics_learning.data_generation import (
    DataGenConfig,
    generate_dataset,
)
from experiments.baseline_dynamics_learning.features import (
    FEATURE_SETS,
    FEATURES,
    WARMUP_ROWS,
    assert_no_leakage,
    build_features,
    find_redundant_pairs,
    resolve_feature_set,
)
from src.plant import PlantConfig


@pytest.fixture
def plant_cfg() -> PlantConfig:
    return PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=1.0)


@pytest.fixture
def raw_df(plant_cfg, tmp_path) -> pd.DataFrame:
    gen = DataGenConfig(
        n_episodes=4,
        steps_per_episode=60,
        u_max=300.0,
        disturbance_std=20.0,
        base_seed=1,
        u_hold_steps=20,
        disturbance_autocorr=0.8,
    )
    return generate_dataset(
        plant_cfg, gen, output_path=str(tmp_path / "test_features_ds.csv")
    )


# --- leakage guard -----------------------------------------------------------


def test_assert_no_leakage_blocks_unmeasured_disturbance():
    with pytest.raises(ValueError):
        assert_no_leakage(["T_current", "d"])


def test_assert_no_leakage_blocks_debug_columns():
    with pytest.raises(ValueError):
        assert_no_leakage(["T_current", "debug_d"])


def test_assert_no_leakage_blocks_target():
    with pytest.raises(ValueError):
        assert_no_leakage(["T_current", "T_next"])


def test_assert_no_leakage_blocks_unknown_feature():
    with pytest.raises(ValueError):
        assert_no_leakage(["not_a_real_feature"])


def test_assert_no_leakage_accepts_registered_features():
    assert_no_leakage(list(FEATURES))  # must not raise


@pytest.mark.parametrize("set_name", list(FEATURE_SETS))
def test_every_named_feature_set_is_leakage_free(set_name):
    resolve_feature_set(set_name)  # must not raise


def test_resolve_feature_set_rejects_unknown_name():
    with pytest.raises(ValueError):
        resolve_feature_set("not_a_real_set")


# --- build_features: episode awareness, warm-up, no NaNs ---------------------


def test_build_features_adds_every_registered_column(raw_df, plant_cfg):
    out = build_features(raw_df, plant_cfg)
    for name in FEATURES:
        assert name in out.columns


def test_build_features_drops_exactly_warmup_rows_per_episode(raw_df, plant_cfg):
    out = build_features(raw_df, plant_cfg)
    n_episodes = raw_df["episode_id"].nunique()
    assert len(out) == len(raw_df) - n_episodes * WARMUP_ROWS


def test_build_features_has_no_nans_in_feature_columns(raw_df, plant_cfg):
    out = build_features(raw_df, plant_cfg)
    assert not out[list(FEATURES)].isna().any().any()


def test_lag_features_do_not_cross_episode_boundaries(plant_cfg, tmp_path):
    """Row 0 of episode 1 must not see episode 0's last temperature as its
    'previous' value."""
    gen = DataGenConfig(
        n_episodes=2,
        steps_per_episode=30,
        u_max=300.0,
        disturbance_std=10.0,
        base_seed=7,
        u_hold_steps=10,
    )
    raw = generate_dataset(
        plant_cfg, gen, output_path=str(tmp_path / "test_boundary_ds.csv")
    )
    out = build_features(raw, plant_cfg)
    first_row_ep1 = out[out["episode_id"] == 1].iloc[0]
    last_row_ep0_T = raw[raw["episode_id"] == 0].iloc[-1]["T_current"]
    # u_lag_1 for the first kept row of episode 1 must come from WITHIN episode 1,
    # not be NaN-filled-as-zero and not equal episode 0's trailing value by coincidence.
    within_ep1_prev = raw[raw["episode_id"] == 1].iloc[WARMUP_ROWS - 1]["u"]
    assert first_row_ep1["u_lag_1"] == pytest.approx(within_ep1_prev)
    assert first_row_ep1["T_current"] != pytest.approx(
        last_row_ep0_T
    )  # sanity: different episodes differ


def test_build_features_is_causal_under_truncation(raw_df, plant_cfg):
    """A feature at row k must not change if rows AFTER k are removed (no lookahead)."""
    full = build_features(raw_df, plant_cfg)
    one_episode = raw_df[raw_df["episode_id"] == raw_df["episode_id"].iloc[0]]
    truncated_source = one_episode.iloc[: WARMUP_ROWS + 5]
    truncated = build_features(truncated_source, plant_cfg)
    row_from_full = full[
        (full["episode_id"] == one_episode["episode_id"].iloc[0])
        & (full["step"] == WARMUP_ROWS)
    ].iloc[0]
    row_from_truncated = truncated.iloc[0]
    for name in FEATURES:
        assert row_from_full[name] == pytest.approx(row_from_truncated[name]), name


def test_build_features_rejects_frame_with_gaps_in_step(plant_cfg):
    bad = pd.DataFrame(
        {
            "episode_id": [0, 0, 0],
            "step": [0, 1, 3],
            "T_current": [30.0, 31.0, 32.0],
            "u": [0.0, 0.0, 0.0],
        }
    )
    with pytest.raises(ValueError):
        build_features(bad, plant_cfg)


def test_build_features_rejects_missing_columns(plant_cfg):
    bad = pd.DataFrame({"episode_id": [0, 0], "step": [0, 1]})
    with pytest.raises(ValueError):
        build_features(bad, plant_cfg)


# --- physics correctness of dT_phys and d_hat_lag1 --------------------------


def test_dT_phys_matches_energy_balance_by_hand(raw_df, plant_cfg):
    out = build_features(raw_df, plant_cfg)
    row = out.iloc[0]
    expected = (
        plant_cfg.dt
        * (row["u"] - plant_cfg.k_loss * (row["T_current"] - plant_cfg.T_amb))
        / plant_cfg.mCp
    )
    assert row["dT_phys"] == pytest.approx(expected)


def test_d_hat_lag1_reconstructs_true_previous_disturbance(raw_df, plant_cfg):
    """d_hat_lag1 solves the SAME energy balance the data was generated from, so it
    must recover the true (simulation-only) disturbance to floating-point precision."""
    out = build_features(raw_df, plant_cfg)
    prev = raw_df.assign(prev_step=raw_df["step"] + 1)[
        ["episode_id", "prev_step", "debug_d"]
    ]
    prev = prev.rename(columns={"debug_d": "true_prev_d"})
    merged = out.merge(
        prev, left_on=["episode_id", "step"], right_on=["episode_id", "prev_step"]
    )
    assert np.allclose(merged["d_hat_lag1"], merged["true_prev_d"], atol=1e-8)


# --- redundancy audit --------------------------------------------------------


def test_find_redundant_pairs_flags_affine_copy():
    x = np.linspace(0, 100, 200)
    df = pd.DataFrame({"T": x, "T_minus_25": x - 25.0, "unrelated": np.sin(x)})
    pairs = find_redundant_pairs(df, threshold=0.999)
    names = {(a, b) for a, b, _ in pairs}
    assert ("T", "T_minus_25") in names or ("T_minus_25", "T") in names


def test_find_redundant_pairs_ignores_genuinely_different_features():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    assert find_redundant_pairs(df, threshold=0.999) == []


def test_committed_feature_sets_have_no_redundant_pairs(raw_df, plant_cfg):
    """Regression test for the Sep-27 bug: Delta_T_amb/T_error were affine copies
    of T_current."""
    out = build_features(raw_df, plant_cfg)
    cols = resolve_feature_set("full")
    assert find_redundant_pairs(out[cols], threshold=0.999) == []
