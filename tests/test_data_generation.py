import numpy as np
import pytest

from experiments.baseline_dynamics_learning.data_generation import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    DataGenConfig,
    episode_train_test_split,
    generate_dataset,
    generate_stress_test_episodes,
)
from src.plant import PlantConfig


@pytest.fixture
def plant_cfg() -> PlantConfig:
    return PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=1.0)


@pytest.fixture
def gen_cfg() -> DataGenConfig:
    return DataGenConfig(
        n_episodes=6, steps_per_episode=50, u_max=300.0,
        disturbance_std=5.0, base_seed=42, u_hold_steps=10,
    )


def test_d_is_never_a_feature_column():
    """Regression test for the v1 bug: unmeasured 'd' leaking into X."""
    assert "d" not in FEATURE_COLUMNS
    assert "debug_d" not in FEATURE_COLUMNS


def test_dataset_reproducible_given_seed(plant_cfg, gen_cfg, tmp_path):
    df1 = generate_dataset(plant_cfg, gen_cfg, output_path=str(tmp_path / "a.csv"))
    df2 = generate_dataset(plant_cfg, gen_cfg, output_path=str(tmp_path / "b.csv"))
    assert np.allclose(df1["T_next"].values, df2["T_next"].values)


def test_different_episodes_have_different_trajectories(plant_cfg, gen_cfg, tmp_path):
    df = generate_dataset(plant_cfg, gen_cfg, output_path=str(tmp_path / "c.csv"))
    first_ep = df[df[GROUP_COLUMN] == 0]["T_next"].values
    second_ep = df[df[GROUP_COLUMN] == 1]["T_next"].values
    assert not np.allclose(first_ep, second_ep)


def test_episode_split_has_no_overlap(plant_cfg, gen_cfg, tmp_path):
    df = generate_dataset(plant_cfg, gen_cfg, output_path=str(tmp_path / "d.csv"))
    train_df, test_df = episode_train_test_split(df, n_test_episodes=2, seed=0)
    train_ids = set(train_df[GROUP_COLUMN].unique())
    test_ids = set(test_df[GROUP_COLUMN].unique())
    assert train_ids.isdisjoint(test_ids)
    assert len(test_ids) == 2
    assert len(train_df) + len(test_df) == len(df)


def test_episode_split_rejects_too_many_test_episodes(plant_cfg, gen_cfg, tmp_path):
    df = generate_dataset(plant_cfg, gen_cfg, output_path=str(tmp_path / "e.csv"))
    with pytest.raises(ValueError):
        episode_train_test_split(df, n_test_episodes=gen_cfg.n_episodes, seed=0)


def test_stress_test_reaches_high_temperatures(plant_cfg):
    df = generate_stress_test_episodes(
        plant_cfg, n_episodes=2, duration_steps=500, u_level=300.0,
        disturbance_std=5.0, base_seed=42,
    )
    # Steady state at u=300 is T_amb + 300/5 = 85 -- should get close.
    assert df["T_next"].max() > 75.0
