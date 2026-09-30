import copy

import pytest

from experiments.baseline_dynamics_learning.config import (
    DEFAULT_BASE, DEFAULT_OVERRIDES, config_fingerprint, deep_merge,
    load_config, validate_config,
)


def test_deep_merge_overrides_leaf_values_without_dropping_siblings():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overrides = {"a": {"x": 99}}
    merged = deep_merge(base, overrides)
    assert merged == {"a": {"x": 99, "y": 2}, "b": 3}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    overrides = {"a": {"x": 2}}
    base_copy, overrides_copy = copy.deepcopy(base), copy.deepcopy(overrides)
    deep_merge(base, overrides)
    assert base == base_copy and overrides == overrides_copy


def test_deep_merge_adds_new_top_level_sections():
    merged = deep_merge({"a": 1}, {"b": {"c": 2}})
    assert merged == {"a": 1, "b": {"c": 2}}


def test_load_config_real_files_merge_and_validate():
    cfg = load_config(DEFAULT_BASE, DEFAULT_OVERRIDES)
    assert "feature_study" in cfg
    assert cfg["data"]["disturbance_autocorr"] == 0.9        # from feature_study.yaml
    assert cfg["physics"]["T_min"] < cfg["physics"]["T_max"]  # from config.yaml, untouched


def test_load_config_without_overrides_still_validates_if_extra_overrides_supplied():
    extra = {"feature_study": {
        "n_folds": 3, "cv_seed": 1, "models": ["linear"], "feature_sets": ["raw"],
        "target_modes": ["absolute_unscaled"], "ablation_target_mode": "absolute_unscaled",
        "target_ablation_feature_set": "raw", "redundancy_threshold": 0.999,
        "safety_recipes": {"x": {"feature_set": "raw", "target_mode": "absolute_unscaled"}},
    }}
    cfg = load_config(DEFAULT_BASE, overrides_path=None, extra_overrides=extra)
    assert cfg["feature_study"]["n_folds"] == 3


@pytest.mark.parametrize("bad_patch, bad_section", [
    ({"physics": {"T_min": 100, "T_max": 50}}, "physics"),
    ({"physics": {"mCp": -1}}, "physics"),
    ({"data": {"disturbance_autocorr": 1.0}}, "data"),
    ({"data": {"n_test_episodes": 999}}, "data"),
])
def test_validate_config_rejects_bad_physical_values(bad_patch, bad_section):
    cfg = load_config(DEFAULT_BASE, DEFAULT_OVERRIDES)
    cfg = deep_merge(cfg, bad_patch)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_rejects_unknown_model_name():
    cfg = load_config(DEFAULT_BASE, DEFAULT_OVERRIDES)
    cfg["feature_study"]["models"] = ["not_a_real_model"]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_rejects_unknown_feature_set():
    cfg = load_config(DEFAULT_BASE, DEFAULT_OVERRIDES)
    cfg["feature_study"]["feature_sets"] = ["not_a_real_set"]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_requires_raw_in_feature_sets():
    cfg = load_config(DEFAULT_BASE, DEFAULT_OVERRIDES)
    cfg["feature_study"]["feature_sets"] = ["physics", "full"]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_config_fingerprint_is_stable_and_sensitive_to_changes():
    cfg = load_config(DEFAULT_BASE, DEFAULT_OVERRIDES)
    fp1 = config_fingerprint(cfg)
    fp2 = config_fingerprint(copy.deepcopy(cfg))
    assert fp1 == fp2
    cfg2 = deep_merge(cfg, {"data": {"n_episodes": cfg["data"]["n_episodes"] + 1}})
    assert config_fingerprint(cfg2) != fp1
