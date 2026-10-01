# Makefile -- thermal-sim-v1
# Every target assumes PYTHONPATH=. (this repo's convention; see README.md).
# Nothing here requires a W&B login: ml_pipeline.wandb_mode defaults to
# "offline" in experiments/baseline_dynamics_learning/config.yaml.

export PYTHONPATH := .

.PHONY: setup test test-fresh train-baselines train-consolidated feature-study clean

setup:
	pip install -r requirements.txt

test:
	pytest tests/ -v

# Regression check for the Day 20 finding: simulates a machine with no W&B
# login and no cached credentials. Should exit 0 exactly like `test` above --
# if this target fails while `test` passes, something reintroduced a hidden
# dependency on an authenticated W&B session.
test-fresh:
	env -u WANDB_API_KEY -u WANDB_MODE pytest tests/test_consolidated_pipeline.py -v

train-baselines:
	python -m experiments.baseline_dynamics_learning.train_baselines

feature-study:
	python -m experiments.baseline_dynamics_learning.run_feature_study

train-consolidated:
	python -m experiments.baseline_dynamics_learning.run_consolidated_pipeline

clean:
	rm -rf outputs wandb .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
