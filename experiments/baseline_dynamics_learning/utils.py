"""utils.py -- small reproducibility helper.

Data generation itself uses per-episode ``np.random.default_rng``
instances (see ``data_generation.py``), never the global ``np.random``
state, so this is only needed to seed the *model-fitting* randomness
(``RandomForestRegressor``'s bootstrap sampling, ``GridSearchCV``'s
CV-fold shuffling, etc.), which sklearn still reads from
``random_state`` args wired to this seed via ``config.yaml``.
"""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fixes global random seeds for any library that still reads them."""
    random.seed(seed)
    np.random.seed(seed)
