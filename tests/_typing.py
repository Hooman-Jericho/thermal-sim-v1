"""Small typed helpers that narrow ``Optional`` model attributes for mypy."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class _Fitted(Protocol):
    """Anything exposing the fitted-parameter attributes of a scratch model."""

    weights: NDArray[np.float64] | None
    bias: float | None


def fitted_weights(model: _Fitted) -> NDArray[np.float64]:
    """Return ``model.weights``, failing the test clearly if unfitted."""
    assert model.weights is not None, "model has not been fitted"
    return model.weights
