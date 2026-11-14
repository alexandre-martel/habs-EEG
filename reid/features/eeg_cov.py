"""EEG spatial covariance matrices (filter bank), shrunk with OAS.

Average referencing makes the raw covariance rank-deficient (rank n-1); OAS
shrinkage makes each epoch's matrix positive definite, as Riemannian tools need.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.mean import mean_riemann


def epoch_covs(ep: np.ndarray) -> np.ndarray:
    """OAS covariance of each epoch: [n_epochs, n_channels, n_channels]."""
    return Covariances(estimator="oas").transform(ep)


def riemann_mean(covs: np.ndarray) -> np.ndarray:
    """Riemannian (affine-invariant) mean of SPD matrices; NaN if none."""
    if len(covs) == 0:
        return np.full(covs.shape[1:], np.nan)
    return covs[0] if len(covs) == 1 else mean_riemann(covs)
