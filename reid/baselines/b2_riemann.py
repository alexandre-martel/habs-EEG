"""Baseline 2: Riemannian matching of EEG spatial covariance matrices.

A session is described, for each condition (EC, EO) and frequency band, by the
Riemannian mean of its window covariances. Two scores:

* ``airm``: minus the affine-invariant Riemannian distance (summed in quadrature
  over conditions x bands);
* ``tangent``: projection onto the tangent space at the Riemannian mean of the
  *training* sessions, z-scored with training statistics, then cosine.

``recenter=True`` is a control: each session is whitened by its own mean
covariance (the usual cross-subject transfer trick), which should erase most of
the identity information.
"""

from __future__ import annotations

import numpy as np
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.distance import pairwise_distance
from pyriemann.tangentspace import TangentSpace

from reid.baselines.b1_spectral_corr import SHORT
from reid.baselines.common import Standardizer
from reid.eval.interface import cosine
from reid.eval.protocol import SessionSet
from reid.features.eeg_cov import riemann_mean

FILTER_BANK = ("theta", "alpha", "beta")


class RiemannModel:
    def __init__(
        self,
        metric: str = "tangent",
        bands=("broad",),
        conditions=("rest_ec", "rest_eo"),
        recenter: bool = False,
    ) -> None:
        self.metric, self.bands, self.conditions, self.recenter = (
            metric,
            tuple(bands),
            conditions,
            recenter,
        )
        band = "fb" if self.bands == FILTER_BANK else "+".join(self.bands)
        conds = "+".join(SHORT.get(c, c) for c in conditions)
        self.name = f"b2_{metric}_{band}_{conds}" + ("_recentered" if recenter else "")
        self.config = {
            "metric": metric,
            "bands": list(self.bands),
            "conditions": list(conditions),
            "recenter": recenter,
        }

    def session_covs(self, feats: dict) -> list[np.ndarray]:
        """One matrix per (condition, band) slot; NaN matrix if unavailable."""
        band_index = {b: i for i, b in enumerate(feats["cov_bands"])}
        ok = (feats["w_eeg_n_epochs"] > 0) & np.isfinite(feats["w_eeg_cov"]).all(
            axis=(1, 2, 3)
        )
        out = []
        for band in self.bands:
            covs = feats["w_eeg_cov"][:, band_index[band]].astype(float)
            whitener = (
                invsqrtm(riemann_mean(covs[ok])) if self.recenter and ok.any() else None
            )
            for condition in self.conditions:
                mean = riemann_mean(covs[ok & (feats["w_condition"] == condition)])
                if whitener is not None and np.isfinite(mean).all():
                    mean = whitener @ mean @ whitener
                out.append(mean)
        return out

    def _slots(self, sessions: SessionSet) -> list[np.ndarray]:
        """[slot][session, c, c] stacks."""
        per_session = [self.session_covs(f) for f in sessions]
        return [
            np.stack([s[k] for s in per_session]) for k in range(len(per_session[0]))
        ]

    def fit(self, train: SessionSet) -> None:
        if self.metric != "tangent":
            return
        self.spaces = []
        for covs in self._slots(train):
            finite = np.isfinite(covs).all(axis=(1, 2))
            self.spaces.append(TangentSpace(metric="riemann").fit(covs[finite]))
        self.scaler = Standardizer().fit(self._tangent(train))

    def _tangent(self, sessions: SessionSet) -> np.ndarray:
        vecs = []
        for space, covs in zip(self.spaces, self._slots(sessions), strict=True):
            missing = ~np.isfinite(covs).all(axis=(1, 2))
            covs = covs.copy()
            covs[missing] = space.reference_  # -> zero tangent vector
            vecs.append(space.transform(covs))
        return np.hstack(vecs)

    def embed(self, sessions: SessionSet) -> np.ndarray:
        if self.metric != "tangent":
            raise AttributeError("embed is only defined for the tangent metric")
        return self.scaler.transform(self._tangent(sessions))

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        if self.metric == "tangent":
            return cosine(self.embed(probe), self.embed(gallery))
        sq = np.zeros((len(probe), len(gallery)))
        for p, g in zip(self._slots(probe), self._slots(gallery), strict=True):
            fp, fg = np.isfinite(p).all(axis=(1, 2)), np.isfinite(g).all(axis=(1, 2))
            d = np.full(sq.shape, np.nan)
            if fp.any() and fg.any():
                d[np.ix_(fp, fg)] = pairwise_distance(p[fp], g[fg], squared=True)
            sq += np.nan_to_num(d, nan=np.nanmean(d) if np.isfinite(d).any() else 0.0)
        return -np.sqrt(sq)
