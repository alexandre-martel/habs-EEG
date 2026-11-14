"""Controls run with every evaluation.

* demographics only: any model must clearly beat it;
* shuffled gallery identities: must give chance (pipeline integrity);
* cohort predictability: if the features predict the cohort very well,
  identification may rest on hardware / period -> look at ``within_cohort``;
* same-day ceiling: a scenario in ``protocol.SCENARIOS`` (optimistic ceiling).

Not implemented on purpose: the inter-ocular "anatomy" control (oculometry is
excluded from the project) and the cross-cohort duplicate-candidates list, which
links pseudonymised IDs to one another and waits for HABS approval.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reid.config import SEED
from reid.eval.protocol import SessionSet

GENDER_PENALTY = 10.0  # in years of age difference


class DemographicModel:
    """Score = -|age difference| - penalty if genders differ (missing -> 0)."""

    name = "control_demographics"

    def fit(self, train: SessionSet) -> None:
        """Nothing to learn."""

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        g, p = gallery.sessions, probe.sessions
        age = np.abs(p.age.to_numpy(float)[:, None] - g.age.to_numpy(float)[None])
        gender = p.gender.to_numpy(object)[:, None] != g.gender.to_numpy(object)[None]
        known = p.gender.notna().to_numpy()[:, None] & g.gender.notna().to_numpy()[None]
        return -np.nan_to_num(age) - GENDER_PENALTY * (gender & known)


def shuffle_gallery(S: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Permute gallery identities (columns): the expected top-1 is chance."""
    return S[:, np.random.default_rng(seed).permutation(S.shape[1])]


def mask_cohorts(
    S: np.ndarray, g_cohort: np.ndarray, p_cohort: np.ndarray
) -> np.ndarray:
    """Restrict matching to the probe's own cohort (other cohorts -> -inf)."""
    same = np.asarray(p_cohort)[:, None] == np.asarray(g_cohort)[None]
    return np.where(same, S, -np.inf)


def cohort_predictability(
    X: np.ndarray, cohort: np.ndarray, groups: np.ndarray, seed: int = SEED
) -> dict:
    """Cross-validated accuracy of a logistic regression predicting the cohort
    from session embeddings (grouped by person), against the majority class."""
    ok = ~np.isnan(np.asarray(cohort, dtype=float))
    X, y, groups = np.nan_to_num(X[ok]), np.asarray(cohort)[ok].astype(int), groups[ok]
    if len(np.unique(y)) < 2:
        return {}
    n_splits = int(min(5, np.bincount(y).max(), len(np.unique(groups))))
    cv = StratifiedGroupKFold(
        n_splits=max(n_splits, 2), shuffle=True, random_state=seed
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    correct = []
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) > 1:
            correct.append(clf.fit(X[tr], y[tr]).predict(X[te]) == y[te])
    majority = np.bincount(y).max() / len(y)
    return {
        "accuracy": float(np.concatenate(correct).mean()),
        "majority": float(majority),
    }
