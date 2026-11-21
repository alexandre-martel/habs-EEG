"""Rank-based identification and one-to-one (Hungarian) matching.

Ties are broken at random *in expectation*: a probe whose true score is tied
with t impostors and beaten by g of them has its true match uniformly at ranks
g+1 ... g+t+1. This keeps constant or coarse scores (e.g. demographics) at
exactly chance level instead of an optimistic or pessimistic bias.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def truth_index(gallery_pids: np.ndarray, probe_pids: np.ndarray) -> np.ndarray:
    """Index of each probe's true gallery entry (-1 if absent)."""
    where = {pid: j for j, pid in enumerate(gallery_pids)}
    return np.array([where.get(pid, -1) for pid in probe_pids])


def rank_counts(S: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per probe: number of gallery entries scoring strictly higher than the
    true match, and number tied with it (excluding itself). NaN = -inf."""
    S = np.where(np.isnan(S), -np.inf, S)
    rows = np.arange(len(truth))
    true = S[rows, truth][:, None]
    greater = (S > true).sum(axis=1)
    ties = (S == true).sum(axis=1) - 1
    return greater, ties


def topk(greater: np.ndarray, ties: np.ndarray, k: int) -> np.ndarray:
    """Probability that the true match is within the top k (random tie-break)."""
    return np.clip((k - greater) / (ties + 1), 0.0, 1.0)


def reciprocal_rank(greater: np.ndarray, ties: np.ndarray) -> np.ndarray:
    """Expected 1/rank under random tie-breaking."""
    return np.array(
        [np.mean(1.0 / np.arange(g + 1, g + t + 2)) for g, t in zip(greater, ties)]
    )


def hungarian(S: np.ndarray) -> np.ndarray:
    """One-to-one assignment maximising the total score: gallery column per probe."""
    finite = np.where(np.isfinite(S), S, np.nanmin(S[np.isfinite(S)]) - 1e6)
    rows, cols = linear_sum_assignment(-finite)
    assigned = np.full(len(S), -1)
    assigned[rows] = cols
    return assigned


def hungarian_correct(
    S: np.ndarray, truth: np.ndarray, n_perm: int = 20, seed: int = 0
) -> np.ndarray:
    """Per-probe probability that the Hungarian assignment is right, ties broken
    at random: ``linear_sum_assignment`` keeps the first optimum it finds, and
    gallery and probes are sorted by the same ids, so with tied scores (e.g.
    demographics) the identity diagonal would win by construction. Rows and
    columns are permuted ``n_perm`` times; with untied scores every permutation
    gives the same answer."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(S))
    for _ in range(n_perm):
        r, c = rng.permutation(S.shape[0]), rng.permutation(S.shape[1])
        assigned = np.full(len(S), -1)
        assigned[r] = c[hungarian(S[np.ix_(r, c)])]
        hits += assigned == truth
    return hits / n_perm
