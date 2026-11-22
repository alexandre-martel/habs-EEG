"""Score-level fusion of single-modality models, as a ``ReIDModel``.

Any ``ReIDModel`` can be a member (LDA on features, contrastive encoder on raw
crops, a teammate's specialist model...): each reads its own feature store.

Training (``fit``), per fold, on the training people only:
1. every member is fitted on all training people;
2. its scores on the inner validation people (day 1 -> day 2; the same inner
   split as the LDA and the encoders) are taken from the member itself when it
   exposes them (``val_scores_`` / ``val_pids_``: the LDA at its chosen
   dimension, the encoder at its early-stopping step, both trained without the
   validation people); otherwise a fresh copy is fitted on the inner people;
3. each score matrix is z-normalised per probe (over the gallery), so that
   cosine, PLDA or probability scores live on one scale whatever the gallery;
4. the weights (on the simplex, step ``step``) maximise the validation top-1,
   ties broken by MRR then by closeness to equal weights; ``weights="equal"``
   skips this search.

Score: sum over members of weight x z-normalised score. Fitted members are
memoised per process (key, training sessions), so several fusions sharing a
member in one run train it once.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np

from reid.config import FEATURES, SEED
from reid.eval.matching import rank_counts, reciprocal_rank, topk, truth_index
from reid.eval.protocol import FeatureStore, SessionSet
from reid.final.model import inner_split
from reid.final.raw import RAW

_FITTED: dict = {}


def znorm(S: np.ndarray) -> np.ndarray:
    """Per-probe z-score over the gallery (NaN -> 0; a constant row -> 0)."""
    S = np.nan_to_num(np.asarray(S, dtype=float))
    sd = S.std(axis=1, keepdims=True)
    return np.where(sd > 1e-12, (S - S.mean(axis=1, keepdims=True)) / sd, 0.0)


def simplex(m: int, step: float) -> np.ndarray:
    """Every weight vector of m non-negative multiples of ``step`` summing to 1."""
    n = round(1 / step)
    grid = [c for c in itertools.product(range(n + 1), repeat=m) if sum(c) == n]
    return np.array(grid, dtype=float) / n


class ScoreFusion:
    """``members``: {key: (factory, store)} with store ``"features"`` or ``"raw"``."""

    def __init__(
        self,
        name: str,
        members: dict[str, tuple[Callable[[], object], str]],
        weights: str = "tuned",
        step: float = 0.1,
        seed: int = SEED,
        roots: dict[str, Path] | None = None,
    ) -> None:
        self.name, self.members, self.weights = name, members, weights
        self.step, self.seed = step, seed
        roots = roots or {"features": FEATURES, "raw": RAW}
        self.stores = {k: FeatureStore(v) for k, v in roots.items()}
        self.config = {
            "members": list(members),
            "stores": {k: s for k, (_, s) in members.items()},
            "weights_mode": weights,
            "step": step,
            "seed": seed,
        }

    def _view(self, key: str, sessions: SessionSet) -> SessionSet:
        return SessionSet(sessions.sessions, self.stores[self.members[key][1]])

    def _fit(self, key: str, train: SessionSet):
        """Member ``key`` fitted on ``train`` (memoised per process)."""
        rows = train.sessions[["participant_id", "day", "task"]].astype(str)
        memo = (key, self.members[key][1], tuple(map(tuple, rows.to_numpy())))
        if memo not in _FITTED:
            model = self.members[key][0]()
            model.fit(self._view(key, train))
            _FITTED[memo] = model
        return _FITTED[memo]

    def _val(self, key: str, model, inner, gal, prb) -> np.ndarray:
        """Member's inner-validation scores, from itself when it exposes the
        same split, else from a copy fitted on the inner people."""
        pids = getattr(model, "val_pids_", None)
        if (
            pids is not None
            and np.array_equal(pids[0], gal.pids)
            and np.array_equal(pids[1], prb.pids)
        ):
            return model.val_scores_
        return self._fit(key, inner).score(self._view(key, gal), self._view(key, prb))

    @staticmethod
    def _metrics(w, Z: list, truth: np.ndarray) -> tuple[float, float]:
        greater, ties = rank_counts(sum(wi * z for wi, z in zip(w, Z)), truth)
        return topk(greater, ties, 1).mean(), reciprocal_rank(greater, ties).mean()

    def _choose(self, Z: list, truth: np.ndarray) -> np.ndarray:
        m = len(Z)
        if self.weights == "equal" or m == 1:
            return np.full(m, 1 / m)
        best, best_key = None, None
        for w in simplex(m, self.step):
            key = (*self._metrics(w, Z, truth), -np.abs(w - 1 / m).sum())
            if best_key is None or key > best_key:
                best, best_key = w, key
        return best

    def fit(self, train: SessionSet) -> None:
        inner, gal, prb = inner_split(train, self.seed)
        self.models = {k: self._fit(k, train) for k in self.members}
        Z = [znorm(self._val(k, m, inner, gal, prb)) for k, m in self.models.items()]
        truth = truth_index(gal.pids, prb.pids)
        single = {}
        for k, z in zip(self.members, Z):
            greater, ties = rank_counts(z, truth)
            single[k] = round(float(topk(greater, ties, 1).mean()), 4)
        self.w_ = self._choose(Z, truth)
        # validation metrics of the chosen and of equal weights, so that the
        # choice between fusion variants can itself be made on validation
        top1, mrr = self._metrics(self.w_, Z, truth)
        top1_eq, mrr_eq = self._metrics(np.full(len(Z), 1 / len(Z)), Z, truth)
        info = {
            "val_top1": float(top1),
            "val_mrr": float(mrr),
            "val_top1_equal": float(top1_eq),
            "val_mrr_equal": float(mrr_eq),
        }
        weights = dict(zip(self.members, np.round(self.w_, 3).tolist()))
        print(f"  {self.name}: val top-1 {single}, weights {weights}", flush=True)
        self.config |= {"weights": weights, "val_member_top1": single, **info}

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        return sum(
            w * znorm(m.score(self._view(k, gallery), self._view(k, probe)))
            for w, (k, m) in zip(self.w_, self.models.items())
        )

    def embed(self, sessions: SessionSet) -> np.ndarray:
        """Concatenated member embeddings x sqrt(weight) (cohort control only)."""
        parts = []
        for w, (k, m) in zip(self.w_, self.models.items()):
            e = np.nan_to_num(np.asarray(m.embed(self._view(k, sessions)), float))
            e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            parts.append(np.sqrt(w) * e)
        return np.concatenate(parts, axis=1)
