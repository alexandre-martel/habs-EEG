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

With ``REID_SCORES`` set to a folder, every member's raw score matrices are
also saved there (``<fusion>_<fold>_val.npz`` at fit, ``..._test.npz`` at
score), with the index of each probe's true gallery entry and no participant
id: fusion variants (seed ensembles, normalisations, member sets) can then be
compared offline, chosen on the inner validation scores. ``<fold>`` is a hash
of the fold's training sessions, the same for every seed and fusion.
"""

from __future__ import annotations

import hashlib
import itertools
import os
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
    """Per-probe z-score over the gallery; missing scores (NaN: a session
    without this member's data) are left out of the mean and spread, then
    set to 0, the neutral value; a constant or empty row -> 0."""
    S = np.asarray(S, dtype=float)
    if np.isfinite(S).all():
        sd = S.std(axis=1, keepdims=True)
        return np.where(sd > 1e-12, (S - S.mean(axis=1, keepdims=True)) / sd, 0.0)
    ok = np.isfinite(S)
    n = np.maximum(ok.sum(axis=1, keepdims=True), 1)
    mean = np.where(ok, S, 0.0).sum(axis=1, keepdims=True) / n
    sd = np.sqrt(np.where(ok, (S - mean) ** 2, 0.0).sum(axis=1, keepdims=True) / n)
    Z = np.where(ok & (sd > 1e-12), (S - mean) / np.where(sd > 1e-12, sd, 1.0), 0.0)
    return np.nan_to_num(Z)


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

    def _dump(self, kind: str, scores: dict, truth: np.ndarray) -> None:
        folder = os.environ.get("REID_SCORES")
        if not folder:
            return
        Path(folder).mkdir(parents=True, exist_ok=True)
        arrays = {f"S:{k}": np.asarray(v, np.float32) for k, v in scores.items()}
        np.savez_compressed(
            Path(folder) / f"{self.name}_{self.fold_}_{kind}.npz",
            truth=truth,
            **arrays,
        )

    def fit(self, train: SessionSet) -> None:
        inner, gal, prb = inner_split(train, self.seed)
        rows = train.sessions[["participant_id", "day", "task"]].astype(str)
        key = "|".join(sorted(map(",".join, rows.to_numpy())))
        self.fold_ = hashlib.sha1(key.encode()).hexdigest()[:10]
        self.models = {k: self._fit(k, train) for k in self.members}
        raw = {k: self._val(k, m, inner, gal, prb) for k, m in self.models.items()}
        Z = [znorm(v) for v in raw.values()]
        truth = truth_index(gal.pids, prb.pids)
        self._dump("val", raw, truth)
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
        raw = {
            k: m.score(self._view(k, gallery), self._view(k, probe))
            for k, m in self.models.items()
        }
        self._dump("test", raw, truth_index(gallery.pids, probe.pids))
        return sum(w * znorm(raw[k]) for w, k in zip(self.w_, self.models))

    def embed(self, sessions: SessionSet) -> np.ndarray:
        """Concatenated member embeddings x sqrt(weight) (cohort control only)."""
        parts = []
        for w, (k, m) in zip(self.w_, self.models.items()):
            e = np.nan_to_num(np.asarray(m.embed(self._view(k, sessions)), float))
            e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            parts.append(np.sqrt(w) * e)
        return np.concatenate(parts, axis=1)


class TaskView:
    """A member that reads another task of the same sessions (e.g. the video
    recorded the same day as the rest): the fusion still sees one row per rest
    session, and this member scores the session's ``task`` recording instead.

    * fit: on the training people's ``task`` recordings, with the inner
      validation people of the rest split (``val_people``), so that its
      validation scores line up with the other members';
    * score / validation scores: NaN where a session has no ``task``
      recording (neutral 0 after ``znorm``).
    """

    def __init__(self, factory: Callable[[], object], task: str, seed: int = SEED):
        self.factory, self.task, self.seed = factory, task, seed

    def _map(self, sessions: SessionSet) -> tuple[SessionSet, np.ndarray]:
        df = sessions.sessions.copy()
        df["task"] = self.task
        keep = np.array(
            [
                sessions.store.available(pid, int(day), self.task)
                for pid, day in zip(df.participant_id, df.day)
            ],
            dtype=bool,
        )
        view = SessionSet(df[keep].reset_index(drop=True), sessions.store)
        return view, np.flatnonzero(keep)

    def _expand(self, S, rows, cols, shape) -> np.ndarray:
        out = np.full(shape, np.nan)
        if len(rows) and len(cols):
            out[np.ix_(rows, cols)] = S
        return out

    def fit(self, train: SessionSet) -> None:
        _, gal, prb = inner_split(train, self.seed)
        view, _ = self._map(train)
        self.model = self.factory()
        self.model.val_people = set(gal.pids)
        self.model.fit(view)
        config = getattr(self.model, "config", {})
        self.config = {f"{self.task}_{k}": v for k, v in config.items()}
        # validation scores on the rest split's validation sessions
        vg, vp = self.model.val_pids_
        rows = [i for i, p in enumerate(prb.pids) if p in set(vp)]
        cols = [j for j, p in enumerate(gal.pids) if p in set(vg)]
        order_p = {p: k for k, p in enumerate(vp)}
        order_g = {p: k for k, p in enumerate(vg)}
        S = self.model.val_scores_[
            np.ix_(
                [order_p[prb.pids[i]] for i in rows],
                [order_g[gal.pids[j]] for j in cols],
            )
        ]
        self.val_scores_ = self._expand(S, rows, cols, (len(prb), len(gal)))
        self.val_pids_ = (gal.pids, prb.pids)

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        g, gi = self._map(gallery)
        p, pi = self._map(probe)
        S = self.model.score(g, p) if len(g) and len(p) else np.empty((0, 0))
        return self._expand(S, pi, gi, (len(probe), len(gallery)))

    def embed(self, sessions: SessionSet) -> np.ndarray:
        v, rows = self._map(sessions)
        e = np.asarray(self.model.embed(v), float)
        out = np.full((len(sessions), e.shape[1]), np.nan)
        out[rows] = e
        return out
