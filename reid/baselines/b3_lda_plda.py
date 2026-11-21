"""Baseline 3: PCA + LDA on per-window features, then two-covariance PLDA.

LDA is trained on the training people using *both* of their days, with one class
per person: it learns a projection that separates people while pulling together
windows of the same person across days, i.e. it learns to ignore what changes
from one session to the next (impedance, fatigue, mood) - the "Fisherfaces" idea.
A session embedding is the mean projection of its windows (cosine score).

PLDA (speaker verification) models the embedding x of a session as
``x = mu + y + e`` with a person term ``y ~ N(0, B)`` and a session term
``e ~ N(0, W)``, and scores a pair with the log-likelihood ratio
"same person" vs "different people".
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from reid.baselines.common import stack_windows, window_matrix
from reid.config import SEED
from reid.eval.interface import cosine
from reid.eval.matching import rank_counts, topk, truth_index
from reid.eval.protocol import SessionSet

ALL = ("eeg", "ecg", "ecg_beat", "eda", "resp")


def _logdet(A: np.ndarray) -> float:
    """log-determinant of a symmetric positive-definite matrix (Cholesky)."""
    return float(2 * np.log(np.diag(np.linalg.cholesky(A))).sum())


class TwoCovPLDA:
    """Two-covariance PLDA with closed-form log-likelihood-ratio scoring.

    With T = B + W, the LLR of a pair (x1, x2) is
    0.5 x1'Q x1 + 0.5 x2'Q x2 + x1'P x2 + c, where
    Q = T^-1 - (T - B T^-1 B)^-1 and P = T^-1 B (T - B T^-1 B)^-1.
    """

    def __init__(self, eps: float = 1e-6, length_norm: bool = True) -> None:
        self.eps, self.length_norm = eps, length_norm

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X - self.mu)
        if self.length_norm:  # makes embeddings closer to Gaussian
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            X *= np.sqrt(X.shape[1])
        return X

    def fit(self, X: np.ndarray, labels: np.ndarray) -> TwoCovPLDA:
        """Moment estimates of B and W from people with >= 2 sessions."""
        self.mu = np.nanmean(X, axis=0)
        X = self._prep(X)
        classes, counts = np.unique(labels, return_counts=True)
        classes = classes[counts >= 2]
        means = np.array([X[labels == c].mean(0) for c in classes])
        resid = np.vstack(
            [X[labels == c] - m for c, m in zip(classes, means, strict=True)]
        )
        d = X.shape[1]
        W = resid.T @ resid / max(len(resid) - len(classes), 1)
        n_mean = np.mean([np.sum(labels == c) for c in classes])
        B = np.cov(means, rowvar=False).reshape(d, d) - W / n_mean
        vals, vecs = np.linalg.eigh(B)  # project back to positive definite
        B = (vecs * np.maximum(vals, self.eps)) @ vecs.T
        W = W + self.eps * np.eye(d)
        self.B, self.W = B, W
        T = B + W
        T_inv = np.linalg.inv(T)
        M = np.linalg.inv(T - B @ T_inv @ B)
        self.Q, self.P = T_inv - M, T_inv @ B @ M
        joint = np.block([[T, B], [B, T]])
        self.const = -0.5 * _logdet(joint) + _logdet(T)
        return self

    def score(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """LLR matrix [n1, n2]."""
        a, b = self._prep(X1), self._prep(X2)
        qa = 0.5 * np.einsum("ij,jk,ik->i", a, self.Q, a)
        qb = 0.5 * np.einsum("ij,jk,ik->i", b, self.Q, b)
        return qa[:, None] + qb[None] + a @ self.P @ b.T + self.const


class LdaModel:
    """PCA + LDA window projection; cosine (``plda=False``) or PLDA scoring."""

    def __init__(
        self,
        modalities=("eeg",),
        conditions=("rest_ec", "rest_eo"),
        eeg_kind: str = "rel",
        n_components=(16, 32, 64, 128, 256),
        plda: bool = False,
        multi_day_only: bool = True,
        seed: int = SEED,
    ) -> None:
        self.feat = {
            "modalities": modalities,
            "conditions": conditions,
            "eeg_kind": eeg_kind,
        }
        self.grid, self.use_plda = tuple(n_components), plda
        self.multi_day_only, self.seed = multi_day_only, seed
        mods = "all" if tuple(modalities) == ALL else "+".join(modalities)
        self.name = f"b3_{'plda' if plda else 'lda'}_{mods}"
        self.config = {
            **self.feat,
            "n_components": list(self.grid),
            "plda": plda,
            "multi_day_only": multi_day_only,
        }

    def _fit_pipeline(self, sessions: SessionSet, n: int) -> Pipeline:
        X, idx = stack_windows(sessions, **self.feat)
        people = sessions.sessions.groupby("participant_id").day.nunique()
        y = sessions.pids[idx]
        if self.multi_day_only:
            keep = np.isin(y, people.index[people >= 2])
            X, y = X[keep], y[keep]
        n = int(min(n, X.shape[0] - 1, X.shape[1]))
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n, random_state=self.seed),
            LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
        )
        return pipe.fit(X, y)

    def _embed(self, pipe: Pipeline, sessions: SessionSet) -> np.ndarray:
        out = []
        for f in sessions:
            W = window_matrix(f, **self.feat)
            out.append(pipe.transform(W).mean(0) if len(W) else None)
        dim = next(len(e) for e in out if e is not None)
        return np.array([e if e is not None else np.full(dim, np.nan) for e in out])

    def _select_n(self, train: SessionSet) -> int:
        """Inner validation: 20 % of the training people with days 1 and 2,
        day 1 -> day 2 top-1 with the model's own score (cosine, or PLDA fitted
        on the inner people: PLDA degrades when the dimension nears the number
        of training people, which cosine does not see)."""
        if len(self.grid) == 1:
            return self.grid[0]
        df = train.sessions
        both = df.groupby("participant_id").day.agg(lambda d: {1, 2} <= set(d))
        pool = np.array(sorted(both.index[both]))
        rng = np.random.default_rng(self.seed)
        val = set(rng.choice(pool, max(len(pool) // 5, 2), replace=False))
        inner = train.subset(~df.participant_id.isin(val).to_numpy())
        gal = train.subset((df.participant_id.isin(val) & (df.day == 1)).to_numpy())
        prb = train.subset((df.participant_id.isin(val) & (df.day == 2)).to_numpy())
        scores = {}
        for n in self.grid:
            pipe = self._fit_pipeline(inner, n)
            p, g = self._embed(pipe, prb), self._embed(pipe, gal)
            if self.use_plda:
                plda = TwoCovPLDA().fit(self._embed(pipe, inner), inner.pids)
                S = plda.score(p, g)
            else:
                S = cosine(p, g)
            greater, ties = rank_counts(S, truth_index(gal.pids, prb.pids))
            scores[n] = topk(greater, ties, 1).mean()
        return max(scores, key=scores.get)

    def fit(self, train: SessionSet) -> None:
        self.n_ = self._select_n(train)
        self.config["n_components_selected"] = self.n_
        self.pipe = self._fit_pipeline(train, self.n_)
        if self.use_plda:
            self.plda = TwoCovPLDA().fit(self._embed(self.pipe, train), train.pids)

    def embed(self, sessions: SessionSet) -> np.ndarray:
        return self._embed(self.pipe, sessions)

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        p, g = self.embed(probe), self.embed(gallery)
        return self.plda.score(p, g) if self.use_plda else cosine(p, g)
