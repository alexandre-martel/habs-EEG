"""The common model interface and two generic adapters.

Every model of the team (matching baselines, per-person classifiers, world-model
embeddings) is evaluated through ``ReIDModel``: ``fit`` on training sessions,
then ``score`` a probe set against a gallery set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from reid.eval.protocol import SessionSet


class ReIDModel(Protocol):
    name: str

    def fit(self, train: SessionSet) -> None: ...

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        """Matrix S[n_probe, n_gallery]: larger = more likely the same person."""
        ...


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a and rows of b (NaN-safe: NaN -> 0)."""
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


class EmbeddingModel:
    """Base class for embedding models: implement ``embed`` (and optionally
    ``fit``); scores are cosine similarities between session embeddings."""

    name = "embedding"

    def fit(self, train: SessionSet) -> None:
        """Default: nothing to learn."""

    def embed(self, sessions: SessionSet) -> np.ndarray:
        """One vector per session, shape [n_sessions, d]."""
        raise NotImplementedError

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        return cosine(self.embed(probe), self.embed(gallery))


class ClassifierModel:
    """Adapter for a one-class-per-person classifier.

    At ``score`` time the classifier is trained on the gallery windows
    (enrolment, one class per gallery session), then the probability of each
    gallery class is averaged over the probe's windows.

    Parameters
    ----------
    name : model name used in run files.
    make_classifier : returns a fresh scikit-learn-like estimator with
        ``fit(X, y)``, ``predict_proba(X)`` and ``classes_``.
    windows : maps a session's feature dict to a [n_windows, d] matrix.
    pretrain : optional hook called with the training sessions (e.g. to fit a
        feature extractor shared across people).
    """

    def __init__(
        self,
        name: str,
        make_classifier: Callable[[], object],
        windows: Callable[[dict], np.ndarray],
        pretrain: Callable[[SessionSet], None] | None = None,
    ) -> None:
        self.name, self.make_classifier = name, make_classifier
        self.windows, self.pretrain = windows, pretrain

    def fit(self, train: SessionSet) -> None:
        if self.pretrain is not None:
            self.pretrain(train)

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        feats = [self.windows(f) for f in gallery]
        X = np.vstack(feats)
        y = np.concatenate([np.full(len(f), j) for j, f in enumerate(feats)])
        clf = self.make_classifier()
        clf.fit(X, y)
        S = np.full((len(probe), len(gallery)), -np.inf)
        for i, f in enumerate(probe):
            proba = clf.predict_proba(self.windows(f)).mean(axis=0)
            S[i, clf.classes_.astype(int)] = proba
        return S
