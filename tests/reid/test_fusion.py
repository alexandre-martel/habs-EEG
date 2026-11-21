"""Score fusion: normalisation, weight search, members with and without their
own inner-validation scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reid.config import SEED
from reid.eval.protocol import SessionSet
from reid.final.fusion import ScoreFusion, simplex, znorm
from reid.final.model import inner_split


def test_znorm_and_simplex() -> None:
    Z = znorm(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [np.nan, 0.0, 1.0]]))
    assert np.allclose(Z[0].mean(), 0) and np.allclose(Z[0].std(), 1)
    assert np.all(Z[1] == 0) and np.isfinite(Z).all()
    W = simplex(3, 0.25)
    assert len(W) == 15 and np.allclose(W.sum(1), 1) and (W >= 0).all()


class Stub:
    """Scores = signal x (same person) + noise; ``own_val`` exposes the inner
    validation scores as the LDA and the encoders do."""

    def __init__(self, signal: float, own_val: bool, seed: int) -> None:
        self.signal, self.own_val, self.seed = signal, own_val, seed
        self.fits = 0

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        rng = np.random.default_rng([self.seed, len(gallery), len(probe)])
        same = probe.pids[:, None] == gallery.pids[None, :]
        return self.signal * same + rng.normal(size=same.shape)

    def embed(self, sessions: SessionSet) -> np.ndarray:
        return np.ones((len(sessions), 2))

    def fit(self, train: SessionSet) -> None:
        self.fits += 1
        if self.own_val:
            _, gal, prb = inner_split(train, SEED)
            self.val_scores_, self.val_pids_ = (
                self.score(gal, prb),
                (gal.pids, prb.pids),
            )


def test_fusion_weights_follow_the_informative_member(tmp_path) -> None:
    pids = [f"P{i:03d}" for i in range(200)]
    df = pd.DataFrame(
        {"participant_id": pids * 2, "day": [1] * 200 + [2] * 200, "task": "rest"}
    )
    train = SessionSet(df)
    stubs = {
        "good": Stub(2.5, own_val=True, seed=1),
        "noise": Stub(0.0, own_val=True, seed=2),
        "external": Stub(1.5, own_val=False, seed=3),
    }
    roots = {"a": tmp_path, "b": tmp_path}
    members = {
        "good": (lambda: stubs["good"], "a"),
        "noise": (lambda: stubs["noise"], "b"),
        "external": (lambda: Stub(1.5, own_val=False, seed=3), "a"),
    }
    fusion = ScoreFusion("t", members, roots=roots)
    fusion.fit(train)
    w = fusion.config["weights"]
    assert w["noise"] <= 0.2 < w["good"], w
    assert fusion.config["val_member_top1"]["good"] > 0.5
    # good/noise trained once (on everyone), external also on the inner people
    assert stubs["good"].fits == 1
    S = fusion.score(
        train.subset((df.day == 1).to_numpy()), train.subset((df.day == 2).to_numpy())
    )
    assert (S.argmax(1) == np.arange(200)).mean() > 0.3  # chance 0.005
    assert fusion.embed(train).shape == (400, 6)
    eq = ScoreFusion("e", members, weights="equal", roots=roots)
    eq.fit(train)
    assert set(eq.config["weights"].values()) == {round(1 / 3, 3)}
