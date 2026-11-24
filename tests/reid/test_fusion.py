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


def test_member_scores_are_dumped_without_ids(tmp_path, monkeypatch) -> None:
    """REID_SCORES: raw member scores + truth index, val at fit and test at
    score, one fold id shared by every fusion of the same training set."""
    monkeypatch.setenv("REID_SCORES", str(tmp_path / "scores"))
    pids = [f"P{i:03d}" for i in range(60)]
    df = pd.DataFrame(
        {"participant_id": pids * 2, "day": [1] * 60 + [2] * 60, "task": "rest"}
    )
    train = SessionSet(df)
    members = {"good": (lambda: Stub(2.5, own_val=True, seed=1), "a")}
    a = ScoreFusion("fa", members, weights="equal", roots={"a": tmp_path})
    b = ScoreFusion("fb", members, weights="equal", roots={"a": tmp_path})
    a.fit(train)
    b.fit(train)
    assert a.fold_ == b.fold_
    gal, prb = (train.subset((df.day == d).to_numpy()) for d in (1, 2))
    S = a.score(gal, prb)
    val = np.load(tmp_path / "scores" / f"fa_{a.fold_}_val.npz")
    test = np.load(tmp_path / "scores" / f"fa_{a.fold_}_test.npz")
    assert set(test.files) == {"truth", "S:good"}
    assert np.array_equal(test["truth"], np.arange(60))
    assert np.allclose(znorm(test["S:good"]), S, atol=1e-4)
    assert val["S:good"].shape == (len(val["truth"]), len(val["truth"]))
    assert not any(str(v).startswith("P") for v in val["truth"])


class FakeStore:
    """Only ``available`` matters to TaskView: video missing for some sessions."""

    def __init__(self, missing) -> None:
        self.missing = set(missing)

    def available(self, pid, day, task) -> bool:
        return (pid, day) not in self.missing


class TaskStub(Stub):
    def fit(self, train: SessionSet) -> None:
        assert set(train.sessions.task) == {"video"}
        self.seen = train
        _, gal, prb = inner_split(train, SEED, getattr(self, "val_people", None))
        self.val_scores_, self.val_pids_ = self.score(gal, prb), (gal.pids, prb.pids)


def test_task_view_aligns_with_the_rest_split() -> None:
    from reid.final.fusion import TaskView

    pids = [f"P{i:03d}" for i in range(100)]
    df = pd.DataFrame(
        {"participant_id": pids * 2, "day": [1] * 100 + [2] * 100, "task": "rest"}
    )
    missing = {("P003", 2), ("P010", 1), ("P050", 1), ("P050", 2)}
    train = SessionSet(df, FakeStore(missing))
    _, gal, prb = inner_split(train, SEED)
    stub = TaskStub(3.0, own_val=True, seed=4)
    view = TaskView(lambda: stub, "video")
    view.fit(train)
    # trained on the video sessions that exist, never on rest
    assert len(stub.seen) == 200 - len(missing)
    # validation: the rest split's people, NaN where a video is missing
    assert view.val_pids_[0] is gal.pids or np.array_equal(view.val_pids_[0], gal.pids)
    assert view.val_scores_.shape == (len(prb), len(gal))
    # people with both video days have scores against every gallery video;
    # the others (one video day only) are NaN, neutral in the fusion
    both = [(p, 1) not in missing and (p, 2) not in missing for p in prb.pids]
    has1 = [(p, 1) not in missing for p in gal.pids]
    V = view.val_scores_
    assert np.isfinite(V[np.ix_(both, both)]).all()
    assert np.isnan(V[~np.array(both)]).all()
    assert np.isnan(V[:, ~np.array(has1)]).all()
    # validation people never trained on (inner split imposed)
    assert not set(stub.val_pids_[0]) - set(gal.pids)
    day1 = train.subset((df.day == 1).to_numpy())
    day2 = train.subset((df.day == 2).to_numpy())
    S = view.score(day1, day2)
    assert S.shape == (100, 100)
    assert np.isnan(S[3]).all() and np.isnan(S[:, 10]).all()
    Z = znorm(S)
    assert np.isfinite(Z).all() and (Z[3] == 0).all() and (Z[:, 10] == 0).all()
    keep = np.isfinite(np.diag(S))
    assert (np.nanargmax(np.where(np.isnan(S), -np.inf, S), 1) == np.arange(100))[
        keep
    ].mean() > 0.5


def test_znorm_ignores_missing_scores() -> None:
    S = np.array([[1.0, 2.0, np.nan, 3.0]])
    Z = znorm(S)
    assert Z[0, 2] == 0 and np.allclose(Z[0, [0, 1, 3]], znorm(S[:, [0, 1, 3]])[0])
