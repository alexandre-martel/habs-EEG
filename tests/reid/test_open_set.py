"""Open-set simulation (scripts/reid_open_set.py) on synthetic score matrices."""

import numpy as np

from reid.final.fusion import znorm
from scripts.reid_open_set import auc, collect, fused, thresholds


def _fold(n=20, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.permutation(n)
    out = {"truth": truth}
    for m in ("a", "b"):
        S = rng.normal(size=(n, n))
        S[np.arange(n), truth] += signal
        out[m] = S
    return out


def test_absent_removes_own_column_and_renormalises():
    fold = _fold()
    F = fused(fold, [["a"], ["b"]], absent=True)
    t = fold["truth"]
    assert np.isneginf(F[np.arange(len(t)), t]).all()
    # same as normalising the matrices without the column in the first place
    rows = []
    for i, j in enumerate(t):
        keep = np.arange(len(t)) != j
        z = [znorm(znorm(fold[m][i : i + 1, keep])) for m in ("a", "b")]
        rows.append(np.mean(z, 0)[0])
    for i, j in enumerate(t):
        np.testing.assert_allclose(np.delete(F[i], j), rows[i])


def test_present_scores_above_absent_with_signal_not_without():
    folds = {k: _fold(seed=k) for k in range(3)}
    d = collect(folds, [["a"], ["b"]])
    assert auc(d["present"]["top"], d["absent"]["top"]) > 0.9
    assert d["present"]["correct"].mean() > 0.9
    noise = {k: _fold(signal=0.0, seed=k) for k in range(3)}
    d0 = collect(noise, [["a"], ["b"]])
    assert 0.3 < auc(d0["present"]["top"], d0["absent"]["top"]) < 0.7
    thr = thresholds(d, "top")
    assert thr["fpir1"] >= thr["fpir5"]


def test_auc_ties_count_half():
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert auc(np.array([2.0, 3.0]), np.array([1.0])) == 1.0
