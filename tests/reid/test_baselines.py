"""Baselines on synthetic cached features: identity is found when present,
chance when absent."""

import numpy as np
import pytest
from synthetic import make_feature_dataset

from reid.baselines.b1_spectral_corr import CorrModel
from reid.eval.protocol import make_folds
from reid.eval.runner import evaluate


def top1(model, identity: float, **kwargs) -> tuple[float, float]:
    inv, store = make_feature_dataset(identity=identity, **kwargs)
    folds = make_folds(inv, n_folds=3)["folds"]
    run = evaluate(
        model, inv, folds, store=store, n_boot=0, run_controls=False, save=False
    )
    return run["pooled"]["top1"]["value"], run["pooled"]["chance_top1"]["value"]


@pytest.mark.parametrize("kind", ["abs", "rel", "bands"])
def test_b1_spectral_variants(kind: str) -> None:
    acc, chance = top1(lambda: CorrModel(eeg_kind=kind), identity=1.0)
    assert acc > 0.8, (kind, acc)
    acc, chance = top1(lambda: CorrModel(eeg_kind=kind), identity=0.0)
    assert acc < chance + 0.15


def test_b1_relative_spectrum_cancels_day_gain() -> None:
    gain = {"day_gain": 1.5, "noise": 0.2}
    abs_acc, _ = top1(lambda: CorrModel(eeg_kind="abs"), identity=0.4, **gain)
    rel_acc, _ = top1(lambda: CorrModel(eeg_kind="rel"), identity=0.4, **gain)
    assert rel_acc > abs_acc


def test_b1_other_modalities() -> None:
    acc, _ = top1(lambda: CorrModel(modalities=("ecg_beat",)), identity=1.0)
    assert acc > 0.8
    assert np.isfinite(acc)


@pytest.mark.parametrize(
    ("metric", "bands"),
    [
        ("tangent", ("broad",)),
        ("airm", ("broad",)),
        ("tangent", ("theta", "alpha", "beta")),
    ],
)
def test_b2_riemann_finds_identity(metric: str, bands: tuple) -> None:
    from reid.baselines.b2_riemann import RiemannModel

    acc, chance = top1(lambda: RiemannModel(metric, bands), identity=1.0, day_gain=0.1)
    assert acc > 0.8, (metric, bands, acc)
    acc, chance = top1(lambda: RiemannModel(metric, bands), identity=0.0)
    assert acc < chance + 0.15


def test_b2_recentering_erases_identity() -> None:
    from reid.baselines.b2_riemann import RiemannModel

    plain, _ = top1(lambda: RiemannModel("tangent"), identity=1.0)
    recentered, chance = top1(
        lambda: RiemannModel("tangent", recenter=True), identity=1.0
    )
    assert recentered < plain and recentered < chance + 0.2


def test_b2_tangent_resists_day_gain_better_than_airm() -> None:
    """AIRM weighs every direction equally; the tangent z-score learns to
    down-weight directions that vary with per-day electrode gains."""
    from reid.baselines.b2_riemann import RiemannModel

    airm, _ = top1(lambda: RiemannModel("airm"), identity=1.0, day_gain=0.3)
    tangent, _ = top1(lambda: RiemannModel("tangent"), identity=1.0, day_gain=0.3)
    assert tangent > airm
