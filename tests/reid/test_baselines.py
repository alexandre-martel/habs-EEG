"""Baselines on synthetic cached features: identity is found when present,
chance when absent."""

import numpy as np
import pandas as pd
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


def test_plda_score_matches_brute_force_llr() -> None:
    from scipy.stats import multivariate_normal as mvn

    from reid.baselines.b3_lda_plda import TwoCovPLDA

    rng = np.random.default_rng(0)
    d, n_people = 4, 400
    y = rng.normal(size=(n_people, d)) * [2.0, 1.0, 0.5, 0.2]
    X = np.repeat(y, 3, axis=0) + rng.normal(size=(3 * n_people, d)) * [
        0.3,
        0.5,
        1.0,
        1.5,
    ]
    plda = TwoCovPLDA(length_norm=False).fit(X, np.repeat(np.arange(n_people), 3))
    x1, x2 = X[:2] - plda.mu, X[3:5] - plda.mu
    T, B = plda.B + plda.W, plda.B
    joint = mvn(np.zeros(2 * d), np.block([[T, B], [B, T]]))
    for a in x1:
        for b in x2:
            brute = (
                joint.logpdf(np.r_[a, b])
                - mvn(np.zeros(d), T).logpdf(a)
                - mvn(np.zeros(d), T).logpdf(b)
            )
            got = plda.score((a + plda.mu)[None], (b + plda.mu)[None])[0, 0]
            assert got == pytest.approx(brute, rel=1e-6, abs=1e-6)


def test_plda_beats_cosine_with_anisotropic_session_noise() -> None:
    """Session noise concentrated where people do not differ: PLDA learns to
    ignore it, cosine cannot."""
    from reid.baselines.b3_lda_plda import TwoCovPLDA
    from reid.eval.interface import cosine
    from reid.eval.matching import rank_counts, topk

    rng = np.random.default_rng(1)
    d = 10
    person_sd = np.r_[np.ones(5), 0.05 * np.ones(5)]
    session_sd = np.r_[0.3 * np.ones(5), 3.0 * np.ones(5)]
    y_train = rng.normal(size=(300, d)) * person_sd
    X_train = np.repeat(y_train, 2, axis=0) + rng.normal(size=(600, d)) * session_sd
    plda = TwoCovPLDA(length_norm=False).fit(X_train, np.repeat(np.arange(300), 2))
    y = rng.normal(size=(50, d)) * person_sd
    g = y + rng.normal(size=y.shape) * session_sd
    p = y + rng.normal(size=y.shape) * session_sd
    truth = np.arange(50)
    acc = {
        name: topk(*rank_counts(S, truth), 1).mean()
        for name, S in (("plda", plda.score(p, g)), ("cosine", cosine(p, g)))
    }
    assert acc["plda"] > acc["cosine"] + 0.2, acc


@pytest.mark.parametrize("plda", [False, True])
def test_b3_lda_finds_identity_despite_day_gain(plda: bool) -> None:
    from reid.baselines.b3_lda_plda import ALL, LdaModel

    def model():
        return LdaModel(ALL, n_components=(16, 32), plda=plda)

    acc, chance = top1(model, identity=0.6, day_gain=0.6)
    b1, _ = top1(lambda: CorrModel(eeg_kind="abs"), identity=0.6, day_gain=0.6)
    assert acc > 0.7 and acc >= b1, (acc, b1)
    acc, chance = top1(model, identity=0.0)
    assert acc < chance + 0.15


def test_supcon_uses_cross_day_positives_only() -> None:
    import torch
    import torch.nn.functional as F

    from reid.baselines.b5_siamese import supcon_loss

    person = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    day = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2])
    a, b, c, d = torch.eye(4)
    # aligned across days within each person -> low loss
    good = F.normalize(torch.stack([a, a, a, a, b, b, b, b]), dim=1)
    # each person's two days point in different directions (session, not person)
    bad = F.normalize(torch.stack([a, a, c, c, b, b, d, d]), dim=1)
    assert supcon_loss(good, person, day) < supcon_loss(bad, person, day)


def test_b5_siamese_finds_identity() -> None:
    from reid.baselines.b3_lda_plda import ALL
    from reid.baselines.b5_siamese import SiameseModel

    def model():
        return SiameseModel(ALL, epochs=60, patience=60, people_per_batch=16)

    acc, _chance = top1(model, identity=1.0)
    assert acc > 0.7, acc


def test_ablations_and_their_figures(tmp_path) -> None:
    import json

    from reid.baselines.b4_ablations import run
    from reid.eval.report import write_report

    inv, store = make_feature_dataset(identity=1.0)
    folds = make_folds(inv, n_folds=3)["folds"]
    records = []
    for kind in ("modality", "task", "duration"):
        records += run(kind, "b1", inv, folds, store, n_boot=50, save=False)
    labels = {r["label"] for r in records}
    assert {"only_eeg", "minus_eeg", "rest_ec", "rest_eo", "rest_10s"} <= labels
    assert "video" not in labels  # no video in the synthetic data: skipped
    short = next(r for r in records if r["label"] == "rest_10s")
    full = next(r for r in records if r["label"] == "rest_60s")
    assert short["pooled"]["top1"]["value"] <= full["pooled"]["top1"]["value"] + 0.05
    (tmp_path / "runs").mkdir()
    for i, r in enumerate(records):
        (tmp_path / "runs" / f"{i}.json").write_text(json.dumps(r, default=float))
    write_report(tmp_path)
    names = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {
        "ablation_modality_b1.png",
        "ablation_duration_b1.png",
        "ablation_task_b1.png",
    } <= names


def test_arousal_split_parses_float_days():
    """labels_train.csv stores day as 1.0 / 2.0 / NaN (regression: 1.0 -> 10)."""
    from reid.baselines.b4_ablations import arousal_split

    labels = pd.DataFrame(
        {
            "participant_id": ["P1"] * 4 + ["P1"] * 2 + ["P2"],
            "day": [1.0, 1.0, 1.0, 1.0, "day-2", "day-2", np.nan],
            "start_marker": [10100, 10200, 10300, 10400, 10100, 10200, 10100],
            "rating_arousal": [1, 2, 8, 9, 3, 7, 5],
        }
    )
    split = arousal_split(labels)
    assert set(split["arousal_low"]) == {("P1", 1), ("P1", 2)}
    assert split["arousal_low"][("P1", 1)] == {10100, 10200}
    assert split["arousal_high"][("P1", 1)] == {10300, 10400}
    assert split["arousal_high"][("P1", 2)] == {10200}
