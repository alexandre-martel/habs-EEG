"""Evaluation pipeline on synthetic data."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from synthetic import make_dataset

from reid.eval import controls, matching, metrics
from reid.eval.interface import ClassifierModel, EmbeddingModel
from reid.eval.protocol import build_split, make_folds
from reid.eval.runner import evaluate


class MeanEmbedding(EmbeddingModel):
    name = "test_mean"

    def embed(self, sessions):
        return np.array([f["w_x"].mean(axis=0) for f in sessions])


class SpyModel(MeanEmbedding):
    """Records every participant seen by fit."""

    seen: ClassVar[list] = []

    def fit(self, train):
        SpyModel.seen.append(set(train.pids))


def _run(identity: float, scenario: str = "d1_to_d2", model=MeanEmbedding):
    inv, store = make_dataset(identity=identity)
    folds = make_folds(inv, n_folds=3)["folds"]
    return evaluate(model, inv, folds, scenario, store=store, n_boot=200, save=False)


def test_perfect_embeddings_give_top1_of_one() -> None:
    pooled = _run(identity=1.0)["pooled"]
    assert pooled["top1"]["value"] == 1.0
    assert pooled["hungarian"]["value"] == 1.0
    assert pooled["eer"]["value"] < 0.01


def test_random_embeddings_give_chance() -> None:
    run = _run(identity=0.0)
    top1, chance = run["pooled"]["top1"], run["pooled"]["chance_top1"]["value"]
    assert top1["ci"][0] <= chance <= top1["ci"][1]
    shuffled = _run(identity=1.0)["controls"]["shuffled"]["top1"]
    assert shuffled["ci"][0] <= chance <= shuffled["ci"][1]


def test_hungarian_beats_nearest_neighbour_on_constructed_case() -> None:
    # Both probes prefer gallery 0; one-to-one assignment resolves the conflict.
    S = np.array([[0.9, 0.1], [0.8, 0.7]])
    block = metrics.Block(S, np.array([0, 1]), one_to_one=True)
    out = metrics.per_probe(block)
    assert out["top1"].mean() == 0.5 and out["hungarian"].mean() == 1.0


def test_fit_never_sees_test_people() -> None:
    SpyModel.seen = []
    inv, store = make_dataset()
    folds = make_folds(inv, n_folds=3)["folds"]
    evaluate(
        SpyModel, inv, folds, store=store, n_boot=0, run_controls=False, save=False
    )
    for fold, seen in zip(folds, SpyModel.seen, strict=True):
        split = build_split(inv, fold, "d1_to_d2", store=store)
        assert not seen & set(split.probe.pids)
        assert not seen & set(split.gallery.pids)


def test_ties_are_broken_at_chance() -> None:
    S = np.zeros((4, 4))
    greater, ties = matching.rank_counts(S, np.arange(4))
    assert matching.topk(greater, ties, 1).mean() == pytest.approx(0.25)


def test_scenarios_and_controls_run() -> None:
    inv, store = make_dataset(identity=1.0)
    folds = make_folds(inv, n_folds=3)["folds"]
    for scenario in ("within_cohort", "with_distractors", "same_day_ceiling"):
        run = evaluate(
            MeanEmbedding, inv, folds, scenario, store=store, n_boot=50, save=False
        )
        assert run["pooled"]["top1"]["value"] > 0.9, scenario
    distract = build_split(inv, folds[0], "with_distractors", store=store)
    assert len(distract.gallery) > len(distract.probe)
    assert set(run["controls"]) >= {"shuffled", "demographics", "cohort_prediction"}
    S = controls.mask_cohorts(np.ones((2, 2)), np.array([1, 2]), np.array([1, 1]))
    assert np.isinf(S[:, 1]).all()


def test_classifier_adapter() -> None:
    from sklearn.linear_model import LogisticRegression

    model = lambda: ClassifierModel(
        "test_logreg", lambda: LogisticRegression(max_iter=500), lambda f: f["w_x"]
    )
    inv, store = make_dataset(identity=1.0, noise=0.5)
    folds = make_folds(inv, n_folds=3)["folds"]
    run = evaluate(model, inv, folds, store=store, n_boot=0, save=False)
    assert run["pooled"]["top1"]["value"] > 0.9


def test_report_from_runs(tmp_path) -> None:
    import json

    from reid.eval.report import write_report

    (tmp_path / "runs").mkdir()
    for identity in (1.0, 0.3):
        run = _run(identity)
        run["model"] = f"test_{identity}"
        (tmp_path / "runs" / f"{identity}.json").write_text(
            json.dumps(run, default=float)
        )
    board = write_report(tmp_path)
    assert len(board) == 2 and (tmp_path / "leaderboard.csv").exists()
    assert {p.name for p in (tmp_path / "figures").iterdir()} == {
        "top1_d1_to_d2_rest.png",
        "gallery_size.png",
        "roc_scores.png",
    }


def test_make_folds_without_single_day_people() -> None:
    inv, _ = make_dataset(n_multi=12, n_single=0)
    folds = make_folds(inv, n_folds=3)["folds"]
    assert sorted(p for f in folds for p in f["test"]) == sorted(
        inv.participant_id.unique()
    )
    assert all(f["distractors"] == [] for f in folds)
    inv, _ = make_dataset(n_multi=12, n_single=2)
    assert sum(len(f["distractors"]) for f in make_folds(inv, 3)["folds"]) == 2


def test_demographic_control_with_missing_gender():
    """Real tables have a few missing genders (pd.NA): no crash, no penalty."""
    from types import SimpleNamespace

    from reid.eval.controls import DemographicModel

    g = pd.DataFrame(
        {"age": [20, 30, np.nan], "gender": pd.array(["F", pd.NA, "M"], dtype="string")}
    )
    p = pd.DataFrame({"age": [21, 30], "gender": pd.array(["M", "F"], dtype="string")})
    S = DemographicModel().score(
        SimpleNamespace(sessions=g), SimpleNamespace(sessions=p)
    )
    np.testing.assert_array_equal(S, [[-11, -9, 0], [-10, 0, -10]])


class FoldDependentDim(MeanEmbedding):
    """Embedding size changes from fold to fold, like the LDA's inner selection."""

    n_fits: ClassVar[int] = 0

    def fit(self, train):
        FoldDependentDim.n_fits += 1
        self.dim = 16 - FoldDependentDim.n_fits

    def embed(self, sessions):
        return super().embed(sessions)[:, : self.dim]


def test_embedding_size_may_change_between_folds() -> None:
    """Regression: the cohort control used to stack all folds' embeddings."""
    run = _run(identity=1.0, model=FoldDependentDim)
    assert run["pooled"]["top1"]["value"] == 1.0
    assert "cohort_prediction" in run["controls"]


def test_hungarian_ties_are_not_resolved_by_row_order() -> None:
    """Regression: with all-tied scores, the identity diagonal used to win."""
    n = 30
    S = np.zeros((n, n))
    acc = matching.hungarian_correct(S, np.arange(n)).mean()
    assert acc < 0.2  # expected 1/n
    S = np.eye(n)  # untied: always right
    assert matching.hungarian_correct(S, np.arange(n)).mean() == 1.0
