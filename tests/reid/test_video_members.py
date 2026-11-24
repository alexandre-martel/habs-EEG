"""Encoders on the video task of the same sessions, fused with rest members
(TaskView, reid.final.fusion), end to end on synthetic raw data."""

import numpy as np

from reid.config import SEED
from reid.eval.matching import rank_counts, topk, truth_index
from reid.eval.protocol import build_split
from reid.final.fusion import ScoreFusion, TaskView
from reid.final.model import ContrastiveModel
from tests.reid.test_final import make_raw_dataset

N_CH = 8


def test_video_member_fuses_with_rest_members():
    inv, store = make_raw_dataset(n_multi=30, n_single=4, n_ch=N_CH)
    rng = np.random.default_rng(SEED)
    for (pid, day, _), feats in list(store.data.items()):
        if (pid, day) == ("P002", 2):
            continue  # one session without video
        video = dict(feats)
        video["seg_condition"] = np.array(["video"])
        video["eeg_0"] = (
            np.tile(feats["eeg_0"].astype(np.float32), 2)
            + rng.normal(size=(N_CH, 2 * feats["eeg_0"].shape[1]))
        ).astype(np.float16)
        store.data[(pid, day, "video")] = video
    multi = sorted(inv.participant_id[inv.day == 2].unique())
    split = build_split(
        inv, {"test": multi[:6], "distractors": []}, "d1_to_d2", store=store
    )

    def enc(conditions):
        return lambda: ContrastiveModel(
            "eeg",
            conditions=conditions,
            steps=30,
            eval_every=10,
            people=8,
            negatives=2,
            device="cpu",
            encoder_kw={"n_channels": N_CH},
            refit=True,
            split_seed=SEED,
        )

    members = {
        "rest": (enc(("rest_ec",)), "raw"),
        "video": (lambda: TaskView(enc(("video",)), "video"), "raw"),
    }
    fusion = ScoreFusion("f", members, weights="equal", roots={})
    fusion.stores = {"raw": store}
    fusion.fit(split.train)
    assert set(fusion.config["val_member_top1"]) == {"rest", "video"}
    S = fusion.score(split.gallery, split.probe)
    assert S.shape == (len(split.probe), len(split.gallery)) and np.isfinite(S).all()
    top1 = topk(*rank_counts(S, truth_index(split.gallery.pids, split.probe.pids)), 1)
    assert top1.mean() > 1 / len(split.gallery)


def test_lda_inner_split_honours_imposed_validation_people(monkeypatch):
    import pandas as pd

    from reid.baselines.b3_lda_plda import LdaModel
    from reid.eval.protocol import SessionSet

    pids = [f"P{i:03d}" for i in range(40)]
    df = pd.DataFrame(
        {"participant_id": pids * 2, "day": [1] * 40 + [2] * 40, "task": "video"}
    )
    rng = np.random.default_rng(0)
    monkeypatch.setattr(LdaModel, "_fit_pipeline", lambda self, s, n: None)
    monkeypatch.setattr(
        LdaModel, "_embed", lambda self, pipe, s: rng.normal(size=(len(s), 4))
    )
    lda = LdaModel(("eeg",), conditions=("video",), n_components=(4, 8))
    lda.val_people = {"P001", "P007", "P099"}  # P099: not in this task's pool
    lda._select_n(SessionSet(df))
    assert set(lda.val_pids_[0]) == set(lda.val_pids_[1]) == {"P001", "P007"}
