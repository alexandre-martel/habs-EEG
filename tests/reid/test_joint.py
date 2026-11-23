"""Joint EEG + ECG-beat + respiration model (reid/final/joint.py)."""

import numpy as np

from reid.eval.matching import rank_counts, topk, truth_index
from reid.eval.protocol import build_split
from tests.reid.test_final import make_raw_dataset


def test_joint_model_finds_identity_without_respiration():
    """Synthetic people carry EEG and beats but no respiration: the missing
    modality must be handled (presence flags), and identity found."""
    from reid.final.joint import JointModel

    inv, store = make_raw_dataset(n_multi=30, n_single=6, n_ch=8)
    multi = sorted(inv.participant_id[inv.day == 2].unique())
    split = build_split(
        inv, {"test": multi[:10], "distractors": []}, "d1_to_d2", store=store
    )
    model = JointModel(
        conditions=("rest_ec",),
        steps=150,
        eval_every=50,
        people=12,
        negatives=4,
        crops={"eeg": 2, "ecg_beat": 2},
        encoder_kw={"eeg": {"n_channels": 8}},
        crop_kw={"ecg_beat": {"n_beats": 8, "stride": 4}},
        device="cpu",
    )
    model.fit(split.train)
    assert model.config["refit"] and model.config["best_step"] > 0
    assert 0 <= model.config["val_top1_step0"] <= 1
    S = model.score(split.gallery, split.probe)
    truth = truth_index(split.gallery.pids, split.probe.pids)
    assert topk(*rank_counts(S, truth), 1).mean() >= 0.7  # chance 0.1
    E = model.embed(split.probe)
    assert E.shape == (len(split.probe), 64) and np.isfinite(E).all()
    assert np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4)
    gal, prb = model.val_pids_
    assert model.val_scores_.shape == (len(prb), len(gal))
