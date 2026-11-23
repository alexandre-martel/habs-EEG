"""Video EEG cache and pretraining of the EEG encoder (reid/final/video.py,
reid/final/pretrain.py), on synthetic people."""

import json
from pathlib import Path

import numpy as np
import pytest

from reid.config import REPO, SEED
from reid.eval.protocol import build_split
from reid.final.raw import EEG_RATE
from reid.final.video import load_video, video_sessions
from tests.reid.test_final import make_raw_dataset

N_CH = 8


def write_video_cache(root: Path, pids, seconds=60, seed=SEED) -> None:
    """Synthetic video EEG with the same person signature as make_raw_dataset's
    rest (alpha frequency + topography), on days 1 and 2."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * EEG_RATE) / EEG_RATE
    (root / "video").mkdir(parents=True, exist_ok=True)
    for pid in pids:
        f_alpha, topo = rng.uniform(8, 12), rng.uniform(0.5, 3.0, size=(N_CH, 1))
        for day in (1, 2):
            phase = rng.uniform(0, 2 * np.pi, size=(N_CH, 1))
            eeg = topo * np.sin(2 * np.pi * f_alpha * t + phase) + rng.normal(
                size=(N_CH, len(t))
            )
            base = root / "video" / f"{pid}_day-{day}"
            np.save(base.with_suffix(".npy"), eeg.astype(np.float16))
            info = {"eeg_reject_uv": 100.0, "ch_names": [f"E{i}" for i in range(N_CH)]}
            base.with_suffix(".json").write_text(json.dumps(info))


def test_video_cache_is_memory_mapped_and_listed(tmp_path):
    write_video_cache(tmp_path, ["P001", "P002"], seconds=10)
    assert video_sessions(tmp_path) == [
        ("P001", 1),
        ("P001", 2),
        ("P002", 1),
        ("P002", 2),
    ]
    feats = load_video("P001", 2, tmp_path)
    assert isinstance(feats["eeg_0"], np.memmap) and feats["eeg_0"].shape == (
        N_CH,
        1280,
    )
    assert list(feats["seg_condition"]) == ["video"]
    assert load_video("P009", 1, tmp_path) is None


@pytest.mark.parametrize("objective", ["mae", "jepa", "session", "xday"])
def test_each_objective_trains_and_is_memoised(tmp_path, objective):
    import torch

    from reid.final.encoders import SpectralTokenEEG
    from reid.final.pretrain import pretrained

    write_video_cache(tmp_path / "v", [f"P{k:03d}" for k in range(8)])
    cfg = {
        "objective": objective,
        "steps": 30,
        "batch": 8,
        "encoder_kw": {"n_channels": N_CH},
    }
    pids = [f"P{k:03d}" for k in range(6)]  # P006, P007 must not be used
    state, info = pretrained(
        pids, cfg, "cpu", root=tmp_path / "v", cache=tmp_path / "c"
    )
    assert info["people"] == 6 and info["sessions"] == 12 and info["steps"] == 30
    assert np.isfinite(info["last"]["loss"])
    net = SpectralTokenEEG(n_channels=N_CH)
    net.load_state_dict(state)  # same architecture as the fine-tuned encoder
    assert torch.isfinite(net(torch.randn(2, N_CH, 4 * EEG_RATE))).all()
    _, again = pretrained(pids, cfg, "cpu", root=tmp_path / "v", cache=tmp_path / "c")
    assert again.get("memo") is True
    _, other = pretrained(
        pids[:5], cfg, "cpu", root=tmp_path / "v", cache=tmp_path / "c"
    )
    assert not other.get("memo")  # other people, other pretraining


def test_pretraining_sees_only_inner_training_people(tmp_path, monkeypatch):
    """Neither the test people nor the inner validation people of the fold may
    reach the pretraining (they decide early stopping and fusion weights)."""
    from reid.final import pretrain
    from reid.final.model import ContrastiveModel, inner_split

    inv, store = make_raw_dataset(n_multi=20, n_single=4, n_ch=N_CH)
    multi = sorted(inv.participant_id[inv.day == 2].unique())
    split = build_split(
        inv, {"test": multi[:5], "distractors": []}, "d1_to_d2", store=store
    )
    seen = {}

    def fake(pids, cfg, device, **kw):
        seen["pids"] = set(pids)
        return None, {"sessions": 0, "people": len(pids), "steps": 0, "seconds": 0.0}

    monkeypatch.setattr(pretrain, "pretrained", fake)
    model = ContrastiveModel(
        "eeg",
        conditions=("rest_ec",),
        steps=20,
        eval_every=10,
        people=8,
        negatives=2,
        device="cpu",
        encoder_kw={"n_channels": N_CH},
        pretrain={"objective": "session"},
    )
    model.fit(split.train)
    _, gal, prb = inner_split(split.train, model.split_seed)
    assert seen["pids"] and not seen["pids"] & set(multi[:5])
    assert not seen["pids"] & (set(gal.pids) | set(prb.pids))
    assert seen["pids"] <= set(split.train.pids)


def test_pretrained_encoder_starts_fine_tuning(tmp_path, monkeypatch):
    """End to end on synthetic data: xday pretraining on video already
    separates people on rest before any fine-tuning (chance 0.25 here)."""
    from reid.final import pretrain
    from reid.final.model import ContrastiveModel

    inv, store = make_raw_dataset(n_multi=24, n_single=0, n_ch=N_CH)
    rng = np.random.default_rng(1)
    (tmp_path / "video").mkdir()
    for pid, day in inv[["participant_id", "day"]].itertuples(index=False):
        # the person's own rest EEG (its signature), 3 times, with fresh noise
        rest = np.asarray(store.load(pid, day, "rest")["eeg_0"], np.float32)
        eeg = np.tile(rest, 3) + rng.normal(size=(N_CH, 3 * rest.shape[1]))
        base = tmp_path / "video" / f"{pid}_day-{day}"
        np.save(base.with_suffix(".npy"), eeg.astype(np.float16))
        base.with_suffix(".json").write_text(
            json.dumps({"eeg_reject_uv": 100.0, "ch_names": []})
        )
    real = pretrain.pretrained
    monkeypatch.setattr(
        pretrain,
        "pretrained",
        lambda p, c, d: real(p, c, d, root=tmp_path, cache=tmp_path / "c"),
    )
    multi = sorted(inv.participant_id[inv.day == 2].unique())
    split = build_split(
        inv, {"test": multi[:4], "distractors": []}, "d1_to_d2", store=store
    )
    model = ContrastiveModel(
        "eeg",
        conditions=("rest_ec",),
        steps=10,
        eval_every=10,
        people=8,
        negatives=0,
        device="cpu",
        encoder_kw={"n_channels": N_CH},
        pretrain={
            "objective": "xday",
            "steps": 150,
            "batch": 8,
            "encoder_kw": {"n_channels": N_CH},
        },
    )
    model.fit(split.train)
    assert model.config["pretrain_steps"] == 150
    assert model.config["val_top1_step0"] > 0.25


@pytest.mark.skipif(
    not (
        REPO / "data/CURATED/P001/P001_day-1_task-emotion-video_multimodal.bdf"
    ).is_file(),
    reason="video recording not downloaded",
)
def test_video_extraction_on_a_real_recording(tmp_path):
    from reid.final.video import run_video_extraction

    counts = run_video_extraction(REPO / "data", tmp_path, limit=1)
    assert counts == {"done": 1, "skipped": 0, "failed": 0}
    ((pid, day),) = video_sessions(tmp_path)
    feats = load_video(pid, day, tmp_path)
    x = np.asarray(feats["eeg_0"], np.float32)
    assert x.shape[0] == 62 and x.shape[1] > 600 * EEG_RATE
    assert np.isfinite(x).all() and 1 < np.median(np.abs(x)) < 50  # µV
    rest = np.load(REPO / "cache/raw/rest" / f"{pid}_day-{day}.npz")
    assert list(feats["ch_names"]) == list(rest["ch_names"])
