"""Final model: raw cache, crops, and the single-modality contrastive encoders."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import DictStore

from reid.config import REPO, SEED
from reid.eval.protocol import SessionSet, build_split
from reid.final.crops import BeatCrops, EEGCrops, RespCrops
from reid.final.raw import EEG_RATE, RESP_RATE, extract_raw_session, resample

SAMPLE = REPO / "sample"


@pytest.mark.skipif(not (SAMPLE / "CURATED").is_dir(), reason="no sample/")
def test_raw_extraction_on_sample() -> None:
    meta = json.loads((SAMPLE / "CURATED/P001/P001_day-1_metadata.json").read_text())
    bdf = SAMPLE / "CURATED/P001/P001_day-1_task-resting-state_multimodal.bdf"
    raw = extract_raw_session(bdf, meta, "resting-state")
    assert list(raw["seg_condition"]) == ["rest_eo", "rest_ec"]
    for k, seconds in enumerate((30, 60)):
        assert raw[f"eeg_{k}"].shape == (62, seconds * EEG_RATE)
        assert raw[f"eeg_{k}"].dtype == np.float16
        assert np.isfinite(raw[f"eeg_{k}"]).all()
        assert abs(len(raw[f"resp_{k}"]) - seconds * RESP_RATE) <= 1
        bpm = 60 * len(raw[f"beat_t_{k}"]) / seconds
        assert 40 < bpm < 140 and raw[f"beats_{k}"].shape[1] == 128


@pytest.mark.parametrize("sfreq", [998.0, 1000.0, 1024.0])
def test_resample_to_fixed_rate(sfreq: float) -> None:
    x = np.sin(2 * np.pi * 10 * np.arange(int(10 * sfreq)) / sfreq)
    y = resample(x, sfreq, EEG_RATE)
    assert abs(len(y) - 10 * EEG_RATE) <= 1


def test_eeg_crops_avoid_artefacts() -> None:
    rng = np.random.default_rng(0)
    seg = rng.normal(size=(4, 20 * EEG_RATE)).astype(np.float16)
    seg[1, 10 * EEG_RATE] = 500.0  # spike at 10 s
    feats = {
        "seg_condition": np.array(["rest_ec"]),
        "eeg_0": seg,
        "eeg_reject_uv": 20.0,
    }
    crops = EEGCrops(crop_s=4.0)
    pool = crops.prepare(feats, ("rest_ec",))
    starts = pool[0][1] / EEG_RATE
    # every kept crop [s, s + 4) avoids the spike
    assert len(starts) and not ((starts <= 10) & (starts + 4 > 10)).any()
    assert crops.sample(pool, 5, rng).shape == (5, 4, 4 * EEG_RATE)
    assert np.abs(crops.all_crops(pool)).max() < 100
    assert crops.prepare(feats, ("rest_eo",)) is None


def test_beat_crops_drop_outliers_and_normalise() -> None:
    rng = np.random.default_rng(0)
    template = np.sin(np.linspace(0, 3 * np.pi, 128)) ** 3
    beats = 50 * template + rng.normal(scale=2, size=(40, 128))
    beats[::10] = rng.normal(size=(4, 128))  # 4 noise "beats"
    feats = {
        "seg_condition": np.array(["rest_ec"]),
        "beats_0": beats.astype(np.float32),
    }
    crops = BeatCrops(n_beats=8, stride=4)
    pool = crops.prepare(feats, ("rest_ec",))
    assert len(pool) == 36
    np.testing.assert_allclose(pool.std(1), 1, atol=1e-3)
    assert crops.all_crops(pool).shape == (8, 8, 128)


def test_resp_crops_shape() -> None:
    feats = {
        "seg_condition": np.array(["rest_ec", "rest_eo"]),
        "resp_0": np.sin(np.arange(60 * RESP_RATE) / 10).astype(np.float32),
        "resp_1": np.full(10 * RESP_RATE, np.nan, np.float32),  # unusable
    }
    crops = RespCrops(crop_s=20.0, stride_s=5.0)
    pool = crops.prepare(feats, ("rest_ec", "rest_eo"))
    assert len(pool) == 1 and crops.all_crops(pool).shape == (9, 20 * RESP_RATE)


def make_raw_dataset(n_multi=30, n_single=6, n_ch=8, seconds=20, seed=SEED):
    """Each person: an alpha frequency, an alpha topography and a beat shape;
    each day: per-electrode gains; each sample: noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * EEG_RATE) / EEG_RATE
    rows, data = [], {}
    for k in range(n_multi + n_single):
        pid = f"P{k:03d}"
        f_alpha = rng.uniform(8, 12)
        topo = rng.uniform(0.5, 3.0, size=(n_ch, 1))
        beat = np.convolve(rng.normal(size=128), np.ones(9) / 9, mode="same")
        for day in (1, 2) if k < n_multi else (1,):
            rows.append(
                {
                    "participant_id": pid,
                    "day": day,
                    "has_rest": True,
                    "has_artefact": False,
                    "has_video": False,
                    "cohort": 1,
                    "age": 30.0,
                    "gender": "F",
                }
            )
            gain = np.exp(0.1 * rng.normal(size=(n_ch, 1)))
            phase = rng.uniform(0, 2 * np.pi, size=(n_ch, 1))
            eeg = gain * (
                topo * np.sin(2 * np.pi * f_alpha * t + phase)
                + rng.normal(size=(n_ch, len(t)))
            )
            data[(pid, day, "rest")] = {
                "seg_condition": np.array(["rest_ec"]),
                "eeg_0": eeg.astype(np.float16),
                "eeg_reject_uv": 100.0,
                "beats_0": (beat + 0.3 * rng.normal(size=(30, 128))).astype(np.float32),
            }
    return pd.DataFrame(rows), DictStore(data)


@pytest.mark.parametrize(
    ("modality", "refit"), [("eeg", False), ("ecg_beat", False), ("eeg", True)]
)
def test_contrastive_encoder_finds_identity(modality: str, refit: bool) -> None:
    from reid.eval.matching import rank_counts, topk, truth_index
    from reid.final.model import ContrastiveModel

    inv, store = make_raw_dataset()
    multi = sorted(inv.participant_id[inv.day == 2].unique())
    fold = {"test": multi[:10], "distractors": []}
    split = build_split(inv, fold, "d1_to_d2", store=store)
    model = ContrastiveModel(
        modality,
        conditions=("rest_ec",),
        steps=150,
        eval_every=50,
        people=12,
        negatives=4,
        device="cpu",
        refit=refit,
        pool_crops=refit,  # also covers session-level pooling
        encoder_kw={"n_channels": 8} if modality == "eeg" else {},
        crop_kw={"n_beats": 8, "stride": 4} if modality == "ecg_beat" else {},
    )
    model.fit(split.train)
    if refit:  # the refitted network has seen every training person
        assert len(model.multi) == len(
            set(split.train.pids[split.train.sessions.day == 2])
        )
    S = model.score(split.gallery, split.probe)
    truth = truth_index(split.gallery.pids, split.probe.pids)
    assert topk(*rank_counts(S, truth), 1).mean() >= 0.7  # chance: 0.1
    with pytest.raises(ValueError):  # window selection is not supported
        model.embed(SessionSet(split.probe.sessions.assign(part="first_10s"), store))
