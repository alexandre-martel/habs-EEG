"""Light raw-signal cache for the final model: ``<out>/<task>/<PID>_day-<d>.npz``.

Each kept sub-segment ``k`` (eyes closed / eyes open rest, eyes-open blocks of
the artefact task; the 5 s instructed actions are artefacts and are left out;
for the emotion-video task, one ``video`` segment from the first clip's onset
to the last clip's end, ~14 min with the ratings in between, read in chunks
of ``CHUNK_S``):

======================  =====================================================
seg_condition, ...      condition, start and duration (s) of each sub-segment
eeg_<k>                 [62, T] µV float16, average reference, 1-45 Hz
                        band-pass (zero phase), resampled to ``EEG_RATE``,
                        clipped to +-60 mV (float16 range)
beats_<k>, beat_t_<k>   [n, 128] individual ECG beats (-250/+450 ms around
                        each R peak, cleaned, not normalised) and R times (s)
resp_<k>                [T] cleaned respiration at ``RESP_RATE``
eeg_reject_uv           peak-to-peak threshold of 2 s epochs (median + 5 MAD
                        over the session), used to drop noisy crops
ch_names, eeg_rate,     channel order and sampling rates
resp_rate, version
======================  =====================================================

About 1.5 MB per 90 s rest session, ~14 MB per video session (cache/ is
never committed).
"""

from __future__ import annotations

import csv
import itertools
import json
import traceback
import warnings
from fractions import Fraction
from pathlib import Path

import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly

from reid import preprocess
from reid.config import CACHE, PAD_S, TASKS
from reid.features.ecg import BEAT_POINTS, BEAT_SPAN_S
from reid.segments import Recording, open_recording, sub_segments

RAW = CACHE / "raw"
VERSION = 2  # 2: EEG clipped to the float16 range (12 sessions had > 65 mV spikes)
EEG_RATE = 128  # Hz: keeps 1-45 Hz with margin, 8x smaller than 1 kHz
RESP_RATE = 25  # Hz: breathing is < 1 Hz, 25 Hz keeps the waveform shape
RESP_PAD_S = 10.0
KEEP = ("rest_ec", "rest_eo", "art_eo")
CHUNK_S = 60.0  # EEG of long segments is read and filtered by chunks
F16_MAX = 6.0e4  # µV; such spikes fail the peak-to-peak test anyway


def resample(x: np.ndarray, sfreq: float, target: float) -> np.ndarray:
    """Polyphase resampling along the last axis, exact for 998/1000/1024 Hz."""
    ratio = Fraction(target / sfreq).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)


def _read(rec: Recording, modality: str, start: float, stop: float, pad_s: float):
    """Signal with up to ``pad_s`` of context, and the pad sizes in samples."""
    x = rec.read(modality, start - pad_s, stop + pad_s)
    lead = round(min(pad_s, start) * rec.sfreq)
    tail = max(x.shape[-1] - lead - round((stop - start) * rec.sfreq), 0)
    return x, (lead, tail)


def eeg_segment(rec: Recording, start: float, stop: float) -> np.ndarray:
    if stop - start > 1.5 * CHUNK_S:  # long: chunks, each filtered with context
        edges = np.append(np.arange(start, stop, CHUNK_S), stop)
        edges = edges[np.r_[np.diff(edges) > 0.5, True]]  # no sliver at the end
        return np.concatenate(
            [eeg_segment(rec, a, b) for a, b in itertools.pairwise(edges)], axis=-1
        )
    x, pad = _read(rec, "eeg", start, stop, PAD_S)
    y = preprocess.clean_window(x * 1e6, rec.sfreq, pad)  # V -> µV
    return resample(y, rec.sfreq, EEG_RATE).astype(np.float32)


def ecg_beats(rec: Recording, start: float, stop: float):
    """Individual beats [n, 128] and their R-peak times (s, segment-relative)."""
    empty = np.empty((0, BEAT_POINTS), np.float32), np.empty(0, np.float32)
    x, (lead, tail) = _read(rec, "ecg", start, stop, PAD_S)
    fs = rec.sfreq
    try:
        clean = nk.ecg_clean(x[0], sampling_rate=int(fs))
        peaks = np.asarray(nk.ecg_peaks(clean, sampling_rate=int(fs))[1]["ECG_R_Peaks"])
    except Exception:  # noqa: BLE001 - one bad segment must never stop extraction
        return empty
    lo, hi = (round(s * fs) for s in BEAT_SPAN_S)
    end = len(clean) - tail
    peaks = peaks[(peaks >= max(lead, -lo)) & (peaks < min(end, len(clean) - hi))]
    if not len(peaks):
        return empty
    grid = np.linspace(0, hi - lo - 1, BEAT_POINTS)
    beats = np.array(
        [np.interp(grid, np.arange(hi - lo), clean[p + lo : p + hi]) for p in peaks]
    )
    return beats.astype(np.float32), ((peaks - lead) / fs).astype(np.float32)


def resp_segment(rec: Recording, start: float, stop: float) -> np.ndarray:
    x, (lead, tail) = _read(rec, "resp", start, stop, RESP_PAD_S)
    y = resample(x[0], rec.sfreq, 100)
    try:
        y = nk.rsp_clean(y, sampling_rate=100)
    except Exception:  # noqa: BLE001
        return np.full(round((stop - start) * RESP_RATE), np.nan, np.float32)
    y = resample(y, 100, RESP_RATE)
    a = round(lead / rec.sfreq * RESP_RATE)
    b = len(y) - round(tail / rec.sfreq * RESP_RATE)
    return y[a:b].astype(np.float32)


def eeg_threshold(segments: list[np.ndarray]) -> float:
    ptp = np.concatenate(
        [preprocess.peak_to_peak(preprocess.epochs(s, EEG_RATE)) for s in segments]
    )
    return preprocess.rejection_threshold(ptp)


def extract_raw_session(bdf: Path, meta: dict, task: str) -> dict:
    """Cached raw signals of one task recording (``task`` as in file names)."""
    rec = open_recording(bdf)
    segs = [s for s in sub_segments(meta, task) if s[0] in KEEP]
    if task == "emotion-video":  # the clips and the ratings between them
        clips = sub_segments(meta, task)
        if clips:
            first = min(c[2] for c in clips)
            last = max(c[2] + c[3] for c in clips)
            segs = [("video", 0, first, last - first)]
    if not segs:
        raise ValueError("no kept sub-segment inside the recording")
    out = {
        "seg_condition": np.array([s[0] for s in segs]),
        "seg_start": np.array([s[2] for s in segs]),
        "seg_duration": np.array([s[3] for s in segs]),
        "eeg_rate": EEG_RATE,
        "resp_rate": RESP_RATE,
        "version": VERSION,
    }
    eeg = []
    for k, (_, _, start, duration) in enumerate(segs):
        stop = start + duration
        eeg.append(eeg_segment(rec, start, stop))
        out[f"eeg_{k}"] = np.clip(eeg[-1], -F16_MAX, F16_MAX).astype(np.float16)
        if "ecg" in rec.channels:
            out[f"beats_{k}"], out[f"beat_t_{k}"] = ecg_beats(rec, start, stop)
        if "resp" in rec.channels:
            out[f"resp_{k}"] = resp_segment(rec, start, stop)
    out["eeg_reject_uv"] = eeg_threshold(eeg)
    out["ch_names"] = np.array(rec.channels["eeg"])
    return out


def run_raw_extraction(
    data_dir: Path,
    out_dir: Path = RAW,
    task: str = "rest",
    shard: tuple[int, int] = (0, 1),
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    """Same contract as ``reid.features.run_extraction`` (resumable, failures
    logged to ``out_dir/failures.csv``, never raises for a session)."""
    file_task = {v: k for k, v in TASKS.items()}[task]
    bdfs = sorted(
        Path(data_dir).glob(f"CURATED/P*/P*_day-*_task-{file_task}_multimodal.bdf")
    )[shard[0] :: shard[1]][:limit]
    (out_dir / task).mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    warnings.filterwarnings("ignore", module="neurokit2")
    for bdf in bdfs:
        pid, day = bdf.name.split("_")[0], int(bdf.name.split("_")[1][4:])
        target = out_dir / task / f"{pid}_day-{day}.npz"
        if target.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            meta_path = bdf.parent / f"{pid}_day-{day}_metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            np.savez(target, **extract_raw_session(bdf, meta, file_task))
            counts["done"] += 1
        except Exception as err:  # noqa: BLE001 - logged, never stops the run
            counts["failed"] += 1
            with open(out_dir / "failures.csv", "a", newline="") as fh:
                csv.writer(fh).writerow(
                    [pid, day, task, repr(err), traceback.format_exc(limit=2)]
                )
    return counts
