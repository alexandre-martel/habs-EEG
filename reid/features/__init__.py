"""Per-session feature extraction into ``<out>/<task>/<PID>_day-<d>.npz``.

Keys of each file (``w_*`` = one row per 10 s analysis window):

======================  =====================================================
w_condition, w_marker   sub-segment name (rest_ec, rest_eo, art_eo, art_blink,
                        ..., video) and its event marker code
w_start, w_duration     window position in the task segment (s)
w_eeg_n_epochs          clean 2 s epochs in the window (after rejection)
w_eeg_psd               [win, 62, 89] mean PSD of clean epochs, µV²/Hz, 1-45 Hz
w_eeg_cov               [win, 4, 62, 62] Riemannian mean of OAS covariances
w_ecg, w_ecg_beat       [win, 3] HR/SDNN/RMSSD and [win, 128] mean beat
w_eda, w_resp           [win, 2] tonic/SCR rate and [win, 3] rate/amp CV/IE
freqs, ch_names         PSD frequencies and EEG channel order
cov_bands               covariance filter-bank band names
eeg_reject_uv           session peak-to-peak rejection threshold (µV)
eeg_pct_rejected        % of epochs rejected
======================  =====================================================
"""

from __future__ import annotations

import csv
import json
import traceback
import warnings
from pathlib import Path

import numpy as np

from reid import preprocess
from reid.config import COV_BANDS, MODALITIES, PAD_S, TASKS
from reid.features import ecg, eda, eeg_cov, eeg_psd, resp
from reid.segments import Recording, Window, open_recording, windows

VERSION = 1
SLOW_PAD_S = 10.0  # context for EDA and respiration (a 10 s window holds ~3 breaths)
SLOW = {"ecg": (ecg, PAD_S), "eda": (eda, SLOW_PAD_S), "resp": (resp, SLOW_PAD_S)}


def _read_padded(rec: Recording, modality: str, w: Window, pad_s: float):
    """Signal of a window with up to ``pad_s`` of context, and the pad sizes."""
    x = rec.read(modality, w.start_s - pad_s, w.start_s + w.duration_s + pad_s)
    lead = round(min(pad_s, w.start_s) * rec.sfreq)
    tail = max(x.shape[-1] - lead - round(w.duration_s * rec.sfreq), 0)
    return x, (lead, tail)


def _eeg(rec: Recording, wins: list[Window]) -> dict:
    """PSD and filter-bank covariances of clean epochs, window by window."""
    fs, per_window = rec.sfreq, []
    for w in wins:
        x, pad = _read_padded(rec, "eeg", w, PAD_S)
        x = x * 1e6  # V -> µV
        bands = [
            preprocess.epochs(preprocess.clean_window(x, fs, pad, b), fs)
            for b in COV_BANDS.values()
        ]
        freqs, psd = eeg_psd.epoch_psd(bands[0], fs)
        covs = np.stack([eeg_cov.epoch_covs(ep) for ep in bands])
        per_window.append((psd, covs, preprocess.peak_to_peak(bands[0])))
    threshold = preprocess.rejection_threshold(
        np.concatenate([p for *_, p in per_window])
    )
    out = {"w_eeg_psd": [], "w_eeg_cov": [], "w_eeg_n_epochs": []}
    for psd, covs, ptp in per_window:
        keep = ptp <= threshold
        out["w_eeg_psd"].append(
            psd[keep].mean(0) if keep.any() else np.full(psd.shape[1:], np.nan)
        )
        out["w_eeg_cov"].append([eeg_cov.riemann_mean(c[keep]) for c in covs])
        out["w_eeg_n_epochs"].append(int(keep.sum()))
    n_total = sum(len(p) for *_, p in per_window)
    return {
        "w_eeg_psd": np.asarray(out["w_eeg_psd"], dtype=np.float32),
        "w_eeg_cov": np.asarray(out["w_eeg_cov"], dtype=np.float32),
        "w_eeg_n_epochs": np.asarray(out["w_eeg_n_epochs"]),
        "freqs": freqs,
        "ch_names": np.array(rec.channels["eeg"]),
        "cov_bands": np.array(list(COV_BANDS)),
        "eeg_reject_uv": threshold,
        "eeg_pct_rejected": 100 * (1 - sum(out["w_eeg_n_epochs"]) / max(n_total, 1)),
    }


def _slow(rec: Recording, wins: list[Window], modality: str) -> dict:
    module, pad_s = SLOW[modality]
    rows, beats = [], []
    for w in wins:
        x, pad = _read_padded(rec, modality, w, pad_s)
        res = module.features(x[0], rec.sfreq, pad)
        if modality == "ecg":
            res, beat = res
            beats.append(beat)
        rows.append(res)
    out = {
        f"w_{modality}": np.asarray(rows, dtype=np.float32),
        f"{modality}_names": np.array(module.NAMES),
    }
    if beats:
        out["w_ecg_beat"] = np.asarray(beats, dtype=np.float32)
    return out


def extract_session(bdf: Path, meta: dict, task: str, modalities=MODALITIES) -> dict:
    """All features of one task recording (``task`` as in file names)."""
    rec = open_recording(bdf)
    wins = windows(meta, task)
    if not wins:
        raise ValueError("no analysis window inside the segment")
    out = {
        "w_condition": np.array([w.condition for w in wins]),
        "w_marker": np.array([w.marker for w in wins]),
        "w_start": np.array([w.start_s for w in wins]),
        "w_duration": np.array([w.duration_s for w in wins]),
        "version": VERSION,
    }
    if "eeg" in modalities:
        out |= _eeg(rec, wins)
    for modality in SLOW:
        if modality in modalities and modality in rec.channels:
            out |= _slow(rec, wins, modality)
    return out


def run_extraction(
    data_dir: Path,
    out_dir: Path,
    task: str = "rest",
    modalities=MODALITIES,
    shard: tuple[int, int] = (0, 1),
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    """Extract every session of ``task`` (short name: rest/artefact/video) found
    under ``data_dir/CURATED``. Never raises for a session: failures go to
    ``out_dir/failures.csv``. ``shard=(i, n)`` keeps every n-th session from i."""
    file_task = {v: k for k, v in TASKS.items()}[task]
    bdfs = sorted(
        Path(data_dir).glob(f"CURATED/P*/P*_day-*_task-{file_task}_multimodal.bdf")
    )
    bdfs = bdfs[shard[0] :: shard[1]][:limit]
    (out_dir / task).mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    warnings.filterwarnings("ignore", module="neurokit2")
    warnings.filterwarnings("ignore", message="Convergence not reached")
    for bdf in bdfs:
        pid, day = bdf.name.split("_")[0], int(bdf.name.split("_")[1][4:])
        target = out_dir / task / f"{pid}_day-{day}.npz"
        if target.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            meta = json.loads(
                (bdf.parent / f"{pid}_day-{day}_metadata.json").read_text()
            )
            feats = extract_session(bdf, meta, file_task, modalities)
            np.savez(target, participant_id=pid, day=day, task=task, **feats)
            counts["done"] += 1
        except Exception as err:  # noqa: BLE001 - log and continue with the next one
            counts["failed"] += 1
            _log_failure(out_dir / "failures.csv", pid, day, task, err)
        print(f"{task} {pid} day-{day}: {counts}", flush=True)
    return counts


def _log_failure(path: Path, pid: str, day: int, task: str, err: Exception) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["participant_id", "day", "task", "error", "where"])
        where = traceback.extract_tb(err.__traceback__)[-1]
        writer.writerow(
            [
                pid,
                day,
                task,
                f"{type(err).__name__}: {err}",
                f"{Path(where.filename).name}:{where.lineno}",
            ]
        )
