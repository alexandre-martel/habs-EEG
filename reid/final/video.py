"""EEG of the emotion-video task, for self-supervised pretraining of the EEG
encoder: ``<out>/video/<PID>_day-<d>.npy`` + ``.json``.

The whole exported video segment (~14 min: 36 clips of 10 s plus the ratings
between them) is kept as one continuous recording, preprocessed exactly like
the rest cache (``reid.final.raw``): average reference, 1-45 Hz zero-phase
band-pass, resampled to ``EEG_RATE``, float16 [62, T] in µV, clipped to the
float16 range. It is read and filtered in chunks of ``CHUNK_S`` with ``PAD_S``
of context on each side (filtered *and* resampled with the context, which is
then cut), so that a 1 kHz recording of 14 min never sits in memory whole.

The EEG is stored as a bare ``.npy`` so that every process memory-maps the
same file (one copy in the page cache however many processes pretrain). The
``.json`` holds the peak-to-peak rejection threshold, channel order and length;
``_starts.npy`` the clean crop starts of the default ``EEGCrops`` (computed
once here: doing it per fold and per process cost ~200 s a fold).
About 13 MB per session, ~8 GB for the 596 sessions: GPU jobs only (the data
are not downloaded locally, apart from a few sessions for the tests).

Labels are never read here: no emotion, rating or identity beyond the file
name. Which people a pretraining may use is decided by the caller.
"""

from __future__ import annotations

import csv
import itertools
import json
import traceback
import warnings
from pathlib import Path

import numpy as np

from reid import preprocess
from reid.config import CACHE, EEG_BAND, PAD_S, TASKS
from reid.final.crops import EEGCrops
from reid.final.raw import EEG_RATE, F16_MAX, _read, eeg_threshold, resample
from reid.segments import open_recording

VIDEO = CACHE / "video"
CHUNK_S = 60.0
VERSION = 1
FILE_TASK = {v: k for k, v in TASKS.items()}["video"]


def _chunk(rec, start: float, stop: float) -> np.ndarray:
    """[C, n] µV at EEG_RATE for [start, stop), filtered with its context."""
    x, (lead, _) = _read(rec, "eeg", start, stop, PAD_S)
    y = preprocess.bandpass(
        preprocess.average_reference(x * 1e6), rec.sfreq, EEG_BAND
    )  # V -> µV
    z = resample(y, rec.sfreq, EEG_RATE)
    a = round(lead / rec.sfreq * EEG_RATE)
    return z[:, a : a + round((stop - start) * EEG_RATE)]


def extract_video_eeg(bdf: Path, meta: dict) -> tuple[np.ndarray, dict]:
    """(EEG [62, T] float16, info) of one emotion-video recording."""
    rec = open_recording(bdf)
    seg = meta["dataset_export"]["segments"][FILE_TASK]
    duration = min(float(seg["duration_s"]), rec.n_samples / rec.sfreq)
    if duration < 10:
        raise ValueError(f"video segment too short ({duration:.1f} s)")
    edges = np.append(np.arange(0.0, duration, CHUNK_S), duration)
    eeg = np.concatenate(
        [_chunk(rec, a, b) for a, b in itertools.pairwise(edges) if b - a > 0.5],
        axis=-1,
    ).astype(np.float32)
    info = {
        "eeg_reject_uv": float(eeg_threshold([eeg])),
        "ch_names": list(rec.channels["eeg"]),
        "duration_s": round(eeg.shape[-1] / EEG_RATE, 3),
        "eeg_rate": EEG_RATE,
        "version": VERSION,
    }
    return np.clip(eeg, -F16_MAX, F16_MAX).astype(np.float16), info


def run_video_extraction(
    data_dir: Path,
    out_dir: Path = VIDEO,
    shard: tuple[int, int] = (0, 1),
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    """Same contract as ``reid.final.raw.run_raw_extraction``: resumable,
    failures logged to ``out_dir/failures.csv``, never raises for a session,
    prints cumulative counts after each session."""
    bdfs = sorted(
        Path(data_dir).glob(f"CURATED/P*/P*_day-*_task-{FILE_TASK}_multimodal.bdf")
    )[shard[0] :: shard[1]][:limit]
    folder = out_dir / "video"
    folder.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    warnings.filterwarnings("ignore", module="mne")
    for bdf in bdfs:
        pid, day = bdf.name.split("_")[0], int(bdf.name.split("_")[1][4:])
        target = folder / f"{pid}_day-{day}"
        if target.with_suffix(".json").exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            meta_path = bdf.parent / f"{pid}_day-{day}_metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            eeg, info = extract_video_eeg(bdf, meta)
            np.save(target.with_suffix(".npy"), eeg)
            crops = EEGCrops()
            pool = crops.prepare(_as_feats(eeg, info), ("video",))
            starts = pool[0][1] if pool else np.empty(0, int)
            np.save(folder / f"{pid}_day-{day}_starts.npy", starts.astype(np.int32))
            info["starts_crop"] = {"crop_s": crops.crop_s, "grid_s": crops.grid_s}
            # written last: its presence marks a complete session
            target.with_suffix(".json").write_text(json.dumps(info))
            counts["done"] += 1
        except Exception as err:  # noqa: BLE001 - logged, never stops the run
            counts["failed"] += 1
            with open(out_dir / "failures.csv", "a", newline="") as fh:
                csv.writer(fh).writerow(
                    [pid, day, "video", repr(err), traceback.format_exc(limit=2)]
                )
        print(f"video {pid} day-{day}: {counts}", flush=True)
    return counts


def _as_feats(eeg, info: dict) -> dict:
    return {
        "seg_condition": np.array(["video"]),
        "eeg_0": eeg,
        "eeg_reject_uv": info["eeg_reject_uv"],
        "ch_names": np.array(info["ch_names"]),
    }


def load_video(pid: str, day: int, root: Path = VIDEO) -> dict | None:
    """The session as a raw-cache-like dict (one ``video`` segment, EEG
    memory-mapped), or None if it was not extracted. With ``starts`` (and
    ``starts_crop``, the crop settings they hold for) when precomputed."""
    base = root / "video" / f"{pid}_day-{day}"
    meta = base.with_suffix(".json")
    if not meta.is_file():
        return None
    info = json.loads(meta.read_text())
    feats = _as_feats(np.load(base.with_suffix(".npy"), mmap_mode="r"), info)
    starts = root / "video" / f"{pid}_day-{day}_starts.npy"
    if starts.is_file() and "starts_crop" in info:
        feats["starts"], feats["starts_crop"] = np.load(starts), info["starts_crop"]
    return feats


def video_sessions(root: Path = VIDEO) -> list[tuple[str, int]]:
    """(pid, day) of every extracted video session."""
    out = []
    for meta in sorted((root / "video").glob("P*_day-*.json")):
        pid, day = meta.stem.split("_day-")
        out.append((pid, int(day)))
    return out
