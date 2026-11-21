"""Task sub-segments from metadata.json, cut into analysis windows, and lazy
reading of the needed channels from a BDF (no full-recording DataFrame).

Sub-segments (segment-relative onset = onset_s - metadata_time_origin_s):
* resting-state: 10002 = 30 s eyes open without blinking, 10004 = 60 s eyes closed;
* artefact: 10000 = 60 s eyes open (twice), 10006-10018 = 5 s instructed actions;
* emotion-video: 10 s trials (``trial_type == "emotion_video"``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from reid.config import EEG_EXCLUDE, RECORD_TYPES, REPO, WINDOW_S

sys.path.insert(0, str(REPO / "starting_notebooks"))
from bdf_loader import uV_to_uS

CONDITIONS = {
    10002: "rest_eo",
    10004: "rest_ec",
    10000: "art_eo",
    10006: "art_blink",
    10008: "art_saccade_h",
    10010: "art_saccade_v",
    10012: "art_head_lr",
    10014: "art_head_ud",
    10016: "art_jaw",
    10018: "art_swallow",
}
ACTIONS = {m for m in CONDITIONS if m >= 10006}  # 5 s instructed actions
TASK_MARKERS = {"resting-state": {10002, 10004}, "artefact": ACTIONS | {10000}}
#: MNE scales BDF data to SI units; this undoes it to get the header units back.
TO_HEADER_UNITS = {"µV": 1e6, "uV": 1e6, "mV": 1e3, "V": 1.0}


@dataclass(frozen=True)
class Window:
    condition: str
    marker: int
    start_s: float  # relative to the task segment (= the BDF start)
    duration_s: float


def sub_segments(meta: dict, task: str) -> list[tuple[str, int, float, float]]:
    """(condition, marker, onset, duration) of the task's sub-segments that lie
    inside the exported segment, clipped to its boundaries."""
    seg = meta["dataset_export"]["segments"][task]
    origin, length = seg["metadata_time_origin_s"], seg["duration_s"]
    out = []
    for ev in meta["events"]:
        if task == "emotion-video":
            if ev.get("trial_type") != "emotion_video":
                continue
            condition = "video"
        elif ev.get("marker_id") in TASK_MARKERS[task]:
            condition = CONDITIONS[ev["marker_id"]]
        else:
            continue
        start = max(ev["onset_s"] - origin, 0.0)
        stop = min(ev["onset_s"] - origin + (ev.get("duration_s") or 0.0), length)
        if stop > start:
            out.append((condition, int(ev["marker_id"]), start, stop - start))
    return out


def windows(meta: dict, task: str, window_s: float = WINDOW_S) -> list[Window]:
    """Cut each sub-segment into consecutive windows of ``window_s``; a last
    partial window is kept if >= 80 % complete. Actions stay one window each."""
    out = []
    for condition, marker, start, duration in sub_segments(meta, task):
        if marker in ACTIONS:
            out.append(Window(condition, marker, start, duration))
            continue
        n = int(duration // window_s) + int(duration % window_s >= 0.8 * window_s)
        for k in range(n):
            length = min(window_s, duration - k * window_s)
            out.append(Window(condition, marker, start + k * window_s, length))
    return out


@dataclass
class Recording:
    """A lazily read BDF with its channels grouped by modality."""

    raw: mne.io.BaseRaw
    channels: dict[str, list[str]]
    n_samples: int

    @property
    def sfreq(self) -> float:
        return float(self.raw.info["sfreq"])

    def read(self, modality: str, start_s: float, stop_s: float) -> np.ndarray:
        """[n_channels, n_samples] in header units (EEG in volts, GSR in µS)."""
        names = self.channels[modality]
        start = max(round(start_s * self.sfreq), 0)
        stop = min(round(stop_s * self.sfreq), self.n_samples)
        x = self.raw.get_data(picks=names, start=start, stop=stop)
        if modality == "eeg":
            return x
        units = self.raw._orig_units
        x = (
            x
            * np.array([TO_HEADER_UNITS.get(units.get(n), 1.0) for n in names])[:, None]
        )
        return uV_to_uS(x) if modality == "eda" else x  # GSR is stored in µV


def open_recording(bdf: Path) -> Recording:
    """Open a BDF lazily and map its channels to modalities via the record map."""
    bdf = Path(bdf)
    record_map = json.loads(bdf.with_suffix(".record_map").read_text(encoding="utf-8"))
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
    channels: dict[str, list[str]] = {}
    for stream in record_map["stream_map"].values():
        for spec in stream.values():
            correction = spec.get("correction", {})
            modality = RECORD_TYPES.get(str(correction.get("RecordType", "")).upper())
            if modality:  # BDF labels are the record-map names cut to 16 chars
                names = [c["name"][:16] for c in correction["Channels"]]
                channels[modality] = [n for n in names if n not in EEG_EXCLUDE]
    total = record_map.get("TotalSamples")
    n_samples = (
        min(raw.n_times, total) if isinstance(total, int) and total > 0 else raw.n_times
    )
    return Recording(raw, channels, n_samples)
