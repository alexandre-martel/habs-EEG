"""Tanasković et al. (2025) ECG features on every resting-state recording.

    python scripts/ecg_tanaskovic_extract.py --data /data --out OUT_DIR [--days 1 2]

Reads CURATED/P*/P*_day-<d>_task-resting-state_multimodal.bdf (ECG channel
only), and writes one row per 10-beat template of the rest blocks 10002 and
10004 (reid.features.ecg_tanaskovic) to OUT_DIR/features.csv, with
participant_id, day, block, template index, time of its first beat, T
polarity, the 16 features and the template resampled to 75 points
(beat_00..beat_74, one every 10 ms). Files that cannot be processed go to
OUT_DIR/failures.csv.

With --task anger: one template per anger video trial (10 s) of the
emotion-video recordings, to anger_features.csv / anger_failures.csv.

Both files name participants: keep OUT_DIR out of anything public. Standard
output only gets counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from ecg_rest_full import rest_segments  # noqa: E402
from reid.features import ecg_tanaskovic as tk  # noqa: E402

NAME = re.compile(r"(P\d+)_day-(\d+)_task-resting-state_multimodal\.bdf$")


def one(bdf: Path) -> list[dict]:
    pid, day = NAME.search(bdf.name).groups()
    raw = mne.io.read_raw_bdf(bdf, include=["ECG"], preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]
    x = raw.get_data(picks="ECG")[0] * 1e3  # stored values (header unit)
    segs = rest_segments(raw)
    if not segs:
        raise ValueError("no rest marker (10002/10004)")
    rows, *_ = tk.extract(x, fs, [(str(s["marker"]), s["start"], s["stop"]) for s in segs])
    dur = {str(s["marker"]): s["duration_s"] for s in segs}
    for r in rows:
        r["block_duration_s"] = dur[r["segment"]]
    return [{"participant_id": pid, "day": int(day), "sfreq": fs} | r for r in rows]


VIDEO = re.compile(r"(P\d+)_day-(\d+)_task-emotion-video_multimodal\.bdf$")
ANGER = "colere"  # metadata emotion_label of the anger videos
PAD_S = 5.0  # signal read around each 10 s trial (filter edges, next beat)


def one_anger(bdf: Path) -> list[dict]:
    """One template per anger video trial (10 s): the first 10 beats whose R
    peak falls inside the trial, as in tk.templates (window -250/+500 ms,
    RR = mean of the 10 intervals that follow, the 10th one may end just
    after the trial). Only the trial +-5 s is read (lazy BDF reading); the
    Pan-Tompkins threshold is computed on that 20 s chunk. A trial with fewer
    than 10 beats (heart rate < ~60 bpm) gives no template."""
    pid, day = VIDEO.search(bdf.name).groups()
    meta = json.loads(next(bdf.parent.glob(f"{pid}_day-{day}_metadata.json")).read_text())
    seg = meta["dataset_export"]["segments"]["emotion-video"]
    origin = seg["metadata_time_origin_s"]
    trials = [e for e in meta["events"] if e.get("trial_type") == "emotion_video"
              and e.get("emotion_label") == ANGER]
    if not trials:
        raise ValueError("no anger trial in metadata")
    raw = mne.io.read_raw_bdf(bdf, include=["ECG"], preload=False, verbose="ERROR")
    fs = raw.info["sfreq"]
    pre, post = tk.ms(tk.PRE_S * 1000, fs), tk.ms(tk.POST_S * 1000, fs)
    rows = []
    for k, ev in enumerate(sorted(trials, key=lambda e: e["onset_s"])):
        t0 = ev["onset_s"] - origin
        dur = ev.get("duration_s") or 10.0
        lo = max(round((t0 - PAD_S) * fs), 0)
        hi = min(round((t0 + dur + PAD_S) * fs), raw.n_times)
        x = raw.get_data(picks="ECG", start=lo, stop=hi)[0] * 1e3
        xf = tk.bandpass(x, fs)
        peaks = tk.pan_tompkins(xf, fs)
        start, stop = round(t0 * fs) - lo, round((t0 + dur) * fs) - lo
        inside = np.flatnonzero((peaks >= start) & (peaks < stop)
                                & (peaks - pre >= 0) & (peaks + post <= len(xf)))
        row = {"participant_id": pid, "day": int(day), "sfreq": fs, "segment": "anger",
               "template": k, "t_first_beat_s": t0, "media_file": ev.get("media_file"),
               "n_beats_in_trial": len(inside)}
        if len(inside) < tk.N_AVG or inside[tk.N_AVG - 1] + 1 >= len(peaks):
            rows.append(row | {"fiducials_ok": False, "T_polarity": np.nan}
                        | dict.fromkeys(tk.NAMES, np.nan))
            continue
        beats = peaks[inside[: tk.N_AVG]]
        rr = float(np.mean(np.diff(peaks[inside[0]: inside[tk.N_AVG - 1] + 2]) / fs))
        tpl = np.mean([xf[b - pre: b + post] for b in beats], axis=0)
        f = tk.fiducials(tpl, fs)
        row |= {"fiducials_ok": f is not None, "T_polarity": f["T_polarity"] if f else np.nan}
        row |= tk.template_features(tpl, f, rr, fs) if f else dict.fromkeys(tk.NAMES, np.nan)
        row |= tk.beat_columns(tpl)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--task", choices=["rest", "anger"], default="rest",
                    help="rest blocks (features.csv) or anger videos (anger_features.csv)")
    a = ap.parse_args()
    task = "resting-state" if a.task == "rest" else "emotion-video"
    bdfs = sorted(
        b for d in a.days
        for b in a.data.glob(f"CURATED/P*/P*_day-{d}_task-{task}_multimodal.bdf")
    )
    run = one if a.task == "rest" else one_anger
    prefix = "" if a.task == "rest" else "anger_"
    a.out.mkdir(parents=True, exist_ok=True)
    rows, failures, start = [], [], time.time()
    for k, bdf in enumerate(bdfs, 1):
        try:
            rows += run(bdf)
        except Exception as err:  # noqa: BLE001 - one bad file must not stop the run
            failures.append({"file": bdf.name, "error": repr(err),
                             "trace": traceback.format_exc(limit=3)})
        if k % 50 == 0 or k == len(bdfs):
            print(f"{k}/{len(bdfs)} files, {len(failures)} failed, "
                  f"{len(rows)} templates, {time.time() - start:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(a.out / f"{prefix}features.csv", index=False)
    pd.DataFrame(failures, columns=["file", "error", "trace"]).to_csv(
        a.out / f"{prefix}failures.csv", index=False)


if __name__ == "__main__":
    main()
