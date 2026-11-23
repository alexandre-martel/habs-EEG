"""Tanasković et al. (2025) ECG features on every resting-state recording.

    python scripts/ecg_tanaskovic_extract.py --data /data --out OUT_DIR [--days 1 2]

Reads CURATED/P*/P*_day-<d>_task-resting-state_multimodal.bdf (ECG channel
only), and writes one row per 10-beat template of the rest blocks 10002 and
10004 (reid.features.ecg_tanaskovic) to OUT_DIR/features.csv, with
participant_id, day, block, template index, time of its first beat, T
polarity and the 16 features. Files that cannot be processed go to
OUT_DIR/failures.csv.

Both files name participants: keep OUT_DIR out of anything public. Standard
output only gets counts.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

import mne
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    bdfs = sorted(
        b for d in a.days
        for b in a.data.glob(f"CURATED/P*/P*_day-{d}_task-resting-state_multimodal.bdf")
    )
    a.out.mkdir(parents=True, exist_ok=True)
    rows, failures, start = [], [], time.time()
    for k, bdf in enumerate(bdfs, 1):
        try:
            rows += one(bdf)
        except Exception as err:  # noqa: BLE001 - one bad file must not stop the run
            failures.append({"file": bdf.name, "error": repr(err),
                             "trace": traceback.format_exc(limit=3)})
        if k % 50 == 0 or k == len(bdfs):
            print(f"{k}/{len(bdfs)} files, {len(failures)} failed, "
                  f"{len(rows)} templates, {time.time() - start:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "features.csv", index=False)
    pd.DataFrame(failures, columns=["file", "error", "trace"]).to_csv(a.out / "failures.csv", index=False)


if __name__ == "__main__":
    main()
