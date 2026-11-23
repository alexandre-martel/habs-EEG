"""ECG on whole resting-state sub-segments instead of 10 s windows.

    python scripts/ecg_rest_full.py sample/CURATED/P001/P001_day-1_task-resting-state_multimodal.bdf [...]

For each BDF, reads only the ECG channel with MNE, finds the rest
sub-segments from the BDF+ annotations (start marker -> end marker:
10002 -> 10003 "calm rest, no blinks", 10004 -> 10005 "eyes closed"),
and on each whole sub-segment:

* cleans the ECG and detects R peaks exactly like ``reid.features.ecg``
  (NeuroKit2 ``ecg_clean`` then ``ecg_peaks``, same 0.3-2.0 s RR bounds),
  with up to ``PAD_S`` of signal read around the segment to absorb filter
  edges, as the windowed pipeline does;
* computes the pipeline's 11 features by calling ``reid.features.ecg.features``
  once on the whole segment, plus whole-segment HRV (pNN50, RR count, share
  of rejected RR intervals);
* plots the raw ECG of the file with the segments shaded, and each cleaned
  segment with its R peaks.

The original pipeline (``reid/features``) is imported, not modified.
Outputs: ``results/ecg_rest_full/<stem>.csv`` and ``.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import neurokit2 as nk  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reid.config import PAD_S  # noqa: E402
from reid.features import ecg  # noqa: E402

REST = {"10002": ("10003", "rest_no_blink"), "10004": ("10005", "rest_eyes_closed")}
RR_BOUNDS_S = (0.3, 2.0)  # same physiological bounds as reid.features.ecg


def rest_segments(raw: mne.io.BaseRaw) -> list[dict]:
    """Rest sub-segments from start/end marker annotations (seconds, samples)."""
    ann = list(zip(raw.annotations.description, raw.annotations.onset))
    fs = raw.info["sfreq"]
    out = []
    for i, (desc, onset) in enumerate(ann):
        if desc not in REST:
            continue
        end_code, name = REST[desc]
        end = next((o for d, o in ann[i + 1 :] if d == end_code), None)
        if end is None:  # no end marker: segment runs to the end of the file
            end = raw.n_times / fs
        out.append({
            "marker": int(desc), "condition": name,
            "start_s": onset, "end_s": end, "duration_s": end - onset,
            "start": round(onset * fs), "stop": round(end * fs),
        })
    return out


def segment_ecg(x: np.ndarray, fs: float, seg: dict) -> dict:
    """Clean, detect R peaks and compute features on one whole segment."""
    pad = round(PAD_S * fs)
    lo, hi = max(seg["start"] - pad, 0), min(seg["stop"] + pad, len(x))
    chunk = x[lo:hi]
    lead, tail = seg["start"] - lo, hi - seg["stop"]

    clean = nk.ecg_clean(chunk, sampling_rate=int(fs))
    peaks = np.asarray(nk.ecg_peaks(clean, sampling_rate=int(fs))[1]["ECG_R_Peaks"])
    inside = peaks[(peaks >= lead) & (peaks < len(chunk) - tail)]

    rr_all = np.diff(inside) / fs
    ok = (rr_all > RR_BOUNDS_S[0]) & (rr_all < RR_BOUNDS_S[1])
    rr = rr_all[ok]
    stats, beat = ecg.features(chunk, fs, (lead, tail))
    row = {k: seg[k] for k in ("marker", "condition", "start_s", "end_s", "duration_s")}
    row |= {
        "n_samples": seg["stop"] - seg["start"],
        "n_r_peaks": len(inside),
        "n_rr_kept": int(ok.sum()),
        "pct_rr_rejected": 100 * (1 - ok.mean()) if len(ok) else np.nan,
        "pnn50_pct": 100 * np.mean(np.abs(np.diff(rr)) > 0.05) if len(rr) > 1 else np.nan,
    }
    row |= dict(zip(ecg.NAMES, stats))
    return {
        "row": row, "beat": beat,
        "clean": clean[lead : len(chunk) - tail],
        "peaks": inside - lead,
    }


def run(bdf: Path, out_dir: Path) -> pd.DataFrame:
    raw = mne.io.read_raw_bdf(bdf, include=["ECG"], preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]
    # MNE scales by the header unit (mV -> V); x1e3 gives back the stored values.
    x = raw.get_data(picks="ECG")[0] * 1e3
    segs = rest_segments(raw)
    print(f"{bdf.name}: ECG only, sfreq = {fs:g} Hz, file = {raw.n_times} samples "
          f"({raw.n_times / fs:.3f} s), header unit = {raw._orig_units.get('ECG')}")
    if not segs:
        print("  no rest marker (10002/10004) in this file")
        return pd.DataFrame()

    results = [segment_ecg(x, fs, s) for s in segs]
    df = pd.DataFrame([r["row"] for r in results])
    df.insert(0, "file", bdf.name)
    for r in df.itertuples():
        print(f"  {r.marker} {r.condition:17s} {r.start_s:8.3f} -> {r.end_s:8.3f} s  "
              f"duration {r.duration_s:.3f} s ({r.n_samples} samples)  "
              f"{r.n_r_peaks} R peaks  HR {r.hr_bpm:.1f} bpm  SDNN {r.sdnn_ms:.1f} ms  "
              f"RMSSD {r.rmssd_ms:.1f} ms")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{bdf.stem}.csv", index=False)

    t = np.arange(len(x)) / fs
    fig, axes = plt.subplots(1 + len(results), 1, figsize=(14, 3 + 2.6 * len(results)),
                             constrained_layout=True)
    axes[0].plot(t, x, lw=0.4, color="0.2")
    for s, c in zip(segs, ("tab:blue", "tab:orange")):
        axes[0].axvspan(s["start_s"], s["end_s"], color=c, alpha=0.12,
                        label=f"{s['marker']} {s['condition']} ({s['duration_s']:.3f} s)")
    axes[0].set(title=f"{bdf.name}: raw ECG (unfiltered)", xlabel="time in file (s)",
                ylabel=f"ECG (stored, header {raw._orig_units.get('ECG')})", xlim=(0, t[-1]))
    axes[0].legend(loc="upper right", fontsize=8)
    for ax, s, r in zip(axes[1:], segs, results):
        ts = s["start_s"] + np.arange(len(r["clean"])) / fs
        ax.plot(ts, r["clean"], lw=0.5, color="0.2")
        ax.plot(ts[r["peaks"]], r["clean"][r["peaks"]], "v", ms=4, color="tab:red")
        ax.set(title=f"{s['marker']} {s['condition']}: cleaned ECG + {len(r['peaks'])} R peaks, "
                     f"HR {r['row']['hr_bpm']:.1f} bpm", xlabel="time in file (s)",
               xlim=(ts[0], ts[-1]))
    fig.savefig(out_dir / f"{bdf.stem}.png", dpi=130)
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bdf", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "ecg_rest_full")
    a = ap.parse_args()
    frames = [run(b, a.out) for b in a.bdf]
    if len(frames) > 1:
        pd.concat(frames).to_csv(a.out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
