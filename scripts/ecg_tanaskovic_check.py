"""Visual and numeric check of reid.features.ecg_tanaskovic on rest recordings.

    python scripts/ecg_tanaskovic_check.py sample/CURATED/P001/P001_day-1_task-resting-state_multimodal.bdf

For each BDF: R peaks of the Pan-Tompkins variant compared with NeuroKit2's
default detector, then every 10-beat template of the rest segments (10002,
10004) with its fiducial points, and the 16 features per template.
Outputs: results/ecg_tanaskovic/<stem>.png and <stem>.csv (not committed:
per-participant values).
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
sys.path.insert(0, str(REPO / "scripts"))
from ecg_rest_full import rest_segments  # noqa: E402
from reid.features import ecg_tanaskovic as tk  # noqa: E402

COLORS = {"Q": "tab:blue", "R": "tab:red", "S": "tab:green",
          "Tstart": "tab:purple", "T": "tab:orange", "Tend": "tab:brown"}


def check(bdf: Path, out: Path) -> pd.DataFrame:
    raw = mne.io.read_raw_bdf(bdf, include=["ECG"], preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]
    x = raw.get_data(picks="ECG")[0] * 1e3  # stored values (header unit)
    segs = rest_segments(raw)
    rows, details, xf, peaks = tk.extract(
        x, fs, [(f"{s['marker']}", s["start"], s["stop"]) for s in segs]
    )

    nk_peaks = np.asarray(nk.ecg_peaks(nk.ecg_clean(x, sampling_rate=int(fs)),
                                       sampling_rate=int(fs))[1]["ECG_R_Peaks"])
    dist = np.array([np.min(np.abs(nk_peaks - p)) for p in peaks]) * 1000 / fs
    print(f"{bdf.name}: fs {fs:g} Hz")
    print(f"  R peaks: Pan-Tompkins variant {len(peaks)}, NeuroKit2 {len(nk_peaks)}; "
          f"matched within 20 ms: {(dist <= 20).sum()}, median offset {np.median(dist):.1f} ms")
    df = pd.DataFrame(rows)
    for s in segs:
        sub = df[df.segment == str(s["marker"])]
        print(f"  {s['marker']} {s['condition']}: {s['duration_s']:.3f} s, "
              f"{len(sub)} templates, fiducials ok {int(sub.fiducials_ok.sum())}")
    out.mkdir(parents=True, exist_ok=True)
    df.insert(0, "file", bdf.name)
    df.to_csv(out / f"{bdf.stem}.csv", index=False)

    n = len(details)
    cols = 4
    nrows = 1 + int(np.ceil(n / cols))
    fig = plt.figure(figsize=(16, 3.2 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows, cols)
    ax = fig.add_subplot(gs[0, :])
    t = np.arange(len(xf)) / fs
    ax.plot(t, xf, lw=0.4, color="0.25")
    ax.plot(peaks / fs, xf[peaks], "v", ms=4, color="tab:red", label="Pan-Tompkins variant")
    ax.plot(nk_peaks / fs, xf[nk_peaks] * 1.08, "|", ms=8, color="tab:blue", label="NeuroKit2")
    for s in segs:
        ax.axvspan(s["start_s"], s["end_s"], color="tab:green", alpha=0.07)
    ax.set(title=f"{bdf.name}: ECG band-passed 1-40 Hz", xlabel="time (s)", xlim=(0, t[-1]))
    ax.legend(loc="upper right", fontsize=8)
    pre = tk.ms(tk.PRE_S * 1000, fs)
    for i, (name, k, tpl, f) in enumerate(details):
        a = fig.add_subplot(gs[1 + i // cols, i % cols])
        tt = (np.arange(len(tpl)) - pre) * 1000 / fs
        a.plot(tt, tpl, color="0.2", lw=1)
        if f:
            for key in tk.FIDUCIALS:
                a.plot(tt[f[key]], tpl[f[key]], "o", ms=5, color=COLORS[key], label=key)
        a.set_title(f"{name} template {k}" + (f", T polarity {f['T_polarity']:+d}" if f
                                              else " (fiducials failed)"), fontsize=9)
        a.set_xlabel("ms from R", fontsize=8)
        if i == 0 and f:
            a.legend(fontsize=7, ncol=2)
    fig.savefig(out / f"{bdf.stem}.png", dpi=110)
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bdf", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "ecg_tanaskovic")
    a = ap.parse_args()
    for b in a.bdf:
        df = check(b, a.out)
        with pd.option_context("display.width", 200, "display.max_columns", 30,
                               "display.float_format", "{:.3g}".format):
            print(df[list(tk.NAMES)].describe().loc[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
