"""ECG features per window: heart rate, SDNN, RMSSD and the average beat shape.

The beat shape (-250/+450 ms around each R peak, resampled to 128 points,
z-normalised) is often identifying, but electrode placement varies between days.
The ECG header says mV while values look like µV: only unit-free features are used.
"""

from __future__ import annotations

import neurokit2 as nk
import numpy as np

NAMES = ("hr_bpm", "sdnn_ms", "rmssd_ms")
BEAT_POINTS = 128
BEAT_SPAN_S = (-0.25, 0.45)


def features(
    x: np.ndarray, sfreq: float, pad: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """(HR, SDNN, RMSSD) and the mean beat; NaN where the signal is unusable."""
    nan = (np.full(len(NAMES), np.nan), np.full(BEAT_POINTS, np.nan))
    try:
        clean = nk.ecg_clean(x, sampling_rate=int(sfreq))
        peaks = nk.ecg_peaks(clean, sampling_rate=int(sfreq))[1]["ECG_R_Peaks"]
    except Exception:  # noqa: BLE001 - one bad window must never stop extraction
        return nan
    stop = len(x) - pad[1]
    rr = np.diff(peaks) / sfreq
    rr = rr[(peaks[1:] >= pad[0]) & (peaks[1:] < stop) & (rr > 0.3) & (rr < 2.0)]
    if len(rr) < 3:
        return nan
    stats = np.array(
        [60 / rr.mean(), 1000 * rr.std(), 1000 * np.sqrt(np.mean(np.diff(rr) ** 2))]
    )
    lo, hi = (round(s * sfreq) for s in BEAT_SPAN_S)
    inside = peaks[(peaks >= max(pad[0], -lo)) & (peaks < min(stop, len(x) - hi))]
    beat = np.mean([clean[p + lo : p + hi] for p in inside], axis=0)
    beat = np.interp(
        np.linspace(0, len(beat) - 1, BEAT_POINTS), np.arange(len(beat)), beat
    )
    return stats, (beat - beat.mean()) / (beat.std() + 1e-12)
