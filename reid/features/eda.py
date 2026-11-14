"""Electrodermal activity per window: tonic level and phasic response rate."""

from __future__ import annotations

import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly

NAMES = ("tonic_us", "scr_per_min")
RATE = 100  # Hz, after downsampling from 1 kHz


def features(x: np.ndarray, sfreq: float, pad: tuple[int, int]) -> np.ndarray:
    """Median skin conductance (µS) and skin conductance responses per minute."""
    factor = int(sfreq // RATE)
    y = resample_poly(x, 1, factor)
    lo, hi = pad[0] // factor, len(y) - pad[1] // factor
    if hi - lo < RATE or not np.isfinite(y).all():
        return np.full(len(NAMES), np.nan)
    tonic = float(np.median(y[lo:hi]))
    try:
        phasic = nk.eda_phasic(
            nk.eda_clean(y, sampling_rate=RATE), RATE, method="highpass"
        )
        peaks = nk.eda_peaks(phasic["EDA_Phasic"], sampling_rate=RATE)[1]["SCR_Peaks"]
    except Exception:  # noqa: BLE001 - one bad window must never stop extraction
        return np.array([tonic, np.nan])
    n = np.sum((peaks >= lo) & (peaks < hi))
    return np.array([tonic, 60 * n / ((hi - lo) / RATE)])
