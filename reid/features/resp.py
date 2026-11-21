"""Respiration per window: breathing rate, amplitude variability, I/E ratio."""

from __future__ import annotations

import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly

NAMES = ("rate_bpm", "amp_cv", "ie_ratio")
RATE = 100  # Hz, after downsampling from 1 kHz


def features(x: np.ndarray, sfreq: float, pad: tuple[int, int]) -> np.ndarray:
    """Breaths whose peak lies in the window (context given by ``pad``)."""
    nan = np.full(len(NAMES), np.nan)
    factor = int(sfreq // RATE)
    y = resample_poly(x, 1, factor)
    lo, hi = pad[0] // factor, len(y) - pad[1] // factor
    try:
        clean = nk.rsp_clean(y, sampling_rate=RATE)
        info = nk.rsp_peaks(clean, sampling_rate=RATE)[1]
    except Exception:  # noqa: BLE001 - one bad window must never stop extraction
        return nan
    peaks, troughs = np.asarray(info["RSP_Peaks"]), np.asarray(info["RSP_Troughs"])
    breaths = []  # (trough before, peak, trough after)
    for p in peaks[(peaks >= lo) & (peaks < hi)]:
        before, after = troughs[troughs < p], troughs[troughs > p]
        if len(before) and len(after):
            breaths.append((before[-1], p, after[0]))
    if len(breaths) < 2:
        return nan
    t0, pk, t1 = np.array(breaths).T
    amp = clean[pk] - clean[t0]
    return np.array(
        [
            60 * RATE / np.diff(pk).mean(),
            amp.std() / (np.abs(amp.mean()) + 1e-12),
            (pk - t0).mean() / (t1 - pk).mean(),
        ]
    )
