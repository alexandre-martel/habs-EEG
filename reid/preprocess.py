"""Common EEG preprocessing, identical for every session and every model.

Per window (read with ``PAD_S`` of context on each side): average reference,
zero-phase Butterworth band-pass, padding removed, then 2 s epochs with 50 %
overlap. Epochs whose peak-to-peak amplitude exceeds a robust per-session
threshold (median + 5 MAD over all epochs of the session) are rejected.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from reid.config import EEG_BAND, EPOCH_S, REJECT_MAD


def bandpass(x: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass along the last axis."""
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def average_reference(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def epochs(x: np.ndarray, sfreq: float, epoch_s: float = EPOCH_S) -> np.ndarray:
    """[n_epochs, n_channels, n_times] with 50 % overlap (none if too short)."""
    size, step = round(epoch_s * sfreq), round(epoch_s * sfreq / 2)
    starts = range(0, x.shape[-1] - size + 1, step)
    return (
        np.stack([x[:, s : s + size] for s in starts])
        if len(starts)
        else np.empty((0, x.shape[0], size))
    )


def clean_window(
    x: np.ndarray, sfreq: float, pad: tuple[int, int], band=EEG_BAND
) -> np.ndarray:
    """Average-referenced, band-passed window with its padding removed."""
    y = bandpass(average_reference(x), sfreq, band)
    return y[:, pad[0] : y.shape[-1] - pad[1]]


def peak_to_peak(ep: np.ndarray) -> np.ndarray:
    """Worst-channel peak-to-peak amplitude of each epoch."""
    return (ep.max(axis=-1) - ep.min(axis=-1)).max(axis=-1)


def rejection_threshold(ptp: np.ndarray, n_mad: float = REJECT_MAD) -> float:
    """median + n_mad * MAD (scaled to a standard deviation) of the session's
    epoch peak-to-peak amplitudes."""
    med = np.median(ptp)
    mad = 1.4826 * np.median(np.abs(ptp - med))
    return float(med + n_mad * mad)
