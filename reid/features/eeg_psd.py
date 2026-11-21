"""EEG power spectra per electrode.

Each 2 s epoch gives a Hann-windowed periodogram; averaging the clean epochs of
a window (50 % overlap) is exactly Welch's estimate with 2 s segments.

Some sessions are sampled at 998 Hz instead of 1000 Hz: their 2 s periodogram
grid is slightly off 0.5 Hz, so every spectrum is linearly interpolated onto the
fixed grid ``FREQS`` (1-45 Hz, 0.5 Hz steps, 89 bins).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import periodogram

from reid.config import BANDS, EEG_BAND, PSD_FMAX

FREQS = np.arange(EEG_BAND[0], PSD_FMAX + 0.25, 0.5)


def epoch_psd(ep: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies (1-45 Hz, 0.5 Hz steps) and PSD [n_epochs, n_channels, n_freqs]
    in V^2/Hz."""
    freqs, psd = periodogram(ep, fs=sfreq, window="hann", axis=-1)
    keep = (freqs >= EEG_BAND[0] - 1) & (freqs <= PSD_FMAX + 1)
    return FREQS, interp1d(freqs[keep], psd[..., keep], axis=-1)(FREQS)


def log_spectrum(psd: np.ndarray, freqs: np.ndarray, kind: str = "abs") -> np.ndarray:
    """Spectral feature from linear PSD [..., n_channels, n_freqs].

    * ``abs``: log10 power spectrum;
    * ``rel``: log10 of the power relative to each electrode's total power,
      less sensitive to day-to-day electrode impedance;
    * ``bands``: log10 relative power in the 5 canonical bands.
    """
    if kind == "abs":
        return np.log10(psd)
    total = psd.sum(axis=-1, keepdims=True)
    if kind == "rel":
        return np.log10(psd / total)
    if kind == "bands":
        power = [
            psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) for lo, hi in BANDS.values()
        ]
        return np.log10(np.stack(power, axis=-1) / total)
    raise ValueError(f"unknown spectrum kind {kind!r}")
