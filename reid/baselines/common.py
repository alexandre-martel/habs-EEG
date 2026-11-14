"""Feature assembly shared by the baselines (from cached per-window features)."""

from __future__ import annotations

import numpy as np

from reid.eval.protocol import SessionSet
from reid.features.eeg_psd import log_spectrum

#: Default sub-segments per task.
CONDITIONS = {
    "rest": ("rest_ec", "rest_eo"),
    "artefact": ("art_eo",),
    "video": ("video",),
}
SLOW_KEYS = {"ecg": "w_ecg", "ecg_beat": "w_ecg_beat", "eda": "w_eda", "resp": "w_resp"}


def _select(feats: dict, condition: str | None) -> np.ndarray:
    """Window mask of a condition (``None`` = every window)."""
    cond = feats["w_condition"]
    return np.ones(len(cond), bool) if condition is None else cond == condition


def session_vector(
    feats: dict, modalities, conditions, eeg_kind: str = "rel"
) -> np.ndarray:
    """One vector per session: for each condition, the log spectrum of the
    epoch-weighted mean PSD (EEG) and the mean of the other modalities' window
    features. Missing condition or modality -> NaN entries."""
    parts = []
    for condition in conditions or (None,):
        mask = _select(feats, condition)
        for modality in modalities:
            if modality == "eeg":
                psd, w = feats["w_eeg_psd"][mask], feats["w_eeg_n_epochs"][mask]
                ok = (w > 0) & np.isfinite(psd).all(axis=(1, 2))
                shape = feats["w_eeg_psd"].shape[1:]
                mean = (
                    np.average(psd[ok], axis=0, weights=w[ok])
                    if ok.any()
                    else np.full(shape, np.nan)
                )
                parts.append(log_spectrum(mean, feats["freqs"], eeg_kind).ravel())
            else:
                x = feats[SLOW_KEYS[modality]][mask]
                parts.append(
                    np.nanmean(x, axis=0)
                    if np.isfinite(x).any()
                    else np.full(x.shape[1], np.nan)
                )
    return np.concatenate(parts)


def window_matrix(
    feats: dict, modalities, conditions, eeg_kind: str = "rel"
) -> np.ndarray:
    """[n_windows, d]: per-window features of the selected conditions; windows
    without a clean EEG epoch are dropped when EEG is used."""
    mask = np.zeros(len(feats["w_condition"]), bool)
    for condition in conditions or (None,):
        mask |= _select(feats, condition)
    blocks = []
    for modality in modalities:
        if modality == "eeg":
            psd = feats["w_eeg_psd"][mask]
            blocks.append(
                log_spectrum(psd, feats["freqs"], eeg_kind).reshape(len(psd), -1)
            )
        else:
            blocks.append(feats[SLOW_KEYS[modality]][mask])
    X = np.hstack(blocks).astype(float)
    if "eeg" in modalities:
        X = X[(feats["w_eeg_n_epochs"][mask] > 0)]
    return X


def stack_windows(sessions: SessionSet, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """All windows of a session set and the index of their session."""
    mats = [window_matrix(f, **kwargs) for f in sessions]
    X = np.vstack(mats) if mats else np.empty((0, 0))
    idx = (
        np.concatenate([np.full(len(m), i) for i, m in enumerate(mats)]) if mats else []
    )
    return X, np.asarray(idx, dtype=int)


class Standardizer:
    """z-score with statistics of the training sessions only (NaN-aware);
    missing values become 0 after scaling."""

    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        self.std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num((X - self.mean) / self.std)


def pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows of a and rows of b."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T
