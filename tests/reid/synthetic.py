"""Synthetic sessions with a controllable amount of identity information."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reid.config import SEED


class DictStore:
    """In-memory stand-in for ``FeatureStore``."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def load(self, pid: str, day: int, task: str) -> dict:
        return self.data[(pid, int(day), task)]

    def available(self, pid: str, day: int, task: str) -> bool:
        return (pid, int(day), task) in self.data


def make_dataset(
    n_multi: int = 60,
    n_single: int = 20,
    dim: int = 16,
    identity: float = 1.0,
    noise: float = 0.05,
    n_windows: int = 6,
    seed: int = SEED,
) -> tuple[pd.DataFrame, DictStore]:
    """People with 2 days (multi) or 1 day (single). Each window's vector is
    ``identity * person_vector + noise * N(0, 1)``."""
    rng = np.random.default_rng(seed)
    rows, data = [], {}
    for k in range(n_multi + n_single):
        pid = f"P{k:03d}"
        person = rng.normal(size=dim)
        days = (1, 2) if k < n_multi else (1,)
        for day in days:
            rows.append(
                {
                    "participant_id": pid,
                    "day": day,
                    "has_rest": True,
                    "has_artefact": False,
                    "has_video": False,
                    "cohort": 1 + k % 2,
                    "age": float(20 + k % 40),
                    "gender": "F" if k % 3 else "M",
                }
            )
            x = identity * person + noise * rng.normal(size=(n_windows, dim))
            data[(pid, day, "rest")] = {
                "w_x": x,
                "w_condition": np.array(["ec", "eo"] * (n_windows // 2)),
                "w_duration": np.full(n_windows, 10.0),
            }
    return pd.DataFrame(rows), DictStore(data)


def make_feature_dataset(
    n_multi: int = 45,
    n_single: int = 10,
    n_ch: int = 8,
    identity: float = 1.0,
    day_gain: float = 0.3,
    noise: float = 0.3,
    seed: int = SEED,
) -> tuple[pd.DataFrame, DictStore]:
    """Sessions in the cached-feature format (w_eeg_psd, w_eeg_cov, w_ecg, ...).

    Each person has a log-spectrum profile and a spatial mixing matrix scaled by
    ``identity``; each day adds a per-electrode gain (impedance, ``day_gain``);
    each window adds noise. ``identity=0`` makes people indistinguishable.
    """
    rng = np.random.default_rng(seed)
    freqs = np.linspace(1, 45, 23)
    base = -np.log10(freqs)[None] * np.ones((n_ch, 1))
    conditions = np.array(["rest_eo"] * 3 + ["rest_ec"] * 6)
    alpha = np.exp(-0.5 * ((freqs - 10) / 1.5) ** 2)[None]
    rows, data = [], {}
    for k in range(n_multi + n_single):
        pid = f"P{k:03d}"
        profile = base + identity * rng.normal(scale=0.3, size=(n_ch, len(freqs)))
        mixing = np.eye(n_ch) + identity * rng.normal(scale=0.4, size=(n_ch, n_ch))
        hr, beat = 60 + identity * rng.normal(scale=8), identity * rng.normal(size=128)
        slow = identity * rng.normal(size=5)
        for day in (1, 2) if k < n_multi else (1,):
            rows.append(
                {
                    "participant_id": pid,
                    "day": day,
                    "has_rest": True,
                    "has_artefact": False,
                    "has_video": False,
                    "cohort": 1 + k % 2,
                    "age": float(20 + k % 40),
                    "gender": "F" if k % 3 else "M",
                }
            )
            gain = rng.normal(scale=day_gain, size=(n_ch, 1))
            psd, cov = [], []
            for cond in conditions:
                log_psd = profile + gain + (0.5 * alpha if cond == "rest_ec" else 0)
                psd.append(
                    10 ** (log_psd + rng.normal(scale=noise, size=log_psd.shape))
                )
                g = np.diag(10 ** gain[:, 0]) @ mixing
                samples = g @ rng.normal(size=(n_ch, 200))
                c = np.cov(samples) + 1e-3 * np.eye(n_ch)
                cov.append(np.stack([c * s for s in (1.0, 0.5, 0.8, 0.3)]))
            n = len(conditions)
            data[(pid, day, "rest")] = {
                "w_condition": conditions,
                "w_duration": np.full(n, 10.0),
                "w_eeg_n_epochs": np.full(n, 9),
                "w_eeg_psd": np.array(psd),
                "w_eeg_cov": np.array(cov),
                "freqs": freqs,
                "cov_bands": np.array(["broad", "theta", "alpha", "beta"]),
                "w_ecg": np.c_[hr + rng.normal(size=(n, 1)), rng.normal(50, 5, (n, 2))],
                "w_ecg_beat": beat + noise * rng.normal(size=(n, 128)),
                "w_eda": slow[:2] + noise * rng.normal(size=(n, 2)),
                "w_resp": slow[2:] + noise * rng.normal(size=(n, 3)),
            }
    return pd.DataFrame(rows), DictStore(data)
