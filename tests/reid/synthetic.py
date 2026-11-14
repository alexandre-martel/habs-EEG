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
