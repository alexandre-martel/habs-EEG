"""Baseline 1: correlation of session spectra (and of any session feature vector).

Each session is summarised by its log power spectrum per electrode (separately
for eyes closed / eyes open), z-scored with the training sessions' statistics;
a day-2 session is matched to the most correlated day-1 session. Nothing but the
z-score is learnt, so this is the floor. With other modalities it gives the
per-modality ranking of step 4.
"""

from __future__ import annotations

import numpy as np

from reid.baselines.common import Standardizer, pearson, session_vector
from reid.eval.protocol import SessionSet

SHORT = {"rest_ec": "ec", "rest_eo": "eo", "art_eo": "arteo", "video": "video"}


class CorrModel:
    """z-scored session vectors compared with Pearson correlation."""

    def __init__(
        self,
        modalities=("eeg",),
        conditions=("rest_ec", "rest_eo"),
        eeg_kind: str = "rel",
        prefix: str = "b1",
    ) -> None:
        self.modalities, self.conditions, self.eeg_kind = (
            modalities,
            conditions,
            eeg_kind,
        )
        conds = "+".join(SHORT.get(c, c) for c in conditions) if conditions else "all"
        mods = "+".join(modalities)
        self.name = (
            f"{prefix}_{eeg_kind}_{conds}"
            if mods == "eeg"
            else f"{prefix}_{mods}_{conds}"
        )
        self.config = {
            "modalities": list(modalities),
            "conditions": conditions,
            "eeg_kind": eeg_kind,
        }

    def vectors(self, sessions: SessionSet) -> np.ndarray:
        return np.array(
            [
                session_vector(f, self.modalities, self.conditions, self.eeg_kind)
                for f in sessions
            ]
        )

    def fit(self, train: SessionSet) -> None:
        self.scaler = Standardizer().fit(self.vectors(train))

    def embed(self, sessions: SessionSet) -> np.ndarray:
        return self.scaler.transform(self.vectors(sessions))

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        return pearson(self.embed(probe), self.embed(gallery))
