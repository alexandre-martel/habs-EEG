"""Re-identification baselines, registered by name for scripts/reid_run.py.

Every entry is a zero-argument factory returning a fresh ``ReIDModel``.
"""

from reid.baselines.b1_spectral_corr import CorrModel
from reid.eval.controls import DemographicModel

REST = {"ec": ("rest_ec",), "eo": ("rest_eo",), "ec+eo": ("rest_ec", "rest_eo")}


def _b1() -> dict:
    """Baseline 1: spectrum variants x conditions, and each modality alone (step 4)."""
    models = {}
    for kind in ("abs", "rel", "bands"):
        for label, conds in REST.items():
            models[f"b1_{kind}_{label}"] = lambda k=kind, c=conds: CorrModel(
                ("eeg",), c, k
            )
    for modality in ("ecg", "ecg_beat", "eda", "resp"):
        models[f"b1_{modality}_ec+eo"] = lambda m=modality: CorrModel(
            (m,), REST["ec+eo"]
        )
    return models


MODELS = {"control_demographics": DemographicModel, **_b1()}
