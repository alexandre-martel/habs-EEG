"""Re-identification baselines, registered by name for scripts/reid_run.py.

Every entry is a zero-argument factory returning a fresh ``ReIDModel``.
"""

from reid.baselines.b1_spectral_corr import CorrModel
from reid.baselines.b2_riemann import FILTER_BANK, RiemannModel
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


def _b2() -> dict:
    """Baseline 2: AIRM / tangent x broadband / filter bank, and the recentering
    control (should erase identity)."""
    models = {}
    for metric in ("airm", "tangent"):
        for band_label, bands in (("broad", ("broad",)), ("fb", FILTER_BANK)):
            for label, conds in REST.items():
                name = f"b2_{metric}_{band_label}_{label}"
                models[name] = lambda m=metric, b=bands, c=conds: RiemannModel(m, b, c)
    models["b2_tangent_fb_ec+eo_recentered"] = lambda: RiemannModel(
        "tangent", FILTER_BANK, REST["ec+eo"], recenter=True
    )
    return models


MODELS = {"control_demographics": DemographicModel, **_b1(), **_b2()}
