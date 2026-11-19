"""Final model: one encoder per modality (EEG, ECG beat shape, respiration),
trained on raw-signal crops (``reid.final.raw``) by cross-day contrast.

``MODELS`` maps run names to zero-argument factories, as in ``reid.baselines``;
they read the raw cache (``cache/raw``), not the feature cache.
"""

from reid.config import SEED


def _factory(modality: str, seed: int = SEED, **kwargs):
    def make():
        from reid.final.model import ContrastiveModel

        return ContrastiveModel(modality, seed=seed, **kwargs)

    return make


#: Session-level contrast for EEG: each unit of the loss is the mean of 8 crops
#: of one (person, day), 64 people per batch. Chosen on inner validation only
#: (mean over folds 0.841 vs 0.780 crop-level; 0.762 with the same batch size
#: but no pooling, so the pooling is what helps).
EEG_POOL = {"pool_crops": True, "crops": 8, "people": 64, "eval_every": 50}

MODELS = {
    **{
        f"f_{m}_s{k}": _factory(m, SEED + k)
        for m in ("eeg", "ecg_beat", "resp")
        for k in range(3)
    },
    **{
        f"f_eeg_pool_s{k}": _factory("eeg", SEED + k, tag="_pool", **EEG_POOL)
        for k in range(3)
    },
}
