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


MODELS = {
    f"f_{m}_s{k}": _factory(m, SEED + k)
    for m in ("eeg", "ecg_beat", "resp")
    for k in range(3)
}
