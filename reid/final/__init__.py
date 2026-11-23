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
    # refit: retrained on all training people up to the early-stopping step
    **{
        f"f_eeg_pool_refit_s{k}": _factory(
            "eeg", SEED + k, tag="_pool_refit", refit=True, **EEG_POOL
        )
        for k in range(3)
    },
    **{
        f"f_{m}_refit_s{k}": _factory(m, SEED + k, tag="_refit", refit=True)
        for m in ("ecg_beat", "resp")
        for k in range(3)
    },
}


# -- EEG encoder pretrained on video EEG (reid.final.pretrain) -----------------
#: Same fine-tuning as f_eeg_pool_refit (and the same inner split as the
#: fusions), starting from an encoder pretrained on the video EEG of the inner
#: training people. The objective is chosen on inner validation (val_top1).
#: max_s caps one pretraining (5 per run, one per fold) to fit a GPU job.
PRETRAIN = {
    o: {"objective": o, "max_s": 240.0} for o in ("mae", "jepa", "session", "xday")
}
MODELS |= {
    f"f_eeg_pt_{o}_s{k}": _factory(
        "eeg",
        SEED + k,
        tag=f"_pt_{o}",
        refit=True,
        split_seed=SEED,
        pretrain=cfg,
        **EEG_POOL,
    )
    for o, cfg in PRETRAIN.items()
    for k in range(3)
}


# -- joint model: 3 encoders + fusion head trained together (reid.final.joint) -
def _joint(k: int, **kw):
    def make():
        from reid.final.joint import JointModel

        return JointModel(seed=SEED + k, split_seed=SEED, **kw)

    return make


MODELS |= {f"j_all3_s{k}": _joint(k) for k in range(3)}


# -- score fusion (reid.final.fusion) ------------------------------------------
#: LDA members read the feature cache, encoders the raw cache. Encoders of a
#: fusion share the LDA's inner split (split_seed=SEED) so that their
#: inner-validation scores line up; their training seed still varies.
def _lda(*modalities):
    def make():
        from reid.baselines.b3_lda_plda import LdaModel

        return LdaModel(modalities)

    return make, "features"


def _members(k: int) -> dict:
    split = {"split_seed": SEED, "refit": True}
    return {
        "lda_eeg": _lda("eeg"),
        "lda_ecg_beat": _lda("ecg_beat"),
        "lda_eeg+ecg_beat+resp": _lda("eeg", "ecg_beat", "resp"),
        "enc_eeg": (
            _factory("eeg", SEED + k, tag="_pool_refit", **split, **EEG_POOL),
            "raw",
        ),
        "enc_ecg_beat": (_factory("ecg_beat", SEED + k, tag="_refit", **split), "raw"),
        "enc_resp": (_factory("resp", SEED + k, tag="_refit", **split), "raw"),
        "joint3": (_joint(k), "raw"),
    }


#: name -> (members, weights). "tuned": simplex weights on inner validation.
FUSIONS = {
    "fuse_ref_lda3": (("lda_eeg+ecg_beat+resp",), "tuned"),
    "fuse_lda3_resp": (("lda_eeg+ecg_beat+resp", "enc_resp"), "tuned"),
    "fuse_lda2_resp": (("lda_eeg", "lda_ecg_beat", "enc_resp"), "tuned"),
    "fuse_enc3": (("enc_eeg", "enc_ecg_beat", "enc_resp"), "tuned"),
    "fuse_enc3_eq": (("enc_eeg", "enc_ecg_beat", "enc_resp"), "equal"),
    "fuse_all5": (
        ("lda_eeg", "lda_ecg_beat", "enc_eeg", "enc_ecg_beat", "enc_resp"),
        "tuned",
    ),
    "fuse_all5_eq": (
        ("lda_eeg", "lda_ecg_beat", "enc_eeg", "enc_ecg_beat", "enc_resp"),
        "equal",
    ),
    # the joint model in place of the three separately trained encoders (alone:
    # fuse_joint, which shares the fitted joint model with the two below)
    "fuse_joint": (("joint3",), "tuned"),
    "fuse_lda2_joint": (("lda_eeg", "lda_ecg_beat", "joint3"), "tuned"),
    "fuse_lda2_joint_eq": (("lda_eeg", "lda_ecg_beat", "joint3"), "equal"),
}


def _fusion(name: str, k: int):
    keys, weights = FUSIONS[name]

    def make():
        from reid.final.fusion import ScoreFusion

        members = _members(k)
        tagged = {
            f"{key}_s{k}" if key.startswith(("enc", "joint")) else key: members[key]
            for key in keys
        }
        return ScoreFusion(f"{name}_s{k}", tagged, weights)

    return make


MODELS |= {f"{n}_s{k}": _fusion(n, k) for n in FUSIONS for k in range(3)}
