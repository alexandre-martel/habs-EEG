"""Baseline 4: ablations of one method by modality, task, duration and emotion.

Not a new model: the chosen method (b1, b3_lda, b3_plda or b5) is re-run on the
shared folds with one factor changed at a time. Each run record carries
``ablation`` (kind), ``label`` and, for durations, ``seconds``.

Artefact-block actions (blinks, saccades, jaw, swallow) cannot be matched across
days: the artefact block was recorded on a single day per person.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from reid.baselines.b1_spectral_corr import CorrModel
from reid.baselines.b3_lda_plda import ALL, LdaModel
from reid.baselines.b5_siamese import SiameseModel
from reid.eval.protocol import FeatureStore
from reid.eval.runner import evaluate

METHODS = {
    "b1": CorrModel,
    "b3_lda": LdaModel,
    "b3_plda": lambda **kw: LdaModel(plda=True, **kw),
    "b5": SiameseModel,
}
REST = ("rest_ec", "rest_eo")

#: label -> (gallery task, probe task, conditions)
TASKS = {
    "rest_ec": ("rest", None, ("rest_ec",)),
    "rest_eo": ("rest", None, ("rest_eo",)),
    "rest_ec+eo": ("rest", None, REST),
    "video": ("video", None, ("video",)),
    "rest+video": ("rest+video", None, (*REST, "video")),
    "rest_to_video": ("rest", "video", None),  # all windows pooled
    "arteo_to_resteo": (
        "artefact",
        "rest",
        {"artefact": ("art_eo",), "rest": ("rest_eo",)},
    ),
}
#: seconds kept per condition (rest: EC has 60 s, EO 30 s; video: 10 s per trial)
DURATIONS = {"rest": (10, 20, 30, 60), "video": (10, 30, 60, 120, 240, 360)}


class CombinedStore:
    """Wraps a store: task ``"a+b"`` concatenates the windows of tasks a and b;
    ``keep`` optionally restricts windows to given markers per (pid, day)."""

    def __init__(self, store: FeatureStore, keep: dict | None = None) -> None:
        self.store, self.keep = store, keep

    def available(self, pid: str, day: int, task: str) -> bool:
        ok = all(self.store.available(pid, day, t) for t in task.split("+"))
        return ok and (self.keep is None or bool(self.keep.get((pid, int(day)))))

    def load(self, pid: str, day: int, task: str) -> dict:
        parts = [self.store.load(pid, day, t) for t in task.split("+")]
        feats = dict(parts[0])
        for key in feats:
            if key.startswith("w_"):
                feats[key] = np.concatenate([p[key] for p in parts])
        feats["task"] = task
        if self.keep is not None:
            mask = np.isin(feats["w_marker"], list(self.keep[(pid, int(day))]))
            feats = {
                k: (v[mask] if k.startswith("w_") else v) for k, v in feats.items()
            }
        return feats


def _factory(method: str, label: str, base: dict | None = None, **kwargs):
    def make():
        model = METHODS[method](**{**(base or {}), **kwargs})
        model.name = f"abl_{method}_{label}"
        return model

    return make


def _with_task(inv: pd.DataFrame, task: str) -> pd.DataFrame:
    if "+" in task:
        inv = inv.assign(
            **{f"has_{task}": inv[[f"has_{t}" for t in task.split("+")]].all(axis=1)}
        )
    return inv


def arousal_split(labels: pd.DataFrame) -> dict[str, dict]:
    """Video-trial start markers below / above each session's median arousal."""
    labels = labels.dropna(subset=["rating_arousal", "start_marker"]).copy()
    labels["day"] = labels.day.astype(str).map(lambda d: int(re.sub(r"\D", "", d) or 0))
    out = {"arousal_low": {}, "arousal_high": {}}
    for (pid, day), g in labels.groupby(["participant_id", "day"]):
        median = g.rating_arousal.median()
        markers = g.start_marker.astype(int)
        out["arousal_low"][(pid, day)] = set(markers[g.rating_arousal <= median])
        out["arousal_high"][(pid, day)] = set(markers[g.rating_arousal > median])
    return out


def run(
    kind: str,
    method: str,
    inv: pd.DataFrame,
    folds: list[dict],
    store: FeatureStore,
    data_dir: Path | None = None,
    n_boot: int = 1000,
    base: dict | None = None,
    save: bool = True,
) -> list[dict]:
    """Run one ablation family; returns the run records (also saved).
    ``base``: constructor arguments of the method (e.g. ``modalities``)."""
    store = CombinedStore(store)
    common = {"folds": folds, "scenario": "d1_to_d2", "n_boot": n_boot, "save": save}
    records = []

    def go(
        label, make, task="rest", probe_task=None, part=None, the_store=store, **extra
    ):
        try:
            records.append(
                evaluate(
                    make,
                    _with_task(inv, task),
                    task=task,
                    probe_task=probe_task,
                    store=the_store,
                    part=part,
                    extra={"ablation": kind, "label": label, **extra},
                    **common,
                )
            )
        except ValueError as err:  # e.g. no person with this task on both days
            print(f"{kind}/{label}: skipped ({err})")

    if kind == "modality":
        for m in ALL:
            go(
                f"only_{m}",
                _factory(
                    method,
                    base=base,
                    label=f"only_{m}",
                    modalities=(m,),
                    conditions=REST,
                ),
            )
            rest = tuple(x for x in ALL if x != m)
            go(
                f"minus_{m}",
                _factory(
                    method,
                    base=base,
                    label=f"minus_{m}",
                    modalities=rest,
                    conditions=REST,
                ),
            )
    elif kind == "task":
        for label, (task, probe_task, conds) in TASKS.items():
            go(
                label,
                _factory(method, base=base, label=label, conditions=conds),
                task,
                probe_task,
            )
    elif kind == "duration":
        for task, seconds in DURATIONS.items():
            conds = REST if task == "rest" else ("video",)
            for s in seconds:
                label = f"{task}_{s}s"
                go(
                    label,
                    _factory(method, base=base, label=label, conditions=conds),
                    task,
                    part=f"first_{s}s",
                    seconds=s,
                )
    elif kind == "arousal":
        labels = pd.read_csv(
            Path(data_dir) / "labels_train.csv", na_values=["TESTSUBJECT"]
        )
        for label, keep in arousal_split(labels).items():
            go(
                label,
                _factory(method, base=base, label=label, conditions=("video",)),
                "video",
                the_store=CombinedStore(store.store, keep),
            )
    else:
        raise ValueError(f"unknown ablation {kind!r}")
    return records
