"""Evaluate any ``ReIDModel`` on the shared folds, with controls, and log the run."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from reid.config import RESULTS, SEED
from reid.eval import controls, metrics
from reid.eval.interface import ReIDModel
from reid.eval.matching import truth_index
from reid.eval.protocol import FeatureStore, Split, build_split

PACKAGES = ("numpy", "scipy", "scikit-learn", "mne", "pyriemann", "neurokit2", "torch")


def _check_no_leak(split: Split) -> None:
    """fit must never see a session of a test person."""
    test = set(split.gallery.pids) | set(split.probe.pids)
    leaked = test & set(split.train.pids)
    assert not leaked, f"test people in training set: {sorted(leaked)[:5]}"


def _score_folds(
    make_model: Callable[[], ReIDModel],
    inv: pd.DataFrame,
    folds: list[dict],
    scenario: str,
    store: FeatureStore | None,
    **split_kw,
) -> tuple[list[metrics.Block], list[tuple[np.ndarray, pd.DataFrame]]]:
    """Fit and score every fold. Returns blocks and gallery embeddings (if any).
    ``split_kw``: task, probe_task, part (see ``build_split``)."""
    blocks, embeddings = [], []
    for fold in folds:
        split = build_split(inv, fold, scenario, store=store, **split_kw)
        if not len(split.probe):
            continue
        _check_no_leak(split)
        model = make_model()
        model.fit(split.train)
        S = np.asarray(model.score(split.gallery, split.probe), dtype=float)
        assert S.shape == (len(split.probe), len(split.gallery)), S.shape
        if scenario == "within_cohort":
            S = controls.mask_cohorts(
                S, split.gallery.sessions.cohort, split.probe.sessions.cohort
            )
        truth = truth_index(split.gallery.pids, split.probe.pids)
        one_to_one = set(split.gallery.pids) == set(split.probe.pids)
        blocks.append(metrics.Block(S, truth, one_to_one))
        try:  # session embeddings, kept for the cohort-predictability control
            embeddings.append((model.embed(split.gallery), split.gallery.sessions))
        except (AttributeError, NotImplementedError):
            pass
    return blocks, embeddings


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def evaluate(
    make_model: Callable[[], ReIDModel],
    inv: pd.DataFrame,
    folds: list[dict],
    scenario: str = "d1_to_d2",
    task: str = "rest",
    probe_task: str | None = None,
    store: FeatureStore | None = None,
    n_boot: int = 1000,
    run_controls: bool = True,
    save: bool = True,
    part: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Run one model on one scenario; returns (and saves) the run record.
    ``part`` restricts gallery/probe windows; ``extra`` is stored in the record
    (e.g. ablation labels)."""
    start = time.time()
    split_kw = {"task": task, "probe_task": probe_task, "part": part}
    blocks, embeddings = _score_folds(
        make_model, inv, folds, scenario, store, **split_kw
    )
    if not blocks:
        raise ValueError(f"no probe session for scenario {scenario!r} / task {task!r}")
    model = make_model()
    name = model.name
    record = {
        "model": name,
        "scenario": scenario,
        "task": task,
        "probe_task": probe_task or task,
        "part": part,
        **(extra or {}),
        "date": datetime.now().isoformat(timespec="seconds"),
        "git": _git_hash(),
        "seed": SEED,
        "config": getattr(model, "config", {}),
        "per_fold": [metrics.summarize([b], n_boot=0) for b in blocks],
        "pooled": metrics.summarize(blocks, n_boot=n_boot),
        "curves": metrics.curves(blocks),
    }
    if scenario in ("d1_to_d2", "d1_to_d3", "d2_to_d3"):
        record["gallery_size"] = metrics.gallery_size_curve(blocks)
    if run_controls:
        record["controls"] = _controls(
            blocks, embeddings, inv, folds, scenario, split_kw
        )
    record["time_s"] = round(time.time() - start, 1)
    record["versions"] = _versions()
    if save:
        save_run(record)
    return record


def _controls(blocks, embeddings, inv, folds, scenario, split_kw):
    """Shuffled identities, demographics, cohort predictability."""
    shuffled = [
        metrics.Block(controls.shuffle_gallery(b.S, SEED + k), b.truth, b.one_to_one)
        for k, b in enumerate(blocks)
    ]
    out = {"shuffled": _short(metrics.summarize(shuffled, n_boot=200))}
    if "age" in inv and inv.age.notna().any():
        demo, _ = _score_folds(
            controls.DemographicModel, inv, folds, scenario, None, **split_kw
        )
        out["demographics"] = _short(metrics.summarize(demo, n_boot=200))
    if embeddings and "cohort" in inv and inv.cohort.notna().any():
        X = np.vstack([e for e, _ in embeddings])
        sessions = pd.concat([s for _, s in embeddings])
        out["cohort_prediction"] = controls.cohort_predictability(
            X, sessions.cohort.to_numpy(float), sessions.participant_id.to_numpy()
        )
    return out


def _short(summary: dict) -> dict:
    keep = ("top1", "top5", "hungarian", "eer", "chance_top1")
    return {k: summary[k] for k in keep if k in summary} | {
        "n_probes": summary["n_probes"]
    }


def save_run(record: dict) -> None:
    """results/runs/<date>_<model>_<scenario>_<task>[_<part>].json"""
    runs = RESULTS / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    tasks = record["task"] + (
        f"-to-{record['probe_task']}" if record["probe_task"] != record["task"] else ""
    )
    stamp = record["date"].replace(":", "").replace("-", "")
    part = f"_{record['part']}" if record.get("part") else ""
    path = runs / f"{stamp}_{record['model']}_{record['scenario']}_{tasks}{part}.json"
    path.write_text(json.dumps(record, indent=1, default=float))
    print(f"run saved -> {path.relative_to(RESULTS.parent)}")
