"""Session sets, feature store, cross-validation folds and evaluation scenarios.

Conventions
-----------
* A *session* is one (participant, day, task). Each session's cached features
  are a dict of arrays; keys starting with ``w_`` have one row per analysis
  window (10 s by default), the others are shared metadata (freqs, channels...).
* A fold holds out a set of people: multi-day people are matched gallery ->
  probe, single-day people are optional gallery distractors. ``fit`` only ever
  sees sessions of the *other* folds' people, so it never sees a test probe
  (nor, more strictly than required, the test gallery: enrolment happens in
  ``score``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from reid.config import FEATURES, RESULTS, SEED

FOLDS_PATH = RESULTS / "splits" / "folds.json"

#: name -> (gallery day, probe day). The same-day ceiling matches the first half
#: of day 1 against its second half: an optimistic ceiling, never a result.
SCENARIOS = {
    "d1_to_d2": (1, 2),
    "d1_to_d3": (1, 3),
    "d2_to_d3": (2, 3),
    "within_cohort": (1, 2),
    "with_distractors": (1, 2),
    "same_day_ceiling": (1, 1),
}


class FeatureStore:
    """Lazy, memoised access to ``<root>/<task>/<PID>_day-<d>.npz``."""

    def __init__(self, root: Path = FEATURES) -> None:
        self.root = Path(root)
        self.load = lru_cache(maxsize=2048)(self._load)

    def path(self, pid: str, day: int, task: str) -> Path:
        return self.root / task / f"{pid}_day-{day}.npz"

    def _load(self, pid: str, day: int, task: str) -> dict:
        with np.load(self.path(pid, day, task), allow_pickle=False) as npz:
            return dict(npz)

    def available(self, pid: str, day: int, task: str) -> bool:
        return self.path(pid, day, task).is_file()


def select_windows(feats: dict, part: str | None) -> dict:
    """Keep a subset of windows: ``first_half``/``second_half`` (per condition,
    so both halves keep every condition) or ``first_<N>s`` (first N seconds of
    each condition)."""
    if not part:
        return feats
    cond = feats["w_condition"]
    keep = np.zeros(len(cond), bool)
    for c in np.unique(cond):
        idx = np.flatnonzero(cond == c)
        if part == "first_half":
            keep[idx[: len(idx) // 2]] = True
        elif part == "second_half":
            keep[idx[len(idx) // 2 :]] = True
        elif part.startswith("first_") and part.endswith("s"):
            seconds = float(part[6:-1])
            dur = feats["w_duration"][idx]
            keep[idx[np.cumsum(dur) <= seconds + 1e-6]] = True
        else:
            raise ValueError(f"unknown window selection {part!r}")
    return {k: (v[keep] if k.startswith("w_") else v) for k, v in feats.items()}


@dataclass
class SessionSet:
    """Sessions (one row each: participant_id, day, task, cohort, age, gender,
    part) with lazy access to their features through a store."""

    sessions: pd.DataFrame
    store: FeatureStore | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.sessions)

    @property
    def pids(self) -> np.ndarray:
        return self.sessions.participant_id.to_numpy()

    def load(self, i: int) -> dict:
        """Features of the i-th session, restricted to its ``part``."""
        row = self.sessions.iloc[i]
        feats = self.store.load(row.participant_id, int(row.day), row.task)
        part = row.get("part")
        return select_windows(feats, part if isinstance(part, str) else None)

    def __iter__(self):
        return (self.load(i) for i in range(len(self)))

    def subset(self, mask: np.ndarray) -> SessionSet:
        return SessionSet(self.sessions[mask].reset_index(drop=True), self.store)


@dataclass
class Split:
    train: SessionSet
    gallery: SessionSet
    probe: SessionSet


def make_folds(inv: pd.DataFrame, n_folds: int = 5, seed: int = SEED) -> dict:
    """Assign people to folds (stratified by cohort when known and large enough).

    Multi-day people are the test people of their fold; single-day people are
    spread over the folds too and act as that fold's gallery distractors (they
    are training data for the other folds).
    """
    if "cohort" not in inv:
        inv = inv.assign(cohort=pd.NA)
    people = inv.groupby("participant_id").agg(
        n_days=("day", "nunique"), cohort=("cohort", "first")
    )
    folds = [{"test": [], "distractors": []} for _ in range(n_folds)]
    for key, multi in (("test", True), ("distractors", False)):
        group = people[(people.n_days >= 2) == multi]
        pids = group.index.to_numpy()
        strata = group.cohort.astype("string").fillna("?").to_numpy()
        counts = pd.Series(strata).value_counts()
        if counts.min() >= n_folds and len(counts) > 1:
            splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        else:
            splitter = KFold(n_folds, shuffle=True, random_state=seed)
        for k, (_, idx) in enumerate(splitter.split(pids, strata)):
            folds[k][key] = sorted(pids[idx].tolist())
    return {"seed": seed, "n_folds": n_folds, "folds": folds}


def save_folds(folds: dict, path: Path = FOLDS_PATH, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists: the whole team must use the same one.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(folds, indent=1))


def load_folds(path: Path = FOLDS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["folds"]


def _sessions(
    inv: pd.DataFrame, task: str, store: FeatureStore | None, part: str | None = None
) -> pd.DataFrame:
    """Sessions of ``inv`` that have ``task`` (and cached features, if a store)."""
    rows = inv[inv[f"has_{task}"]].copy()
    rows["task"], rows["part"] = task, part
    if store is not None:  # drop sessions whose feature extraction failed
        ok = [
            store.available(r.participant_id, int(r.day), task)
            for r in rows.itertuples()
        ]
        rows = rows[np.asarray(ok, dtype=bool)]
    return rows


def build_split(
    inv: pd.DataFrame,
    fold: dict,
    scenario: str,
    task: str = "rest",
    probe_task: str | None = None,
    store: FeatureStore | None = None,
) -> Split:
    """Train / gallery / probe sets of one fold for one scenario.

    ``task`` is the gallery (and training) task; ``probe_task`` defaults to it
    and allows cross-task transfer (e.g. gallery = rest, probe = video).
    """
    g_day, p_day = SCENARIOS[scenario]
    probe_task = probe_task or task
    held = set(fold["test"]) | set(fold["distractors"])
    test = set(fold["test"])

    train = _sessions(inv[~inv.participant_id.isin(held)], task, store)
    halves = (
        ("first_half", "second_half") if scenario == "same_day_ceiling" else (None,) * 2
    )
    gal = _sessions(inv[inv.day == g_day], task, store, halves[0])
    prb = _sessions(inv[inv.day == p_day], probe_task, store, halves[1])
    gal, prb = gal[gal.participant_id.isin(test)], prb[prb.participant_id.isin(test)]
    both = set(gal.participant_id) & set(prb.participant_id)
    gal, prb = gal[gal.participant_id.isin(both)], prb[prb.participant_id.isin(both)]
    if scenario == "with_distractors":
        extra = _sessions(
            inv[inv.participant_id.isin(fold["distractors"])], task, store
        )
        gal = pd.concat([gal, extra.drop_duplicates("participant_id")])

    def as_set(df: pd.DataFrame) -> SessionSet:
        df = df.sort_values(["participant_id", "day"]).reset_index(drop=True)
        return SessionSet(df, store)

    return Split(as_set(train), as_set(gal), as_set(prb))
