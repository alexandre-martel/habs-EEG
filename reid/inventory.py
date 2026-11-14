"""Session inventory built from file names and metadata.json, never from BDF content.

Two sources of file listings:

* a local dataset root (``CURATED/<PID>/...``): metadata.json files are read for
  the exact segment durations;
* the Hugging Face file listing (public, no download): only names and sizes,
  durations are then estimated from the BDF size (``duration_source="bdf_size"``).

The committed inventory only holds what the public file listing already
shows (IDs, days, tasks, durations). Nothing from the participant tables
(cohort, age, gender) nor the protocol name (some names reveal gender, e.g.
``PerfumFemme2025``) is written to the public repository: ``load_inventory``
joins them at run time from ``participants_train.csv``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from reid.config import HF_REPO, RESULTS, TASKS, default_data_dir

FILE_RE = re.compile(
    r"(?:^|/)CURATED/(?P<pid>P\d{3})/(?P=pid)_day-(?P<day>\d+)_"
    r"(?:task-(?P<task>[a-z-]+)_multimodal\.(?P<ext>bdf|record_map)|metadata\.json)$"
)

#: Linear fit size = offset + rate * duration, calibrated on the two P001 sample
#: BDFs (95.079 s / 20 093 632 B and 212.322 s / 44 556 226 B). Planning aid only.
BDF_BYTES_PER_S = 208_646.3
BDF_OFFSET_BYTES = 255_750.0

COLUMNS = [
    "participant_id",
    "day",
    "has_rest",
    "has_artefact",
    "has_video",
    "rest_duration_s",
    "artefact_duration_s",
    "duration_source",
    "rest_mb",
    "artefact_mb",
    "video_mb",
]


def parse_listing(entries: Iterable[tuple[str, int]]) -> pd.DataFrame:
    """Parse (path, size) pairs into one row per recognised CURATED file."""
    rows = []
    for path, size in entries:
        match = FILE_RE.search(str(path).replace("\\", "/"))
        if match:
            kind = match["ext"] or "metadata"
            rows.append(
                {
                    "participant_id": match["pid"],
                    "day": int(match["day"]),
                    "task": match["task"],
                    "kind": kind,
                    "path": str(path),
                    "size": int(size),
                }
            )
    return pd.DataFrame(
        rows, columns=["participant_id", "day", "task", "kind", "path", "size"]
    )


def list_local(data_dir: Path) -> list[tuple[str, int]]:
    """(path, size) of every file under ``data_dir/CURATED``."""
    return [
        (str(p), p.stat().st_size)
        for p in sorted((Path(data_dir) / "CURATED").glob("P*/*"))
        if p.is_file()
    ]


def list_hf(repo_id: str = HF_REPO) -> list[tuple[str, int]]:
    """(path, size) of every CURATED file on the Hugging Face hub (public listing)."""
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(
        repo_id, repo_type="dataset", path_in_repo="CURATED", recursive=True
    )
    return [(e.path, e.size) for e in tree if getattr(e, "size", None) is not None]


def estimate_duration_s(size_bytes: float) -> float:
    """Approximate recording duration from the BDF file size."""
    return round((size_bytes - BDF_OFFSET_BYTES) / BDF_BYTES_PER_S, 1)


def read_metadata(path: Path) -> dict:
    """Protocol and per-task segment durations from a session metadata.json."""
    export = json.loads(Path(path).read_text(encoding="utf-8"))["dataset_export"]
    durations = {
        task: seg.get("duration_s") for task, seg in export.get("segments", {}).items()
    }
    return {"protocol": export.get("protocol"), "durations": durations}


def load_participants(data_dir: Path | None) -> pd.DataFrame | None:
    """participants_train.csv with the held-out sentinel read as missing."""
    path = Path(data_dir) / "participants_train.csv" if data_dir else None
    if path is None or not path.is_file():
        return None
    return pd.read_csv(path, na_values=["TESTSUBJECT"])


def _session_row(files: pd.DataFrame, local: bool) -> dict:
    """One inventory row from all files of a single (participant, day)."""
    bdf = files[files.kind == "bdf"].set_index("task")["size"]
    row: dict = {f"has_{flag}": task in bdf.index for task, flag in TASKS.items()}
    for task, flag in TASKS.items():
        row[f"{flag}_mb"] = round(bdf[task] / 1e6, 1) if task in bdf.index else None
    meta = files[files.kind == "metadata"]
    info = read_metadata(Path(meta.path.iloc[0])) if local and len(meta) else None
    row["duration_source"] = "metadata" if info else "bdf_size"
    for task, flag in (("resting-state", "rest"), ("artefact", "artefact")):
        if task not in bdf.index:
            row[f"{flag}_duration_s"] = None
        elif info:
            duration = info["durations"].get(task)
            row[f"{flag}_duration_s"] = round(duration, 3) if duration else None
        else:
            row[f"{flag}_duration_s"] = estimate_duration_s(bdf[task])
    return row


def build_inventory(files: pd.DataFrame, local: bool) -> pd.DataFrame:
    """One row per session (participant x day) with available tasks and durations."""
    rows = [
        {"participant_id": pid, "day": day, **_session_row(group, local)}
        for (pid, day), group in files.groupby(["participant_id", "day"])
    ]
    inv = pd.DataFrame(rows)[COLUMNS]
    return inv.sort_values(["participant_id", "day"]).reset_index(drop=True)


def attach_participants(
    inv: pd.DataFrame, participants: pd.DataFrame | None
) -> pd.DataFrame:
    """Add cohort, age and gender (all NA when the table is unavailable)."""
    inv = inv.copy()
    table = (
        participants.drop_duplicates("participant_id").set_index("participant_id")
        if participants is not None
        else pd.DataFrame(columns=["cohort", "age", "gender"])
    )
    inv["cohort"] = inv.participant_id.map(table["cohort"]).astype("Int64")
    inv["age"] = inv.participant_id.map(table["age"]).astype("Float64")
    inv["gender"] = inv.participant_id.map(table["gender"]).astype("string")
    return inv


def load_inventory(
    data_dir: Path | None = None, path: Path = RESULTS / "inventory.csv"
) -> pd.DataFrame:
    """Committed inventory joined at run time with the participant table."""
    inv = pd.read_csv(path)
    return attach_participants(inv, load_participants(data_dir or default_data_dir()))


def day_patterns(inv: pd.DataFrame) -> pd.DataFrame:
    """Number of people per cohort and per set of available days ("1+2", ...)."""
    if "cohort" not in inv:
        inv = inv.assign(cohort=pd.NA)
    people = inv.groupby("participant_id").agg(
        cohort=("cohort", "first"),
        days=("day", lambda d: "+".join(map(str, sorted(d)))),
        n_days=("day", "nunique"),
    )
    people["cohort"] = people.cohort.astype("string").fillna("?")
    table = pd.crosstab(people.cohort, people.days).add_prefix("days ")
    per_n = pd.crosstab(people.cohort, people.n_days)
    for n in per_n.columns:
        table[f"{n} day(s)"] = per_n[n]
    table.loc["total"] = table.sum()
    return table


def pair_counts(inv: pd.DataFrame) -> pd.DataFrame:
    """People usable for each gallery->probe day pair, per task present on both days."""
    pairs = {"d1_to_d2": (1, 2), "d1_to_d3": (1, 3), "d2_to_d3": (2, 3)}
    out = {}
    for name, (g, p) in pairs.items():
        both = inv[inv.day == g].merge(
            inv[inv.day == p], on="participant_id", suffixes=("_g", "_p")
        )
        out[name] = {
            flag: int((both[f"has_{flag}_g"] & both[f"has_{flag}_p"]).sum())
            for flag in TASKS.values()
        }
        out[name]["any"] = len(both)
    return pd.DataFrame(out).T
