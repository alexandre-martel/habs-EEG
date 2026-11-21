"""Inventory tests on synthetic file trees and on the bundled P001 sample."""

import json
from pathlib import Path

import pandas as pd
import pytest

from reid import inventory as inv_mod
from reid.config import REPO


def _write_session(root: Path, pid: str, day: int, tasks: dict[str, float]) -> None:
    """Fake CURATED session: empty BDF/record_map files plus a metadata.json."""
    folder = root / "CURATED" / pid
    folder.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for ext in ("bdf", "record_map"):
            (folder / f"{pid}_day-{day}_task-{task}_multimodal.{ext}").touch()
    segments = {task: {"duration_s": dur} for task, dur in tasks.items()}
    meta = {"events": [], "dataset_export": {"protocol": "X", "segments": segments}}
    (folder / f"{pid}_day-{day}_metadata.json").write_text(json.dumps(meta))


def test_parse_listing_ignores_unrelated_files() -> None:
    files = inv_mod.parse_listing(
        [
            ("CURATED/P010/P010_day-2_task-resting-state_multimodal.bdf", 10),
            ("CURATED/P010/P010_day-2_metadata.json", 1),
            ("PREPROCESSED/P010/P010_day-2_metadata.json", 1),
            ("CURATED/P010/P011_day-2_metadata.json", 1),  # ID mismatch
            ("videos/x.mp4", 5),
        ]
    )
    assert files[["participant_id", "day", "kind"]].values.tolist() == [
        ["P010", 2, "bdf"],
        ["P010", 2, "metadata"],
    ]


def test_build_inventory_local(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "P001",
        1,
        {"resting-state": 95.0, "artefact": 212.0, "emotion-video": 800.0},
    )
    _write_session(tmp_path, "P001", 2, {"resting-state": 94.0, "emotion-video": 800.0})
    _write_session(tmp_path, "P002", 1, {"emotion-video": 800.0})
    (tmp_path / "participants_train.csv").write_text(
        "cohort,participant_id,age,gender\n1,P001,30,F\nTESTSUBJECT,P002,TESTSUBJECT,"
        "TESTSUBJECT\n"
    )

    files = inv_mod.parse_listing(inv_mod.list_local(tmp_path))
    inv = inv_mod.build_inventory(files, local=True)
    assert list(inv.columns) == inv_mod.COLUMNS
    assert not {"cohort", "age", "gender", "protocol"} & set(inv.columns)

    inv = inv_mod.attach_participants(inv, inv_mod.load_participants(tmp_path))
    p1d2 = inv.query("participant_id == 'P001' and day == 2").iloc[0]
    assert p1d2.has_rest and not p1d2.has_artefact and p1d2.rest_duration_s == 94.0
    assert inv.cohort.iloc[:2].tolist() == [1, 1] and pd.isna(inv.cohort.iloc[2])
    assert inv_mod.pair_counts(inv).loc["d1_to_d2"].to_dict() == {
        "rest": 1,
        "artefact": 0,
        "video": 1,
        "any": 1,
    }
    patterns = inv_mod.day_patterns(inv)
    assert (
        patterns.loc["total", "2 day(s)"] == 1
        and patterns.loc["total", "1 day(s)"] == 1
    )


@pytest.mark.parametrize(
    ("size", "duration"), [(20_093_632, 95.079), (44_556_226, 212.322)]
)
def test_duration_estimate_matches_sample(size: int, duration: float) -> None:
    assert abs(inv_mod.estimate_duration_s(size) - duration) < 0.5


@pytest.mark.skipif(not (REPO / "sample" / "CURATED").is_dir(), reason="no sample/")
def test_sample_p001() -> None:
    files = inv_mod.parse_listing(inv_mod.list_local(REPO / "sample"))
    inv = inv_mod.build_inventory(files, local=True)
    row = inv.iloc[0]
    assert (len(inv), row.participant_id, row.day) == (1, "P001", 1)
    assert row.has_rest and row.has_artefact and not row.has_video
    assert row.rest_duration_s == pytest.approx(95.079)
