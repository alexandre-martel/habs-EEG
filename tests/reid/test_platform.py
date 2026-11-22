"""HABS job runner: extraction counts and the deadline (reid/platform.py)."""

import sys
import time

from reid.platform import _extract_one, _last_counts


def test_last_counts_keeps_the_latest_cumulative_line(tmp_path):
    log = tmp_path / "shard.log"
    log.write_text(
        "rest P001 day-1: {'done': 1, 'skipped': 0, 'failed': 0}\n"
        "some warning: not a dict\n"
        "rest P003 day-2: {'done': 2, 'skipped': 0, 'failed': 1}\n"
    )
    assert _last_counts(log) == {"done": 2, "skipped": 0, "failed": 1}
    log.write_text("rest: {'done': 5, 'skipped': 1, 'failed': 0}\n")
    assert _last_counts(log) == {"done": 5, "skipped": 1, "failed": 0}
    log.write_text("nothing yet\n")
    assert _last_counts(log) == {}


def test_extract_one_stops_at_the_deadline_and_keeps_counts(tmp_path):
    code = (
        "print(\"rest P001 day-1: {'done': 1, 'skipped': 0, 'failed': 0}\", "
        "flush=True); import time; time.sleep(30)"
    )
    start = time.time()
    counts, status = _extract_one(
        [sys.executable, "-c", code], tmp_path / "a.log", time.time() + 2
    )
    assert status == "timeout" and time.time() - start < 10
    assert counts == {"done": 1, "skipped": 0, "failed": 0}

    counts, status = _extract_one(
        [sys.executable, "-c", "print(\"rest: {'done': 3}\")"],
        tmp_path / "b.log",
        time.time() + 30,
    )
    assert (counts, status) == ({"done": 3}, "ok")
    counts, status = _extract_one(
        [sys.executable, "-c", "raise SystemExit(3)"],
        tmp_path / "c.log",
        time.time() + 30,
    )
    assert status == "exit 3"


def test_select_part_from_file_or_default(tmp_path):
    from reid.platform import select_part

    job = {"part": "a", "parts": {"a": [["m1"]], "b": [["m2"], ["m3"]]}, "x": 1}
    missing = tmp_path / "none"
    assert select_part(job, missing) == {"part": "a", "groups": [["m1"]], "x": 1}
    chosen = tmp_path / "part"
    chosen.write_text("b\n")
    assert select_part(job, chosen)["groups"] == [["m2"], ["m3"]]
    chosen.write_text("zzz")
    try:
        select_part(job, chosen)
    except SystemExit as err:
        assert "unknown part" in str(err)
    else:
        raise AssertionError("an unknown part must stop the job")
    assert select_part({"groups": [["m"]]}, chosen) == {"groups": [["m"]]}


def test_every_model_of_the_job_file_is_registered():
    import yaml

    from reid.baselines import MODELS as BASELINES
    from reid.final import MODELS as FINAL
    from reid.platform import JOB

    job = yaml.safe_load(JOB.read_text())
    assert job["part"] in job["parts"]
    names = {m for groups in job["parts"].values() for g in groups for m in g}
    assert names and not names - set(BASELINES | FINAL), names
