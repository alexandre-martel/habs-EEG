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
