"""Re-identification entry point for the HABS platform: process(data_dir, out_dir).

The repository's process.py belongs to the team's classification pipeline and is
left untouched. To run a re-identification job, process.py only has to delegate:

    from reid.platform import process  # noqa: F401

Runs the feature extraction described in reid_job.yaml (tasks, modalities,
shard) and writes out_dir/features/<task>/<PID>_day-<d>.npz, plus
out_dir/features/failures.csv for sessions that could not be processed.
Locally: habs run --data ./sample --out ./out
"""

from pathlib import Path

import yaml

from reid.features import run_extraction

JOB = Path(__file__).resolve().parent.parent / "reid_job.yaml"


def process(data_dir: Path, out_dir: Path) -> None:
    job = yaml.safe_load(JOB.read_text())
    shard = tuple(int(v) for v in str(job.get("shard", "0/1")).split("/"))
    for task in job["tasks"]:
        counts = run_extraction(
            Path(data_dir), Path(out_dir) / "features", task, job["modalities"], shard
        )
        print(f"{task}: {counts}", flush=True)
