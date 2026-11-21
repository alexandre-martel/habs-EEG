"""Re-identification entry point for the HABS platform: process(data_dir, out_dir).

The team's process.py delegates here (``from reid.platform import process``)
when a re-identification job is run. What runs is described by reid_job.yaml:

* ``job: features`` -- feature extraction only, written to out_dir/features
  (the cache that ``scripts/reid_run.py`` reads), shardable;
* ``job: final`` -- everything inside one job, since a job cannot read another
  job's output: feature and raw-crop caches extracted into local scratch (not
  out_dir: they would be uploaded), in parallel over ``workers`` processes,
  then each group of ``groups`` run through the common evaluation in its own
  process (groups run concurrently; models of a group share fitted members).
  Run files go to out_dir/runs, logs to out_dir/logs, plus out_dir/job.json
  (hardware, versions, timings, extraction counts).

Locally: habs run --data ./sample --out ./out
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
JOB = REPO / "reid_job.yaml"


def _extract_shard(data_dir: str, cache: str, task: str, modalities, shard) -> dict:
    from reid.features import run_extraction
    from reid.final.raw import run_raw_extraction

    return {
        "features": run_extraction(
            Path(data_dir), Path(cache) / "features", task, modalities, shard
        ),
        "raw": run_raw_extraction(Path(data_dir), Path(cache) / "raw", task, shard),
    }


def _extract(data_dir: Path, cache: Path, job: dict) -> dict:
    """Both caches of every task, ``workers`` shards in parallel."""
    n = int(job.get("workers") or os.cpu_count() or 1)
    counts = {}
    # fork on Linux (before any CUDA use): does not depend on how the platform
    # started Python, unlike spawn/forkserver which re-import __main__
    ctx = multiprocessing.get_context("fork" if sys.platform == "linux" else None)
    with ProcessPoolExecutor(n, mp_context=ctx) as pool:
        for task in job["tasks"]:
            futures = [
                pool.submit(
                    _extract_shard,
                    str(data_dir),
                    str(cache),
                    task,
                    job["modalities"],
                    (i, n),
                )
                for i in range(n)
            ]
            total = {"features": {}, "raw": {}}
            for f in futures:
                for kind, c in f.result().items():
                    for k, v in c.items():
                        total[kind][k] = total[kind].get(k, 0) + v
            counts[task] = total
            print(f"{task}: {total}", flush=True)
    return counts


def _pump(stream, prefix: str, log) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        print(f"[{prefix}] {line}", end="", flush=True)


def _run_groups(data_dir: Path, cache: Path, out_dir: Path, job: dict) -> list:
    """One ``scripts/reid_run.py`` process per group, all at once."""
    env = os.environ | {"REID_CACHE": str(cache), "REID_RUNS": str(out_dir / "runs")}
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for k, models in enumerate(job["groups"]):
        cmd = [sys.executable, str(REPO / "scripts" / "reid_run.py")]
        cmd += ["--data", str(data_dir), "--task", job.get("eval_task", "rest")]
        for m in models:
            cmd += ["--model", m]
        for s in job.get("scenarios", ["d1_to_d2"]):
            cmd += ["--scenario", s]
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log = open(logs / f"group{k}.log", "w")  # noqa: SIM115 - closed below
        thread = threading.Thread(target=_pump, args=(proc.stdout, f"g{k}", log))
        thread.start()
        procs.append((proc, thread, log))
    codes = []
    for proc, thread, log in procs:
        codes.append(proc.wait())
        thread.join()
        log.close()
    return codes


def _hardware() -> dict:
    info = {"cpus": os.cpu_count(), "python": sys.version.split()[0]}
    try:
        import torch

        info |= {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def process(data_dir: Path, out_dir: Path) -> None:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = yaml.safe_load(JOB.read_text())
    report = {"job": job, "hardware": _hardware(), "timings_s": {}}
    print(json.dumps(report["hardware"]), flush=True)
    start = time.time()

    if job.get("job", "features") == "features":
        from reid.features import run_extraction

        shard = tuple(int(v) for v in str(job.get("shard", "0/1")).split("/"))
        for task in job["tasks"]:
            counts = run_extraction(
                data_dir, out_dir / "features", task, job["modalities"], shard
            )
            print(f"{task}: {counts}", flush=True)
        return

    cache = Path(job.get("scratch") or tempfile.gettempdir()) / "reid_cache"
    report["extraction"] = _extract(data_dir, cache, job)
    report["timings_s"]["extraction"] = round(time.time() - start, 1)
    for kind in ("features", "raw"):  # failure lists are small: keep them
        failures = cache / kind / "failures.csv"
        if failures.is_file():
            shutil.copy(failures, out_dir / f"failures_{kind}.csv")
    (out_dir / "job.json").write_text(json.dumps(report, indent=1, default=str))

    report["exit_codes"] = _run_groups(data_dir, cache, out_dir, job)
    report["timings_s"]["total"] = round(time.time() - start, 1)
    (out_dir / "job.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"done: {report['timings_s']}, exit codes {report['exit_codes']}")
