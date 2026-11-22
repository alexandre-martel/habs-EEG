"""Re-identification entry point for the HABS platform: process(data_dir, out_dir).

The team's process.py delegates here (``from reid.platform import process``)
when a re-identification job is run. What runs is described by reid_job.yaml:

* ``job: features`` -- feature extraction only, written to out_dir/features
  (the cache that ``scripts/reid_run.py`` reads), shardable;
* ``job: final`` -- everything inside one job, since a job cannot read another
  job's output: feature and raw-crop caches extracted into local scratch (not
  out_dir: they would be uploaded) by ``workers`` independent
  ``scripts/reid_extract.py`` processes (one shard each, one compute thread
  each), then each group of ``groups`` run through the common evaluation in its
  own process (groups run concurrently; models of a group share fitted
  members). Run files go to out_dir/runs, logs to out_dir/logs, plus
  out_dir/job.json (hardware, versions, timings, extraction counts).

The platform kills a job after one hour, and a killed job keeps nothing. So
everything stops at ``deadline_min`` (default 52) minutes: unfinished
processes are terminated and job.json is still written; run files of models
that finished before are kept.

Extraction used to fork a process pool from a parent that had already loaded
torch and CUDA: on the H100 it stalled (one session in an hour, where the Mac
extracts 590 in 18 min). Separate processes inherit nothing.

Locally: habs run --data ./sample --out ./out
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
JOB = Path(os.environ.get("REID_JOB") or REPO / "reid_job.yaml")
SCRIPTS = REPO / "scripts"
#: one compute thread per process unless said otherwise: 40 extraction
#: processes x 40 BLAS/OpenMP threads each would be 1600 threads on 40 CPUs
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _threads(n: int) -> dict:
    return {v: str(max(int(n), 1)) for v in THREAD_VARS}


def _remaining(deadline: float) -> float:
    return max(deadline - time.time(), 0.0)


def _extract_one(cmd: list, log: Path, deadline: float) -> tuple[dict, str]:
    """One shard of one cache, in its own process: (counts, status)."""
    env = os.environ | _threads(1)
    with log.open("w") as fh:
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, text=True, stdout=fh, stderr=subprocess.STDOUT
        )
        try:
            code = proc.wait(timeout=_remaining(deadline))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return _last_counts(log), "timeout"
    return _last_counts(log), "ok" if code == 0 else f"exit {code}"


def _last_counts(log: Path) -> dict:
    """Last counts printed by scripts/reid_extract.py (they are cumulative, so
    a shard stopped by the deadline still reports what it extracted)."""
    counts = {}
    for line in log.read_text().splitlines():
        at = line.find(": {")
        if at >= 0:
            try:
                counts = ast.literal_eval(line[at + 2 :])
            except (ValueError, SyntaxError):
                pass
    return counts


def _extract(data_dir: Path, cache: Path, logs: Path, job: dict, deadline: float):
    """Both caches of every task, ``workers`` shards per cache, ``workers``
    processes at a time. Returns ({task: {kind: counts}}, failed shards)."""
    n = int(job.get("workers") or os.cpu_count() or 1)
    logs.mkdir(parents=True, exist_ok=True)
    counts, problems = {}, []
    with ThreadPoolExecutor(n) as pool:
        for task in job["tasks"]:
            futures = {}
            for kind in ("features", "raw"):
                for i in range(n):
                    cmd = [sys.executable, str(SCRIPTS / "reid_extract.py")]
                    cmd += ["--data", str(data_dir), "--task", task]
                    cmd += ["--out", str(cache / kind), "--shard", f"{i}/{n}"]
                    if kind == "raw":
                        cmd.append("--raw")
                    else:
                        cmd += ["--modalities", ",".join(job["modalities"])]
                    log = logs / f"extract_{task}_{kind}_{i}.log"
                    futures[(kind, i)] = pool.submit(_extract_one, cmd, log, deadline)
            total = {"features": {}, "raw": {}}
            for (kind, i), f in futures.items():
                c, status = f.result()
                if status != "ok":
                    problems.append(f"{task} {kind} shard {i}/{n}: {status}")
                for k, v in c.items():
                    total[kind][k] = total[kind].get(k, 0) + v
            counts[task] = total
            print(f"{task}: {total}", flush=True)
    for p in problems:
        print(f"extraction problem: {p}", flush=True)
    return counts, problems


def _pump(stream, prefix: str, log) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        print(f"[{prefix}] {line}", end="", flush=True)


def _run_groups(
    data_dir: Path, cache: Path, out_dir: Path, job: dict, deadline: float
) -> list:
    """One ``scripts/reid_run.py`` process per group, all at once; the CPUs
    are shared between groups. A group still running at the deadline is
    terminated (its finished models are already saved)."""
    per_group = (os.cpu_count() or 1) // max(len(job["groups"]), 1)
    env = os.environ | _threads(per_group)
    env |= {"REID_CACHE": str(cache), "REID_RUNS": str(out_dir / "runs")}
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for k, models in enumerate(job["groups"]):
        cmd = [sys.executable, str(SCRIPTS / "reid_run.py")]
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
        try:
            codes.append(proc.wait(timeout=_remaining(deadline)))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            codes.append("timeout")
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
    start = time.time()
    deadline = start + 60 * float(job.get("deadline_min", 52))
    report = {"job": job, "hardware": _hardware(), "timings_s": {}}
    print(json.dumps(report["hardware"]), flush=True)

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
    report["extraction"], report["extraction_problems"] = _extract(
        data_dir, cache, out_dir / "logs", job, deadline
    )
    report["timings_s"]["extraction"] = round(time.time() - start, 1)
    print(f"extraction done in {report['timings_s']['extraction']} s", flush=True)
    for kind in ("features", "raw"):  # failure lists are small: keep them
        failures = cache / kind / "failures.csv"
        if failures.is_file():
            shutil.copy(failures, out_dir / f"failures_{kind}.csv")
    (out_dir / "job.json").write_text(json.dumps(report, indent=1, default=str))

    report["exit_codes"] = _run_groups(data_dir, cache, out_dir, job, deadline)
    report["timings_s"]["total"] = round(time.time() - start, 1)
    (out_dir / "job.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"done: {report['timings_s']}, exit codes {report['exit_codes']}")
