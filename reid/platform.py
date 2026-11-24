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
#: written (and committed on the runner only) by .github/workflows/reid_gpu.yml
#: to pick one of the ``parts`` of reid_job.yaml without a commit on main
PART = REPO / "reid_job_part"
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
    """The ``caches`` of the job (features / raw for every task, video once),
    ``workers`` shards per cache, all queued at once on ``workers`` processes.
    Returns ({task or "video": {kind: counts}}, failed shards)."""
    n = int(job.get("workers") or os.cpu_count() or 1)
    caches = job.get("caches", ["features", "raw"])
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(t, k) for t in job["tasks"] for k in ("features", "raw") if k in caches]
    jobs += [("video", "video")] if "video" in caches else []
    counts, problems, futures = {}, [], {}
    with ThreadPoolExecutor(n) as pool:
        for task, kind in jobs:
            for i in range(n):
                cmd = [sys.executable, str(SCRIPTS / "reid_extract.py")]
                cmd += ["--data", str(data_dir), "--out", str(cache / kind)]
                cmd += ["--shard", f"{i}/{n}"]
                if kind == "video":
                    cmd.append("--video")
                else:
                    cmd += ["--task", task]
                if kind == "raw":
                    cmd.append("--raw")
                if kind == "features":
                    cmd += ["--modalities", ",".join(job["modalities"])]
                log = logs / f"extract_{task}_{kind}_{i}.log"
                futures[(task, kind, i)] = pool.submit(_extract_one, cmd, log, deadline)
        for (task, kind, i), f in futures.items():
            c, status = f.result()
            if status != "ok":
                problems.append(f"{task} {kind} shard {i}/{n}: {status}")
            total = counts.setdefault(task, {}).setdefault(kind, {})
            for k, v in c.items():
                total[k] = total.get(k, 0) + v
    for task, total in counts.items():
        print(f"{task}: {total}", flush=True)
    for p in problems:
        print(f"extraction problem: {p}", flush=True)
    return counts, problems


def _pump(stream, prefix: str, log) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        print(f"[{prefix}] {line}", end="", flush=True)


def _group_command(group: list, data_dir: Path, job: dict) -> list:
    """``[model, ...]``: evaluate the models (scripts/reid_run.py);
    ``["pretrain", model, fold, ...]``: precompute the model's pretraining for
    those folds (scripts/reid_pretrain.py), shared through the disk memo."""
    task = job.get("eval_task", "rest")
    if group and group[0] == "pretrain":
        cmd = [sys.executable, str(SCRIPTS / "reid_pretrain.py"), "--task", task]
        cmd += ["--data", str(data_dir), "--model", str(group[1])]
        for fold in group[2:]:
            cmd += ["--fold", str(fold)]
        return cmd
    cmd = [sys.executable, str(SCRIPTS / "reid_run.py")]
    cmd += ["--data", str(data_dir), "--task", task]
    for m in group:
        cmd += ["--model", m]
    for sc in job.get("scenarios", ["d1_to_d2"]):
        cmd += ["--scenario", sc]
    return cmd


def _run_groups(
    data_dir: Path, cache: Path, out_dir: Path, job: dict, deadline: float
) -> list:
    """One ``scripts/reid_run.py`` process per group, all at once; the CPUs
    are shared between groups. A group still running at the deadline is
    terminated (its finished models are already saved)."""
    per_group = (os.cpu_count() or 1) // max(len(job["groups"]), 1)
    env = os.environ | _threads(per_group)
    env |= {"REID_CACHE": str(cache), "REID_RUNS": str(out_dir / "runs")}
    env |= {"REID_SCORES": str(out_dir / "scores")}
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for k, models in enumerate(job["groups"]):
        cmd = _group_command(models, data_dir, job)
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
        info["ram_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1
        )
        info["tmp_free_gb"] = round(
            shutil.disk_usage(tempfile.gettempdir()).free / 1e9, 1
        )
    except (ValueError, OSError):
        pass
    try:
        import torch

        info |= {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def select_part(job: dict, part_file: Path = PART) -> dict:
    """``groups`` of the chosen part: the one named in ``part_file`` when it
    exists, else ``part``; a job without ``parts`` keeps its ``groups``."""
    if "parts" not in job:
        return job
    name = part_file.read_text().strip() if part_file.is_file() else ""
    name = name or str(job.get("part", ""))
    if name not in job["parts"]:
        raise SystemExit(f"unknown part {name!r}; known: {sorted(job['parts'])}")
    part = job["parts"][name]  # a list of groups, or a dict of overrides
    part = part if isinstance(part, dict) else {"groups": part}
    return {k: v for k, v in job.items() if k != "parts"} | part | {"part": name}


def _ecg_rest(data_dir: Path, out_dir: Path, job: dict, deadline: float) -> dict:
    """``job: ecg_rest`` -- scripts/ecg_rest_full.py on the resting-state BDFs
    of ``participants`` x ``days`` (CURATED). Per-participant values identify
    people and the repository is public: they go to out_dir/ecg_rest only
    (private team storage; the workflows leave that folder out of the GitHub
    artifact), the script's output goes to a file there, and the job log only
    gets counts."""
    dest = out_dir / "ecg_rest"
    dest.mkdir(parents=True, exist_ok=True)
    bdfs, missing = [], 0
    for pid in job.get("participants", []):
        for day in job.get("days", [1, 2]):
            bdf = data_dir / "CURATED" / pid / f"{pid}_day-{day}_task-resting-state_multimodal.bdf"
            if bdf.is_file():
                bdfs.append(bdf)
            else:
                missing += 1
    print(f"ecg_rest: {len(bdfs)} resting-state file(s) found, {missing} missing", flush=True)
    if not bdfs:
        return {"files": 0, "missing": missing, "status": "nothing to do"}
    cmd = [sys.executable, str(SCRIPTS / "ecg_rest_full.py"), *map(str, bdfs), "--out", str(dest)]
    with (dest / "log.txt").open("w") as log:
        try:
            code = subprocess.run(
                cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                timeout=_remaining(deadline),
            ).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    status = "ok" if code == 0 else f"exit {code}"
    print(f"ecg_rest: {status}, outputs in out_dir/ecg_rest (not in the job log)", flush=True)
    return {"files": len(bdfs), "missing": missing, "status": status}


def _ecg_tanaskovic(data_dir: Path, out_dir: Path, job: dict, deadline: float) -> dict:
    """``job: ecg_tanaskovic`` -- scripts/ecg_tanaskovic_extract.py on every
    resting-state BDF of ``days``. Output names participants: it goes to
    out_dir/private/ecg_tanaskovic (left out of the GitHub artifact by the
    workflows), with the script's output; the job log and job.json only get
    aggregate counts."""
    dest = out_dir / "private" / "ecg_tanaskovic"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPTS / "ecg_tanaskovic_extract.py"),
           "--data", str(data_dir), "--out", str(dest),
           "--days", *map(str, job.get("days", [1, 2]))]
    with (dest / "log.txt").open("w") as log:
        try:
            code = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=_remaining(deadline)).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    summary = {"status": "ok" if code == 0 else f"exit {code}"}
    try:
        import pandas as pd

        f = pd.read_csv(dest / "features.csv")
        summary |= {
            "sessions": int(f.groupby(["participant_id", "day"]).ngroups),
            "participants": int(f.participant_id.nunique()),
            "participants_both_days": int((f.groupby("participant_id").day.nunique() == 2).sum()),
            "templates": len(f),
            "templates_with_fiducials": int(f.fiducials_ok.sum()),
            "t_polarity_negative_share": round(float((f.T_polarity == -1).mean()), 3),
            "failed_files": len(pd.read_csv(dest / "failures.csv")),
        }
    except Exception as err:  # noqa: BLE001 - the summary must not hide the status
        summary["summary_error"] = repr(err)
    print(f"ecg_tanaskovic: {summary}", flush=True)

    # ``evaluate: true``: random-forest identification on these features
    # (scripts/ecg_tanaskovic_eval.py). Its JSON holds aggregate metrics only,
    # so it goes to out_dir/runs, which the workflows upload.
    if job.get("evaluate") and code == 0:
        runs = out_dir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        cmd = [sys.executable, str(SCRIPTS / "ecg_tanaskovic_eval.py"),
               "--features", str(dest / "features.csv"),
               "--out", str(runs / f"{stamp}_ecg_tanaskovic_rf.json"),
               "--seeds", str(job.get("seeds", 10)), "--draws", str(job.get("draws", 50))]
        logs = out_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "ecg_tanaskovic_eval.txt").open("w") as log:
            try:
                ev = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=_remaining(deadline)).returncode
            except subprocess.TimeoutExpired:
                ev = "timeout"
        summary["evaluation"] = "ok" if ev == 0 else f"exit {ev}"
        print(f"ecg_tanaskovic evaluation: {summary['evaluation']}", flush=True)
    return summary


def process(data_dir: Path, out_dir: Path) -> None:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = select_part(yaml.safe_load(JOB.read_text()))
    print(f"part: {job.get('part', '-')}, groups: {job.get('groups')}", flush=True)
    start = time.time()
    deadline = start + 60 * float(job.get("deadline_min", 52))
    report = {"job": job, "hardware": _hardware(), "timings_s": {}}
    print(json.dumps(report["hardware"]), flush=True)

    if job.get("job") in ("ecg_rest", "ecg_tanaskovic"):
        run = _ecg_rest if job["job"] == "ecg_rest" else _ecg_tanaskovic
        report[job["job"]] = run(data_dir, out_dir, job, deadline)
        report["timings_s"]["total"] = round(time.time() - start, 1)
        (out_dir / "job.json").write_text(json.dumps(report, indent=1, default=str))
        return

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
