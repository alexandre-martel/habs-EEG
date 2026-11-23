"""Structure of the HABS Hugging Face dataset and targeted downloads.

    python scripts/hf_habs.py tree
    python scripts/hf_habs.py fetch P001 --days 1 2 --tasks resting-state

``tree`` reads only the repository metadata (no token needed). ``fetch``
downloads the requested BDF/record_map pairs and the day metadata; the dataset
is gated, so it needs a token (``hf auth login`` or ``HF_TOKEN``) from an
account whose access request was accepted.
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "HABSteam/multimodal-eeg-emotion"
NAME = re.compile(
    r"^(?P<root>CURATED|PREPROCESSED)/(?P<pid>P\d+)/(?P=pid)_day-(?P<day>\d+)_"
    r"(?:task-(?P<task>[\w-]+)_multimodal|metadata)\.(?P<ext>\w+)$"
)


def tree() -> None:
    info = HfApi().dataset_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    gb = lambda b: f"{b / 1e9:.1f} GB"  # noqa: E731
    print(f"{REPO}  gated={info.gated}  {len(files)} files  {gb(sum(s for _, s in files))}")

    top = collections.defaultdict(lambda: [0, 0])
    for f, s in files:
        top[f.split("/")[0]][0] += 1
        top[f.split("/")[0]][1] += s
    for k, (n, s) in sorted(top.items()):
        print(f"  {k:24s} {n:5d} files  {gb(s)}")

    pids, bdfs, sizes = collections.defaultdict(set), collections.Counter(), collections.defaultdict(list)
    for f, s in files:
        m = NAME.match(f)
        if not m:
            continue
        pids[m["root"]].add(m["pid"])
        if m["ext"] == "bdf":
            bdfs[(m["root"], m["task"], m["day"])] += 1
            sizes[(m["root"], m["task"])].append(s)
    for root, p in pids.items():
        print(f"{root}: {len(p)} participants ({min(p)} .. {max(p)})")
    print("BDF files per (folder, task, day):")
    for k in sorted(bdfs):
        print(f"  {k[0]:13s} {k[1]:14s} day-{k[2]}  {bdfs[k]}")
    print("BDF size per task: median / total")
    for k, v in sorted(sizes.items()):
        print(f"  {k[0]:13s} {k[1]:14s} {statistics.median(v) / 1e6:5.0f} MB  {gb(sum(v))}")


def fetch(pid: str, days: list[int], tasks: list[str], root: str, out: Path) -> None:
    patterns = []
    for d in days:
        patterns.append(f"{root}/{pid}/{pid}_day-{d}_metadata.json")
        for t in tasks:
            patterns.append(f"{root}/{pid}/{pid}_day-{d}_task-{t}_multimodal.*")
    path = snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns, local_dir=out)
    for f in sorted(Path(path, root, pid).glob("*")):
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tree")
    f = sub.add_parser("fetch")
    f.add_argument("pid")
    f.add_argument("--days", type=int, nargs="+", default=[1])
    f.add_argument("--tasks", nargs="+", default=["resting-state"],
                   help="resting-state, emotion-video, artefact")
    f.add_argument("--root", default="CURATED", choices=["CURATED", "PREPROCESSED"])
    f.add_argument("--out", type=Path, default=Path("sample"))
    a = ap.parse_args()
    if a.cmd == "tree":
        tree()
    else:
        fetch(a.pid, a.days, a.tasks, a.root, a.out)


if __name__ == "__main__":
    main()
