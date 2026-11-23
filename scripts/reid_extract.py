"""Extract per-session features into cache/features/<task>/<PID>_day-<d>.npz.

python scripts/reid_extract.py --data ./data --task rest
python scripts/reid_extract.py --data ./data --task artefact --shard 0/4
python scripts/reid_extract.py --raw --task rest     # raw-signal cache (final model)
python scripts/reid_extract.py --video               # video-task EEG (pretraining)

Resumable: existing files are skipped; failures are logged to
cache/features/failures.csv and never stop the run. Keys: see reid/features
(``--raw``: reid/final/raw.py, written to cache/raw/).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.config import FEATURES, MODALITIES, default_data_dir
from reid.features import run_extraction
from reid.final.raw import RAW, run_raw_extraction
from reid.final.video import VIDEO, run_video_extraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=None, help="dataset root")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--raw", action="store_true", help="raw-signal cache")
    parser.add_argument(
        "--video", action="store_true", help="video-task EEG cache (pretraining)"
    )
    parser.add_argument(
        "--task", action="append", default=[], choices=["rest", "artefact", "video"]
    )
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th session from i")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    i, n = (int(v) for v in args.shard.split("/"))
    data = args.data or default_data_dir()
    if args.video:
        counts = run_video_extraction(
            data, args.out or VIDEO, (i, n), args.overwrite, args.limit
        )
        print(f"video: {counts}")
        return
    for task in args.task or ["rest"]:
        if args.raw:
            counts = run_raw_extraction(
                data, args.out or RAW, task, (i, n), args.overwrite, args.limit
            )
            print(f"{task}: {counts}")
            continue
        counts = run_extraction(
            data,
            args.out or FEATURES,
            task,
            args.modalities.split(","),
            (i, n),
            args.overwrite,
            args.limit,
        )
        print(f"{task}: {counts}")


if __name__ == "__main__":
    main()
