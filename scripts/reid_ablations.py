"""Ablations of one method (step 6): modality, task, duration, emotion.

python scripts/reid_ablations.py --method b3_lda --modalities all --what all
python scripts/reid_ablations.py --method b1 --what duration
Then: python scripts/reid_report.py  (figures results/figures/ablation_*.png)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.baselines.b3_lda_plda import ALL
from reid.baselines.b4_ablations import METHODS, run
from reid.config import FEATURES, default_data_dir
from reid.eval.protocol import FeatureStore, load_folds
from reid.inventory import load_inventory

KINDS = ("modality", "task", "duration", "arousal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--what", nargs="+", default=["all"], choices=[*KINDS, "all"])
    parser.add_argument(
        "--modalities", default="eeg", help="base modalities, comma-separated, or 'all'"
    )
    parser.add_argument("--eeg-kind", default="rel", choices=["abs", "rel", "bands"])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    data_dir = args.data or default_data_dir()
    mods = ALL if args.modalities == "all" else tuple(args.modalities.split(","))
    base = {"modalities": mods, "eeg_kind": args.eeg_kind}
    inv, folds = load_inventory(data_dir), load_folds()
    for kind in KINDS if "all" in args.what else args.what:
        for r in run(
            kind,
            args.method,
            inv,
            folds,
            FeatureStore(args.features),
            data_dir,
            args.n_boot,
            base,
        ):
            top1 = r["pooled"]["top1"]
            print(
                f"{kind:9s} {r['label']:18s} top-1 {top1['value']:.3f} "
                f"[{top1['ci'][0]:.3f}, {top1['ci'][1]:.3f}]"
            )


if __name__ == "__main__":
    main()
