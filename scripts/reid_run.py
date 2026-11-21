"""Run models through the common evaluation (same folds for the whole team).

python scripts/reid_run.py --make-folds                  # once, then commit
python scripts/reid_run.py --model b1_rel_ec+eo --scenario d1_to_d2
python scripts/reid_run.py --model all --scenario all --task rest
python scripts/reid_run.py --list
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.baselines import MODELS as BASELINES
from reid.config import FEATURES
from reid.eval.protocol import (
    FOLDS_PATH,
    SCENARIOS,
    FeatureStore,
    load_folds,
    make_folds,
    save_folds,
)
from reid.eval.runner import evaluate
from reid.final import MODELS as FINAL
from reid.final.raw import RAW
from reid.inventory import load_inventory

MODELS = BASELINES | FINAL


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", default=[], help="name or 'all'")
    parser.add_argument("--scenario", action="append", default=[], help="name or 'all'")
    parser.add_argument("--task", default="rest", choices=["rest", "artefact", "video"])
    parser.add_argument("--probe-task", default=None, help="cross-task transfer")
    parser.add_argument("--data", type=Path, default=None, help="dataset root")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--raw", type=Path, default=RAW, help="raw cache (f_ models)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--make-folds", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="overwrite folds.json")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(MODELS))
        return
    inv = load_inventory(args.data)
    if args.make_folds:
        save_folds(make_folds(inv, args.n_folds), force=args.force)
        print(f"folds written -> {FOLDS_PATH}")
        return
    if not FOLDS_PATH.exists():
        sys.exit("results/splits/folds.json missing: run with --make-folds first.")
    folds = load_folds()
    stores = {"features": FeatureStore(args.features), "raw": FeatureStore(args.raw)}
    models = list(MODELS) if "all" in args.model else args.model
    scenarios = list(SCENARIOS) if "all" in args.scenario else args.scenario
    for name in models:
        for scenario in scenarios or ["d1_to_d2"]:
            try:
                run = evaluate(
                    MODELS[name],
                    inv,
                    folds,
                    scenario,
                    args.task,
                    args.probe_task,
                    stores["raw" if name in FINAL else "features"],
                    n_boot=args.n_boot,
                )
            except ValueError as err:
                print(f"{name} / {scenario}: skipped ({err})")
                continue
            top1 = run["pooled"]["top1"]
            print(
                f"{name:28s} {scenario:18s} top-1 {top1['value']:.3f} "
                f"[{top1['ci'][0]:.3f}, {top1['ci'][1]:.3f}] "
                f"chance {run['pooled']['chance_top1']['value']:.3f}"
            )


if __name__ == "__main__":
    main()
