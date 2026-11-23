"""Pretrain the EEG encoder of a model for some folds, ahead of its evaluation.

python scripts/reid_pretrain.py --model f_eeg_pt_maelong_s0 --fold 0 --fold 1

Computes exactly the pretraining that ``scripts/reid_run.py --model <model>``
would compute inside ``fit`` (same fold split, same inner people, same
options) and leaves it in the disk memo (``CACHE/pretrain``). Run in parallel
processes at the start of a GPU job, the pretrainings of the 5 folds then run
at once instead of one after the other inside the evaluation, which can run
alongside: it finds them in the memo, or waits on the lock of the one being
computed. Prints nothing but the pretraining summary: no score, no test person.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.eval.protocol import FeatureStore, build_split, load_folds
from reid.final import MODELS
from reid.final.model import inner_split
from reid.final.pretrain import pretrained
from reid.final.raw import RAW
from reid.inventory import load_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, action="append", required=True)
    parser.add_argument("--scenario", default="d1_to_d2")
    parser.add_argument("--task", default="rest")
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()

    model = MODELS[args.model]()
    cfg = getattr(model, "pretrain", None) or getattr(model, "eeg_pretrain", None)
    if not cfg:
        sys.exit(f"{args.model} has no pretraining")
    inv, folds = load_inventory(args.data), load_folds()
    for k in args.fold:
        split = build_split(
            inv, folds[k], args.scenario, store=FeatureStore(RAW), task=args.task
        )
        if not len(split.probe):
            continue
        inner, _, _ = inner_split(split.train, model.split_seed)
        pids = sorted(set(inner.sessions.participant_id))
        start = time.time()
        _, info = pretrained(pids, cfg, model.device)
        print(
            f"fold {k}: {info.get('steps')} steps, {info.get('people')} people, "
            f"{time.time() - start:.0f} s{' (memo)' if info.get('memo') else ''}",
            flush=True,
        )


if __name__ == "__main__":
    main()
