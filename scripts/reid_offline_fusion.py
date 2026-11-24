"""Offline score fusion from the member score matrices saved by GPU jobs
(``REID_SCORES``, see ``reid.final.fusion``).

    python scripts/reid_offline_fusion.py cache/gpu/dump_a/*/scores cache/gpu/dump_b/*/scores

Every variant is a list of *slots*, fused with equal weights (or weights tuned
on the inner validation); a slot is a list of members whose z-normalised
scores are averaged first (a seed ensemble: ``enc_eeg_s0..s4`` in one slot).
Each variant is scored on the inner validation (the choice criterion: top-1
and MRR pooled over the 5 folds' validation probes) and on the test folds
(top-1 with its person-level bootstrap CI, chance level). Variants are
ranked on validation; the test column is read for the chosen one only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.eval import metrics
from reid.eval.matching import rank_counts, reciprocal_rank, topk
from reid.final.fusion import simplex, znorm

NAME = re.compile(r"^(?P<fusion>.+)_(?P<fold>[0-9a-f]{10})_(?P<kind>val|test)\.npz$")


def load(folders: list[Path]) -> dict:
    """{kind: {fold: {"truth": array, member: matrix}}}, members merged across
    files (the truth of a fold must agree between files)."""
    out: dict = {"val": defaultdict(dict), "test": defaultdict(dict)}
    for folder in folders:
        for f in sorted(Path(folder).glob("*.npz")):
            m = NAME.match(f.name)
            if not m:
                continue
            data = np.load(f)
            slot = out[m["kind"]][m["fold"]]
            if "truth" in slot:
                assert np.array_equal(slot["truth"], data["truth"]), f
            slot["truth"] = data["truth"]
            for key in data.files:
                if key.startswith("S:"):
                    slot[key[2:]] = data[key].astype(float)
    return out


def rank_norm(S: np.ndarray) -> np.ndarray:
    """Per-probe ranks scaled to [-0.5, 0.5] (insensitive to score shape)."""
    r = np.argsort(np.argsort(np.nan_to_num(S, nan=-np.inf), axis=1), axis=1)
    return r / max(S.shape[1] - 1, 1) - 0.5


def dual_znorm(S: np.ndarray) -> np.ndarray:
    """Mean of per-probe and per-gallery z-scores. Transductive: the gallery
    side uses the other probes' scores (no label), reported apart."""
    return (znorm(S) + znorm(S.T).T) / 2


NORMS = {"z": znorm, "rank": rank_norm, "dual": dual_znorm}


def parts(folds: dict, slots, norm: str) -> list[tuple[list, np.ndarray]]:
    """Per fold: the slots' normalised scores (computed once per variant)."""
    f = NORMS[norm]
    return [
        (
            [znorm(np.mean([f(fold[m]) for m in slot], axis=0)) for slot in slots],
            fold["truth"],
        )
        for fold in folds.values()
    ]


def val_metrics(P: list, w) -> tuple[float, float]:
    t1, rr = [], []
    for Z, truth in P:
        g, ties = rank_counts(sum(wi * z for wi, z in zip(w, Z)), truth)
        t1.append(topk(g, ties, 1))
        rr.append(reciprocal_rank(g, ties))
    return float(np.concatenate(t1).mean()), float(np.concatenate(rr).mean())


def tuned(P: list, step=0.1) -> np.ndarray:
    """Weights maximising pooled validation top-1, then MRR, then closeness to
    equal (as ScoreFusion, but pooled over the folds' validation probes)."""
    m = len(P[0][0])
    best, key = None, None
    for w in simplex(m, step):
        k = (*val_metrics(P, w), -np.abs(w - 1 / m).sum())
        if key is None or k > key:
            best, key = w, k
    return best


def test_summary(P: list, w) -> dict:
    blocks = [
        metrics.Block(sum(wi * z for wi, z in zip(w, Z)), truth, True) for Z, truth in P
    ]
    s = metrics.summarize(blocks)
    return {
        k: [round(s[k]["value"], 4), *[round(x, 4) for x in s[k].get("ci", [])]]
        for k in ("top1", "mrr", "chance_top1", "hungarian")
    }


def variants(members: set[str], seeds: list[int]) -> dict:
    """Named slot lists for the members present."""

    def enc(name, ks):
        return [f"{name}_s{k}" for k in ks if f"{name}_s{k}" in members]

    lda2 = [["lda_eeg"], ["lda_ecg_beat"]]
    out = {}
    for k in seeds:
        out[f"all5 s{k}"] = [
            *lda2,
            enc("enc_eeg", [k]),
            enc("enc_ecg_beat", [k]),
            enc("enc_resp", [k]),
        ]
    for n in (3, 5, 10):
        ks = seeds[:n]
        if len(ks) < n:
            continue
        e = {x: enc(x, ks) for x in ("enc_eeg", "enc_ecg_beat", "enc_resp")}
        out[f"all5 ens{n}"] = [*lda2, e["enc_eeg"], e["enc_ecg_beat"], e["enc_resp"]]
        out[f"all6 ens{n}"] = [*out[f"all5 ens{n}"], ["lda_eeg+ecg_beat+resp"]]
        out[f"enc3 ens{n}"] = [e["enc_eeg"], e["enc_ecg_beat"], e["enc_resp"]]
        pt = enc("enc_eeg_pt", ks)
        if pt:
            out[f"all5 ens{n} eeg+pt"] = [
                *lda2,
                e["enc_eeg"] + pt,
                e["enc_ecg_beat"],
                e["enc_resp"],
            ]
            out[f"all6 ens{n} +pt slot"] = [*out[f"all6 ens{n}"], pt]
        # the same encoders on the video of the same sessions (TaskView)
        v = {x: enc(f"{x}_vid", ks) for x in ("enc_eeg", "enc_ecg_beat", "enc_resp")}
        if all(v.values()):
            vid = [v["enc_eeg"], v["enc_ecg_beat"], v["enc_resp"]]
            out[f"vid3 ens{n}"] = vid
            out[f"all5+vid3 ens{n}"] = [*out[f"all5 ens{n}"], *vid]
            out[f"all5+vid eeg ens{n}"] = [*out[f"all5 ens{n}"], v["enc_eeg"]]
            # rest and video of one signal averaged in one slot
            out[f"all5 x vid ens{n}"] = [
                *lda2,
                *[e[x] + v[x] for x in ("enc_eeg", "enc_ecg_beat", "enc_resp")],
            ]
    return {k: v for k, v in out.items() if all(v)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--norms", default="z,rank,dual")
    ap.add_argument("--tuned", action="store_true", help="also tuned weights")
    ap.add_argument("--json", type=Path, help="write the table here")
    args = ap.parse_args()
    data = load(args.folders)
    val, test = data["val"], data["test"]
    assert set(val) == set(test) and len(val) == 5, (len(val), len(test))
    members = set.intersection(*(set(f) - {"truth"} for f in test.values()))
    seeds = sorted({int(m.rsplit("_s", 1)[1]) for m in members if "enc_eeg_s" in m})
    print(f"{len(val)} folds, members: {sorted(members)}")
    rows = []
    for name, slots in variants(members, seeds).items():
        for norm in args.norms.split(","):
            pv, pt = parts(val, slots, norm), parts(test, slots, norm)
            ws = {"eq": np.full(len(slots), 1 / len(slots))}
            if args.tuned and len(slots) <= 6:
                ws["tuned"] = tuned(pv)
            for mode, w in ws.items():
                v1, vrr = val_metrics(pv, w)
                rows.append(
                    {
                        "variant": name,
                        "norm": norm,
                        "weights": mode,
                        "w": np.round(w, 2).tolist(),
                        "val_top1": round(v1, 4),
                        "val_mrr": round(vrr, 4),
                        "test": test_summary(pt, w),
                    }
                )
    rows.sort(key=lambda r: (r["val_top1"], r["val_mrr"]), reverse=True)
    for r in rows:
        t = r["test"]["top1"]
        print(
            f"{r['variant']:24s} {r['norm']:5s} {r['weights']:5s} "
            f"val {r['val_top1']:.3f} mrr {r['val_mrr']:.3f} | "
            f"test {t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}] "
            f"hung {r['test']['hungarian'][0]:.3f}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
