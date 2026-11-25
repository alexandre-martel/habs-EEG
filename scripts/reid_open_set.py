"""Open set: can we tell that a day-2 person was *not* recorded on day 1?

    python scripts/reid_open_set.py            # simulation on the final model
    python scripts/reid_open_set.py --real     # + real absent people (rest LDA)

**Simulation** (final 11-member model, score dumps of the GPU jobs): for each
probe, its own day-1 session is removed from the gallery (its score set to NaN
before the per-probe z-normalisation, so the normalisation is recomputed
without it, as it would be for a real absent person); the probe is then an
"absent" probe. Each person thus gives one *present* and one *absent* probe.
The decision statistic is the best fused score of the probe (z units), or the
margin between the two best. The threshold is fixed on the inner validation
(no test label used) and read on test:

- DIR: present, ranked first and above the threshold (identified);
- FPIR: absent and above the threshold (wrongly matched to someone).

**Real absent people** (``--real``): the people recorded on day 2 only are the
distractors of their fold; their day-2 rest is probed against the fold's day-1
gallery with the two rest LDA members, fitted on the fold's training people.
Comparing their best score with the simulated absent probes of the same model
checks that the simulation is representative. Only aggregates are written
(no participant id).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.config import RESULTS, SEED
from reid.final.fusion import znorm
from scripts.reid_offline_fusion import load, variants

GPU = RESULTS / "gpu"
FINAL_JOBS = (  # rest s0-4, rest s5-9, video encoders, video LDA + seeds 5-6
    "20261124T134720-d511ae",
    "20261124T143623-24b2b5",
    "20261124T223621-74a207",
    "20261124T234226-eb178c",
)
FINAL = "all5+vid3+lvid3 ens10"
N_BOOT = 2000


def fused(fold: dict, slots, absent: bool) -> np.ndarray:
    """Fused score matrix [probe, gallery]; with ``absent``, each probe's own
    gallery column is removed: left out of every normalisation (member and
    slot level, as if it had never been in the gallery), -inf at the end."""
    t = fold["truth"]
    rows = np.arange(len(t))

    def drop(Z):
        if absent:
            Z = np.array(Z, float)
            Z[rows, t] = np.nan
        return Z

    def slot(s):
        return drop(znorm(np.mean([drop(znorm(drop(fold[m]))) for m in s], 0)))

    F = np.mean([slot(s) for s in slots], 0)
    if absent:
        F[rows, t] = -np.inf
    return F


def stats(F: np.ndarray, truth: np.ndarray) -> dict:
    """Per probe: best score, margin to the second, whether rank 1 is right."""
    srt = np.sort(F, axis=1)
    return {
        "top": srt[:, -1],
        "margin": srt[:, -1] - srt[:, -2],
        "correct": F.argmax(1) == truth,
    }


def collect(folds: dict, slots) -> dict:
    """Present and absent statistics pooled over folds, with a person index
    (present and absent probes of a person are paired for the bootstrap)."""
    pres, abs_ = [], []
    for fold in folds.values():
        for absent, acc in ((False, pres), (True, abs_)):
            F = fused(fold, slots, absent)
            st = stats(F, fold["truth"])
            # judges agreeing with the fused rank 1 (a vote, 0..len(slots))
            st["agree"] = np.sum(
                [fused(fold, [s], absent).argmax(1) == F.argmax(1) for s in slots], 0
            ).astype(float)
            acc.append(st)

    def cat(parts):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    return {"present": cat(pres), "absent": cat(abs_)}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a present probe > score of an absent probe), ties count 1/2."""
    d = pos[:, None] - neg[None]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def operating(d: dict, key: str, thr: float) -> dict:
    p, a = d["present"], d["absent"]
    return {
        "dir": float((p["correct"] & (p[key] >= thr)).mean()),
        "present_rejected": float((p[key] < thr).mean()),
        "fpir": float((a[key] >= thr).mean()),
    }


def boot(d: dict, fn, rng) -> list[float]:
    """Person-level bootstrap (present and absent probe of a person together)."""
    n = len(d["present"]["top"])
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        sub = {s: {k: v[i] for k, v in d[s].items()} for s in d}
        vals.append(fn(sub))
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])]


def thresholds(val: dict, key: str) -> dict:
    """Thresholds fixed on validation: absent accepted at most 5 % / 1 %, and
    the equal-error point (present rejected = absent accepted)."""
    a = np.sort(val["absent"][key])
    p = val["present"][key]
    grid = np.unique(np.concatenate([a, p]))
    eer = grid[np.argmin([abs((p < g).mean() - (a >= g).mean()) for g in grid])]
    return {
        "fpir5": float(np.quantile(a, 0.95)),
        "fpir1": float(np.quantile(a, 0.99)),
        "eer": float(eer),
    }


def analyse(data: dict, slots, rng) -> dict:
    val, test = collect(data["val"], slots), collect(data["test"], slots)
    out = {"n_val": len(val["present"]["top"]), "n_test": len(test["present"]["top"])}
    for key in ("top", "margin", "agree"):
        pos, neg = test["present"][key], test["absent"][key]
        r = {
            "auc_val": round(auc(val["present"][key], val["absent"][key]), 4),
            "auc_test": round(auc(pos, neg), 4),
            "auc_test_ci": boot(
                test, lambda s, key=key: auc(s["present"][key], s["absent"][key]), rng
            ),
        }
        for name, thr in thresholds(val, key).items():
            op = operating(test, key, thr)
            r[name] = {
                "threshold": round(thr, 3),
                **{k: round(v, 4) for k, v in op.items()},
                **{
                    f"{k}_ci": boot(
                        test,
                        lambda s, k=k, key=key, thr=thr: operating(s, key, thr)[k],
                        rng,
                    )
                    for k in ("dir", "fpir")
                },
            }
        out[key] = r
    out["closed_set_top1"] = round(float(test["present"]["correct"].mean()), 4)
    return out, val, test


def per_slot(data: dict, slots: list, names: list) -> list[dict]:
    """Each judge alone: closed-set top-1 and absent-detection AUC (test)."""
    rows = []
    for name, slot in zip(names, slots):
        t = collect(data["test"], [slot])
        v = collect(data["val"], [slot])
        rows.append(
            {
                "judge": name,
                "top1_test": round(float(t["present"]["correct"].mean()), 4),
                "auc_top_val": round(auc(v["present"]["top"], v["absent"]["top"]), 4),
                "auc_top_test": round(auc(t["present"]["top"], t["absent"]["top"]), 4),
            }
        )
    return rows


SLOT_NAMES = [
    "LDA EEG (repos)",
    "LDA battement (repos)",
    "réseau EEG (repos)",
    "réseau battement (repos)",
    "réseau respiration (repos)",
    "réseau EEG (vidéo)",
    "réseau battement (vidéo)",
    "réseau respiration (vidéo)",
    "LDA EEG (vidéo)",
    "LDA battement (vidéo)",
    "LDA respiration (vidéo)",
]


def final_slots() -> tuple[dict, list]:
    data = load([GPU / j / "scores" for j in FINAL_JOBS])
    members = set.intersection(*(set(f) - {"truth"} for f in data["test"].values()))
    seeds = sorted({int(m.rsplit("_s", 1)[1]) for m in members if "enc_eeg_s" in m})
    return data, variants(members, seeds)[FINAL]


# -- real absent people (day 2 only), rest LDA -----------------------------------
def real_absent(rng) -> dict:
    """Rest LDA (EEG, beat) per fold: test probes (present), the same probes
    with their own gallery session removed (simulated absent), and the
    day-2-only distractors (real absent)."""
    from reid.baselines.b3_lda_plda import LdaModel
    from reid.eval.protocol import FeatureStore, SessionSet, build_split, load_folds
    from reid.inventory import load_inventory

    inv, store = load_inventory(), FeatureStore()
    out = {"present": [], "simulated": [], "real": []}
    for fold in load_folds():
        split = build_split(inv, fold, "d1_to_d2", store=store)
        gal, prb = split.gallery, split.probe
        df = inv[inv.participant_id.isin(fold["distractors"]) & inv.has_rest]
        days = df.groupby("participant_id").day.apply(set)
        only2 = sorted(days.index[days.apply(lambda d: d == {2})])
        rows = inv[inv.participant_id.isin(only2) & (inv.day == 2) & inv.has_rest]
        rows = rows.assign(task="rest", part=None)
        rows = rows[[store.available(p, 2, "rest") for p in rows.participant_id]]
        extra = SessionSet(rows.reset_index(drop=True), store)
        Sp, Sx = [], []
        for mods in (("eeg",), ("ecg_beat",)):
            model = LdaModel(mods)
            model.fit(split.train)
            Sp.append(model.score(gal, prb))
            Sx.append(model.score(gal, extra))
        truth = np.array([list(gal.pids).index(p) for p in prb.pids])
        fold_d = {"truth": truth, "eeg": Sp[0], "beat": Sp[1]}
        slots = [["eeg"], ["beat"]]
        out["present"].append(stats(fused(fold_d, slots, False), truth))
        out["simulated"].append(stats(fused(fold_d, slots, True), truth))
        Fx = np.mean([znorm(znorm(S)) for S in Sx], 0)
        out["real"].append(stats(Fx, np.full(len(Fx), -1)))
        print(f"fold: {len(prb.pids)} present, {len(Fx)} real absent", flush=True)
    d = {
        k: {s: np.concatenate([p[s] for p in v]) for s in v[0]} for k, v in out.items()
    }
    res = {
        "n_present": len(d["present"]["top"]),
        "n_real_absent": len(d["real"]["top"]),
    }
    for key in ("top", "margin"):
        res[key] = {
            "auc_simulated": round(auc(d["present"][key], d["simulated"][key]), 4),
            "auc_real": round(auc(d["present"][key], d["real"][key]), 4),
            "simulated_mean_sd": [
                round(float(d["simulated"][key].mean()), 3),
                round(float(d["simulated"][key].std()), 3),
            ],
            "real_mean_sd": [
                round(float(d["real"][key].mean()), 3),
                round(float(d["real"][key].std()), 3),
            ],
            "present_mean_sd": [
                round(float(d["present"][key].mean()), 3),
                round(float(d["present"][key].std()), 3),
            ],
        }
    res["closed_set_top1"] = round(float(d["present"]["correct"].mean()), 4)
    return res, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS / "open_set.json")
    ap.add_argument("--npz", type=Path, help="save the score distributions here")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    data, slots = final_slots()
    res, _val, test = analyse(data, slots, rng)
    res["per_judge"] = per_slot(data, slots, SLOT_NAMES)
    dists = {f"{s}_{k}": test[s][k] for s in ("present", "absent") for k in test[s]}
    if args.real:
        res["real_absent_rest_lda"], d = real_absent(rng)
        dists |= {f"lda_{s}_{k}": d[s][k] for s in d for k in d[s]}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if args.npz:
        np.savez(args.npz, **dists)


if __name__ == "__main__":
    main()
