"""Identification with the ECG features of Tanasković et al. (2025), on HABS rest.

    python scripts/ecg_tanaskovic_eval.py --features features.csv --out run.json [--seeds 10]

Input: OUT_DIR/features.csv of scripts/ecg_tanaskovic_extract.py (one row per
10-beat template, 16 features, rest blocks 10002 = eyes open 30 s and
10004 = eyes closed 60 s). Output: one JSON of aggregate metrics only (no
participant identifier, no per-participant result), safe for the public repo.

Model as in the paper: scikit-learn RandomForestClassifier with default
hyperparameters ("hyperparameter tuning is intentionally omitted"), one class
per participant, features as extracted (Bazett-corrected intervals, no other
normalisation). Templates without all fiducial points are dropped.

Protocols
---------
A, one day (the paper's conditions: one recording per person), run on day 1
and on day 2 separately:
  A1 random stratified split, 33 % of the templates in test (the paper's
     split). Neighbouring templates of the same block fall on both sides:
     optimistic, kept because it is what the paper reports.
  A2 temporal split inside each block: the last third of the templates in
     test, the template just before them dropped (gap), the rest in train.
  A3 train on one block, test on the other (eyes closed -> eyes open and the
     reverse), the closest analogue of the paper's baseline -> anger test.
B, two days (not in the paper): train on every template of day 1, test on
  day 2, and the reverse, for the participants with rest on both days.
  Reported per template and per session (class probabilities averaged over
  the probe session's templates, then ranked: top-1, top-5, MRR, 95 % CI by
  bootstrap over participants). B42 repeats B on random galleries of 42
  participants, the gallery size of results/leaderboard.csv (5 folds), so
  that the numbers compare with b1_ecg_beat and the demographic control.

Feature sets: all16 (the 16 ECG features of Table 1), paper8 (the 8-feature
ECG consensus subset reported in the paper, chosen on their data), and
all16_rel (amplitudes and slopes divided by RQ_amp, RQ_amp removed: a guard
against electrode gain changing between days; our variant, not the paper's).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reid.features.ecg_tanaskovic import NAMES  # noqa: E402

EO, EC = 10002, 10004
PAPER8 = ("ECGQRScrest", "QRS_int", "QR_slope", "RQ_amp", "RS_amp", "RS_slope",
          "RT_amp", "TT2_amp")
REL = ("RS_amp", "RT_amp", "TT1_amp", "TT2_amp",
       "QR_slope", "RS_slope", "TT1_slope", "TT2_slope")
GALLERY = 42


def load(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    if "participant_id" not in f:  # local check files carry only the file name
        f["participant_id"] = f.file.str.extract(r"(P\d+)_")[0]
        f["day"] = f.file.str.extract(r"_day-(\d+)_")[0].astype(int)
    f = f[f.fiducials_ok.astype(bool)].copy()
    f = f[np.isfinite(f[list(NAMES)]).all(axis=1)]
    f["segment"] = f.segment.astype(int)
    f["pid"] = pd.factorize(f.participant_id)[0]  # identifiers never leave this script
    for c in REL:
        f[c + "_rel"] = f[c] / f.RQ_amp.where(f.RQ_amp.abs() > 1e-9)
    return f[np.isfinite(f[[c + "_rel" for c in REL]]).all(axis=1)]


FEATURES = {
    "all16": list(NAMES),
    "paper8": list(PAPER8),
    "all16_rel": [c for c in NAMES if c != "RQ_amp" and c not in REL]
                 + [c + "_rel" for c in REL],
}


def rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, n_jobs=-1)


def template_metrics(y, pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision_macro": p,
            "recall_macro": r, "f1_macro": f1}


def fit_score(train, test, cols, seed) -> dict:
    m = rf(seed).fit(train[cols].to_numpy(), train.pid.to_numpy())
    out = template_metrics(test.pid.to_numpy(), m.predict(test[cols].to_numpy()))
    n = train.pid.nunique()
    return out | {"n_classes": n, "chance": 1 / n, "n_train": len(train), "n_test": len(test),
                  "importance": dict(zip(cols, m.feature_importances_.round(4).tolist()))}


def mean_sd(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out |= {k + "_sd": float(np.std([r[k] for r in runs])) for k in
            ("accuracy", "f1_macro") if k in out}
    imp = pd.DataFrame([r["importance"] for r in runs]).mean().sort_values(ascending=False)
    return out | {"repeats": len(runs), "importance": imp.round(4).to_dict()}


def keep_min(f: pd.DataFrame, by: list[str], n: int) -> pd.DataFrame:
    return f[f.groupby(by).pid.transform("size") >= n]


# ---- A: one day --------------------------------------------------------------

def a1(day: pd.DataFrame, cols, seeds) -> dict:
    d = keep_min(day, ["pid"], 3)
    runs = []
    for s in seeds:
        tr, te = train_test_split(d, test_size=0.33, stratify=d.pid, random_state=s)
        runs.append(fit_score(tr, te, cols, s))
    return mean_sd(runs)


def a2(day: pd.DataFrame, cols, seeds) -> dict:
    d = keep_min(day, ["pid", "segment"], 3).sort_values(["pid", "segment", "template"])
    k = d.groupby(["pid", "segment"]).cumcount()
    n = d.groupby(["pid", "segment"]).pid.transform("size")
    n_test = np.ceil(n / 3).astype(int)
    test = k >= n - n_test
    train = k < n - n_test - 1  # one template dropped between train and test
    return mean_sd([fit_score(d[train], d[test], cols, s) for s in seeds])


def a3(day: pd.DataFrame, cols, seeds) -> dict:
    both = day.groupby("pid").segment.nunique() == 2
    d = day[day.pid.isin(both[both].index)]
    out = {}
    for name, a, b in (("ec_to_eo", EC, EO), ("eo_to_ec", EO, EC)):
        tr, te = d[d.segment == a], d[d.segment == b]
        out[name] = mean_sd([fit_score(tr, te, cols, s) for s in seeds])
    return out


# ---- B: two days -------------------------------------------------------------

def session_ranks(model, test: pd.DataFrame, cols) -> np.ndarray:
    """Rank (1 = first) of the true class per probe session, from the class
    probabilities averaged over the session's templates."""
    proba = pd.DataFrame(model.predict_proba(test[cols].to_numpy()), index=test.index)
    mean = proba.groupby(test.pid.to_numpy()).mean()
    classes = model.classes_
    ranks = []
    for pid, row in mean.iterrows():
        score = row.to_numpy()
        true = score[np.searchsorted(classes, pid)]
        # ties counted at their expected rank, as in reid.eval
        ranks.append(1 + (score > true).sum() + 0.5 * ((score == true).sum() - 1))
    return np.asarray(ranks)


def rank_metrics(ranks: np.ndarray, rng, n_boot=1000) -> dict:
    top1 = (ranks <= 1).astype(float)
    boot = [top1[rng.integers(0, len(top1), len(top1))].mean() for _ in range(n_boot)]
    return {"session_top1": float(top1.mean()),
            "session_top1_lo": float(np.percentile(boot, 2.5)),
            "session_top1_hi": float(np.percentile(boot, 97.5)),
            "session_top5": float((ranks <= 5).mean()),
            "session_mrr": float((1 / ranks).mean()), "n_probes": len(ranks)}


def b(f: pd.DataFrame, cols, seeds) -> dict:
    both = f.groupby("pid").day.nunique() == 2
    d = f[f.pid.isin(both[both].index)]
    out = {}
    for name, ga, pr in (("d1_to_d2", 1, 2), ("d2_to_d1", 2, 1)):
        tr, te = d[d.day == ga], d[d.day == pr]
        runs, sess = [], []
        for s in seeds:
            m = rf(s).fit(tr[cols].to_numpy(), tr.pid.to_numpy())
            r = template_metrics(te.pid.to_numpy(), m.predict(te[cols].to_numpy()))
            n = tr.pid.nunique()
            runs.append(r | {"n_classes": n, "chance": 1 / n, "n_train": len(tr),
                             "n_test": len(te),
                             "importance": dict(zip(cols, m.feature_importances_.round(4)))})
            sess.append(rank_metrics(session_ranks(m, te, cols), np.random.default_rng(s)))
        out[name] = mean_sd(runs) | {k: float(np.mean([x[k] for x in sess])) for k in sess[0]}
    return out


def b42(f: pd.DataFrame, cols, n_draws: int) -> dict:
    both = f.groupby("pid").day.nunique() == 2
    pids = np.asarray(both[both].index)
    d = f[f.pid.isin(pids)]
    rng = np.random.default_rng(0)
    top1, top5, mrr = [], [], []
    for k in range(n_draws):
        g = rng.choice(pids, size=min(GALLERY, len(pids)), replace=False)
        tr, te = d[(d.day == 1) & d.pid.isin(g)], d[(d.day == 2) & d.pid.isin(g)]
        ranks = session_ranks(rf(k).fit(tr[cols].to_numpy(), tr.pid.to_numpy()), te, cols)
        top1.append((ranks <= 1).mean())
        top5.append((ranks <= 5).mean())
        mrr.append((1 / ranks).mean())
    return {"gallery": min(GALLERY, len(pids)), "draws": n_draws,
            "chance": 1 / min(GALLERY, len(pids)),
            "session_top1": float(np.mean(top1)), "session_top1_sd": float(np.std(top1)),
            "session_top1_p2.5": float(np.percentile(top1, 2.5)),
            "session_top1_p97.5": float(np.percentile(top1, 97.5)),
            "session_top5": float(np.mean(top5)), "session_mrr": float(np.mean(mrr))}


def counts(f: pd.DataFrame) -> dict:
    per = f.groupby(["pid", "day"]).size()
    return {
        "templates": len(f), "participants": int(f.pid.nunique()),
        "participants_day1": int(f[f.day == 1].pid.nunique()),
        "participants_day2": int(f[f.day == 2].pid.nunique()),
        "participants_both_days": int((f.groupby("pid").day.nunique() == 2).sum()),
        "templates_per_session_median": float(per.median()),
        "templates_per_session_min": int(per.min()),
        "templates_per_block_median": f.groupby("segment").apply(
            lambda g: float(g.groupby(["pid", "day"]).size().median())).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10, help="repeats of each protocol")
    ap.add_argument("--draws", type=int, default=50, help="random galleries of 42 in B42")
    ap.add_argument("--sets", nargs="+", default=list(FEATURES))
    a = ap.parse_args()
    start = time.time()
    f = load(a.features)
    seeds = list(range(a.seeds))
    try:
        git = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        git = ""
    report = {"model": "ecg_tanaskovic_rf", "date": datetime.now().isoformat(timespec="seconds"),
              "git": git, "config": {"seeds": a.seeds, "draws": a.draws, "test_size": 0.33,
                                     "classifier": "RandomForestClassifier(defaults)"},
              "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
              "counts": counts(f), "results": {}}
    for name in a.sets:
        cols = FEATURES[name]
        res = {}
        for day in sorted(f.day.unique()):
            one = f[f.day == day]
            if one.pid.nunique() < 2:
                continue
            res[f"A1_day{day}"] = a1(one, cols, seeds)
            res[f"A2_day{day}"] = a2(one, cols, seeds)
            res[f"A3_day{day}"] = a3(one, cols, seeds)
            print(f"{name} day {day}: A1 {res[f'A1_day{day}']['accuracy']:.3f} "
                  f"A2 {res[f'A2_day{day}']['accuracy']:.3f} "
                  f"A3 {res[f'A3_day{day}']['ec_to_eo']['accuracy']:.3f}", flush=True)
        if (f.groupby("pid").day.nunique() == 2).sum() >= 2:
            res["B"] = b(f, cols, seeds)
            res["B42"] = b42(f, cols, a.draws)
            print(f"{name} B d1->d2: template {res['B']['d1_to_d2']['accuracy']:.3f}, "
                  f"session top-1 {res['B']['d1_to_d2']['session_top1']:.3f}; "
                  f"B42 {res['B42']['session_top1']:.3f}", flush=True)
        report["results"][name] = res
    report["time_s"] = round(time.time() - start, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=float))
    print(f"written {a.out} in {report['time_s']} s", flush=True)


if __name__ == "__main__":
    main()
