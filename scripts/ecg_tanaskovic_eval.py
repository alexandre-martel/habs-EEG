"""Identification with the ECG features of Tanasković et al. (2025), on HABS rest.

    python scripts/ecg_tanaskovic_eval.py --features features.csv --out run.json [--seeds 10]

Input: OUT_DIR/features.csv of scripts/ecg_tanaskovic_extract.py (one row per
10-beat template, 16 features, rest blocks 10002 = eyes open without
blinking, 30 s, then 10004 = eyes closed, 60 s, ~5 s apart). Output: one
JSON of aggregate metrics only (no participant identifier, no
per-participant result), safe for the public repo.

Model as in the paper: scikit-learn RandomForestClassifier with default
hyperparameters, one class per participant, features as extracted
(Bazett-corrected intervals). Templates without all fiducial points are
dropped.

Every protocol reports, besides per-template metrics, per-session ranks:
for each tested session, class probabilities are averaged over its
templates, people are ranked from most to least probable, and rank 1 /
rank 5 (session_top1 / session_top5) give the share of sessions whose true
person is first / in the first five (chance: 1/N and 5/N).

Protocols ("forward" = training data recorded before test data)
---------------------------------------------------------------
A, one day (the paper's conditions), day 1 and day 2 separately:
  A1          random stratified split, 33 % in test (the paper's split; not
              chronological, neighbouring templates on both sides).
  A2          per block: last third of the block in test, one template
              dropped before it, the rest in train. Not strictly forward:
              the end of the eyes-open block (test) precedes the eyes-closed
              block (train).
  A2_strict   per session, on the time of the first beat: the last third of
              the session in test (end of the eyes-closed block), one template
              dropped before it, everything earlier in train. Forward.
  A3          eo_to_ec: train eyes open, test eyes closed (forward);
              ec_to_eo: the reverse (backward, kept as a symmetry check).
C, rest and anger videos, one day (the paper's baseline vs anger), with
  --anger: one template per 10 s anger video (3 videos per session):
  C_pooled         rest + anger templates pooled, random stratified 33 %
                   (the paper's split over its two segments).
  C_rest_to_anger  train on rest, test on anger (the paper's
                   train-on-one-segment, test-on-the-other).
  C_anger_to_rest  the reverse (1-3 training templates per person).
B, two days (not in the paper), participants with rest on both days:
  d1_to_d2 (forward) and d2_to_d1 (backward): train on every template of one
  day, test on the other; per template and per session (probabilities
  averaged over the probe session, top-1 / top-5 / MRR, 95 % CI by bootstrap
  over participants). Control: training labels shuffled (must give chance).
  B42      random galleries of 42 participants.
  B_folds  galleries = test participants of each fold of
           results/splits/folds.json, the folds of results/leaderboard.csv,
           so the numbers compare directly with b1_ecg_beat and the
           demographic control there.
  LDA_folds  baseline with the same features: StandardScaler + LDA learnt on
           the *other* folds' participants, both days pooled (learns which
           feature directions stay stable from day to day), then cosine
           between session means, day 1 gallery -> day 2 probe. Same recipe
           as b3_lda in the leaderboard.

Feature sets
------------
all16        the 16 ECG features of Table 1 (raw).
paper8       the paper's 8-feature ECG subset.
all16_rel    amplitudes and slopes divided by RQ_amp (RQ_amp removed).
all16_zday   each feature z-scored per day, over all templates of that day
             (every participant; labels not used). Removes a shift common to
             everyone between days. A random forest is invariant to any
             per-feature affine rescaling applied to train and test alike,
             so on one day (A) this equals all16 exactly; it can only change
             B, where train and test days get their own statistics.
all16_qday   each feature replaced by its percentile rank per day (robust to
             outliers and to non-linear day effects); B only, as above.
all16_rel_zday  both.
rr_only      RR interval alone (heart rate): what heart rate alone gives.
beat         the whole template (-250/+500 ms), 75 points (one every 10 ms),
             raw amplitudes: the full beat shape, same forest.
beat_z       the same, each template centred and scaled by its own standard
             deviation (shape only, amplitude removed; as b1_ecg_beat).
beat_z_all16 beat_z and the 16 features together.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reid.features.ecg_tanaskovic import BEAT_NAMES, NAMES  # noqa: E402

EO, EC = 10002, 10004
PAPER8 = ("ECGQRScrest", "QRS_int", "QR_slope", "RQ_amp", "RS_amp", "RS_slope",
          "RT_amp", "TT2_amp")
REL = ("RS_amp", "RT_amp", "TT1_amp", "TT2_amp",
       "QR_slope", "RS_slope", "TT1_slope", "TT2_slope")
GALLERY = 42
FOLDS = REPO / "results" / "splits" / "folds.json"

ALL16 = list(NAMES)
ALL16_REL = [c for c in NAMES if c != "RQ_amp" and c not in REL] + [c + "_rel" for c in REL]
FEATURES = {
    "all16": ALL16,
    "paper8": list(PAPER8),
    "all16_rel": ALL16_REL,
    "all16_zday": [c + "_zd" for c in ALL16],
    "all16_rel_zday": [c + "_zd" for c in ALL16_REL],
    "all16_qday": [c + "_qd" for c in ALL16],
    "rr_only": ["RR"],
    "beat": list(BEAT_NAMES),
    "beat_z": [c + "_z" for c in BEAT_NAMES],
    "beat_z_all16": [c + "_z" for c in BEAT_NAMES] + ALL16,
}
ONE_DAY_SETS = ("all16", "paper8", "all16_rel", "rr_only",  # zday == raw on one day
                "beat", "beat_z", "beat_z_all16")
C_SETS = ("all16", "paper8", "beat", "beat_z", "beat_z_all16")


ANG = 1  # segment code of the anger video templates


def load(path: Path, anger: Path | None = None) -> pd.DataFrame:
    f = pd.read_csv(path)
    if "participant_id" not in f:  # local check files carry only the file name
        f["participant_id"] = f.file.str.extract(r"(P\d+)_")[0]
        f["day"] = f.file.str.extract(r"_day-(\d+)_")[0].astype(int)
    if anger is not None:
        try:
            g = pd.read_csv(anger)
        except pd.errors.EmptyDataError:  # no emotion-video recording found
            g = pd.DataFrame()
        if len(g):
            g["segment"] = ANG
            f = pd.concat([f, g], ignore_index=True)
    f = f[f.fiducials_ok.astype(bool)].copy()
    f = f[np.isfinite(f[ALL16]).all(axis=1)]
    f["segment"] = f.segment.astype(int)
    for c in REL:
        f[c + "_rel"] = f[c] / f.RQ_amp.where(f.RQ_amp.abs() > 1e-9)
    f = f[np.isfinite(f[[c + "_rel" for c in REL]]).all(axis=1)].copy()
    by = [f.day, f.segment == ANG]  # per day (rest and anger apart), labels not used
    for c in set(ALL16) | set(ALL16_REL):
        g = f.groupby(by)[c]
        f[c + "_zd"] = (f[c] - g.transform("mean")) / g.transform("std")
        f[c + "_qd"] = g.rank(pct=True)
    if set(BEAT_NAMES) <= set(f.columns):  # full beat shape (newer extractions)
        b = f[list(BEAT_NAMES)].to_numpy(dtype=float)
        bz = (b - b.mean(axis=1, keepdims=True)) / (b.std(axis=1, keepdims=True) + 1e-12)
        f = pd.concat([f, pd.DataFrame(bz, index=f.index,
                                       columns=[c + "_z" for c in BEAT_NAMES])], axis=1)
    f["pid"] = pd.factorize(f.participant_id)[0]  # identifiers never leave this script
    return f


def rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, n_jobs=-1)


def template_metrics(y, pred) -> dict:
    out = {"accuracy": accuracy_score(y, pred)}
    for avg in ("weighted", "macro"):
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        out |= {f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f1}
    return out


def fit_score(train, test, cols, seed) -> dict:
    m = rf(seed).fit(train[cols].to_numpy(), train.pid.to_numpy())
    out = template_metrics(test.pid.to_numpy(), m.predict(test[cols].to_numpy()))
    n = train.pid.nunique()
    # per session (one per person here: a single day is tested): probabilities
    # averaged over the person's test templates, then rank 1 / rank 5
    known = test[test.pid.isin(m.classes_)]
    out |= rank_metrics(session_ranks(m, known, cols), np.random.default_rng(seed))
    return out | {"n_classes": n, "chance": 1 / n, "chance_top5": min(5 / n, 1.0),
                  "n_train": len(train), "n_test": len(test),
                  "importance": dict(zip(cols, m.feature_importances_.round(4).tolist()))}


def mean_sd(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float, np.floating))]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out |= {k + "_sd": float(np.std([r[k] for r in runs])) for k in
            ("accuracy", "f1_weighted", "f1_macro", "session_top1", "session_top5") if k in out}
    if "importance" in runs[0]:
        imp = pd.DataFrame([r["importance"] for r in runs]).mean().sort_values(ascending=False)
        out["importance"] = imp.round(4).to_dict()
    return out | {"repeats": len(runs)}


def keep_min(f: pd.DataFrame, by: list[str], n: int) -> pd.DataFrame:
    return f[f.groupby(by).pid.transform("size") >= n]


def tail_split(d: pd.DataFrame, by: list[str]) -> tuple[pd.Series, pd.Series]:
    """Last third of each group (in time order) in test, the template before
    it dropped, the rest in train."""
    k = d.groupby(by).cumcount()
    n = d.groupby(by).pid.transform("size")
    n_test = np.ceil(n / 3).astype(int)
    return k < n - n_test - 1, k >= n - n_test


# ---- A: one day --------------------------------------------------------------

def a1(day, cols, seeds) -> dict:
    d = keep_min(day, ["pid"], 3)
    runs = []
    for s in seeds:
        tr, te = train_test_split(d, test_size=0.33, stratify=d.pid, random_state=s)
        runs.append(fit_score(tr, te, cols, s))
    return mean_sd(runs)


def a2(day, cols, seeds) -> dict:
    d = keep_min(day, ["pid", "segment"], 3).sort_values(["pid", "segment", "t_first_beat_s"])
    train, test = tail_split(d, ["pid", "segment"])
    return mean_sd([fit_score(d[train], d[test], cols, s) for s in seeds])


def a2_strict(day, cols, seeds) -> dict:
    d = keep_min(day, ["pid"], 3).sort_values(["pid", "t_first_beat_s"])
    train, test = tail_split(d, ["pid"])
    out = mean_sd([fit_score(d[train], d[test], cols, s) for s in seeds])
    # a test template is always later than every training template of its session
    last_train = d[train].groupby("pid").t_first_beat_s.max()
    first_test = d[test].groupby("pid").t_first_beat_s.min()
    common = last_train.index.intersection(first_test.index)
    out["check_forward"] = bool((first_test[common] > last_train[common]).all())
    out["test_share_eyes_closed"] = float((d[test].segment == EC).mean())
    return out


def a3(day, cols, seeds) -> dict:
    both = day.groupby("pid").segment.nunique() == 2
    d = day[day.pid.isin(both[both].index)]
    out = {}
    for name, a, b in (("eo_to_ec", EO, EC), ("ec_to_eo", EC, EO)):
        tr, te = d[d.segment == a], d[d.segment == b]
        out[name] = mean_sd([fit_score(tr, te, cols, s) for s in seeds])
    out["eo_to_ec"]["direction"], out["ec_to_eo"]["direction"] = "forward", "backward"
    return out


# ---- C: rest and anger videos, one day -----------------------------------------

def c_protocols(day: pd.DataFrame, cols, seeds) -> dict:
    has = day[day.segment == ANG].pid.unique()
    d = day[day.pid.isin(has)]
    rest, ang = d[d.segment != ANG], d[d.segment == ANG]
    rest = rest[rest.pid.isin(ang.pid.unique())]
    pooled = keep_min(d, ["pid"], 3)
    runs = []
    for s in seeds:
        tr, te = train_test_split(pooled, test_size=0.33, stratify=pooled.pid, random_state=s)
        runs.append(fit_score(tr, te, cols, s) | {"test_share_anger": float((te.segment == ANG).mean())})
    return {"C_pooled": mean_sd(runs),
            "C_rest_to_anger": mean_sd([fit_score(rest, ang, cols, s) for s in seeds]),
            "C_anger_to_rest": mean_sd([fit_score(ang, rest, cols, s) for s in seeds]),
            "anger_templates_per_session_median": float(ang.groupby("pid").size().median())}


# ---- B: two days -------------------------------------------------------------

def ranks_from_scores(score: pd.DataFrame) -> np.ndarray:
    """score: rows = probe pid, columns = gallery pid. Returns [n, 2]: number
    of gallery entries scored above the true one, and size of its tie group
    (itself included). Ties are broken at random, in expectation, as in
    reid.eval: a constant score gives exactly chance."""
    out = []
    for pid, row in score.iterrows():
        s, true = row.to_numpy(), row[pid]
        out.append(((s > true).sum(), (s == true).sum()))
    return np.asarray(out, dtype=float)


def session_ranks(model, test: pd.DataFrame, cols) -> np.ndarray:
    proba = pd.DataFrame(model.predict_proba(test[cols].to_numpy()),
                         index=test.index, columns=model.classes_)
    return ranks_from_scores(proba.groupby(test.pid.to_numpy()).mean())


def expected_topk(ranks: np.ndarray, k: int) -> np.ndarray:
    above, ties = ranks[:, 0], ranks[:, 1]
    return np.clip((k - above) / ties, 0, 1)


def expected_rr(ranks: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(1 / (a + np.arange(1, t + 1))) for a, t in ranks.astype(int)])


def rank_metrics(ranks: np.ndarray, rng=None, n_boot=1000) -> dict:
    top1 = expected_topk(ranks, 1)
    out = {"session_top1": float(top1.mean()),
           "session_top5": float(expected_topk(ranks, 5).mean()),
           "session_mrr": float(expected_rr(ranks).mean()), "n_probes": len(ranks)}
    if rng is not None:
        boot = [top1[rng.integers(0, len(top1), len(top1))].mean() for _ in range(n_boot)]
        out |= {"session_top1_lo": float(np.percentile(boot, 2.5)),
                "session_top1_hi": float(np.percentile(boot, 97.5))}
    return out


def both_days(f: pd.DataFrame) -> pd.DataFrame:
    both = f.groupby("pid").day.nunique() == 2
    return f[f.pid.isin(both[both].index)]


def b(f, cols, seeds) -> dict:
    d = both_days(f)
    out = {}
    for name, ga, pr in (("d1_to_d2", 1, 2), ("d2_to_d1", 2, 1)):
        tr, te = d[d.day == ga], d[d.day == pr]
        runs = []
        for s in seeds:
            m = rf(s).fit(tr[cols].to_numpy(), tr.pid.to_numpy())
            r = template_metrics(te.pid.to_numpy(), m.predict(te[cols].to_numpy()))
            n = tr.pid.nunique()
            r |= {"n_classes": n, "chance": 1 / n, "chance_top5": min(5 / n, 1.0),
                  "n_train": len(tr), "n_test": len(te),
                  "importance": dict(zip(cols, m.feature_importances_.round(4)))}
            runs.append(r | rank_metrics(session_ranks(m, te, cols), np.random.default_rng(s)))
        out[name] = mean_sd(runs) | {"direction": "forward" if ga < pr else "backward"}
    # control: shuffled training labels (day 1 -> day 2), must give chance
    tr, te = d[d.day == 1], d[d.day == 2]
    shuf = []
    for s in seeds[:3]:
        y = np.random.default_rng(100 + s).permutation(tr.pid.to_numpy())
        m = rf(s).fit(tr[cols].to_numpy(), y)
        shuf.append(rank_metrics(session_ranks(m, te, cols))["session_top1"])
    out["shuffled_labels_session_top1"] = float(np.mean(shuf))
    return out


def b42(f, cols, n_draws: int) -> dict:
    d = both_days(f)
    pids = np.asarray(sorted(d.pid.unique()))
    rng = np.random.default_rng(0)
    runs = []
    for k in range(n_draws):
        g = rng.choice(pids, size=min(GALLERY, len(pids)), replace=False)
        tr, te = d[(d.day == 1) & d.pid.isin(g)], d[(d.day == 2) & d.pid.isin(g)]
        runs.append(rank_metrics(session_ranks(rf(k).fit(tr[cols].to_numpy(),
                                                         tr.pid.to_numpy()), te, cols)))
    top1 = [r["session_top1"] for r in runs]
    return mean_sd(runs) | {"gallery": min(GALLERY, len(pids)), "chance": 1 / GALLERY,
                            "chance_top5": 5 / GALLERY,
                            "session_top1_p2.5": float(np.percentile(top1, 2.5)),
                            "session_top1_p97.5": float(np.percentile(top1, 97.5))}


def fold_ids(f: pd.DataFrame) -> list[np.ndarray]:
    """Test participants of each shared fold, as internal pid codes."""
    if not FOLDS.is_file():
        return []
    code = dict(zip(f.participant_id, f.pid))
    folds = json.loads(FOLDS.read_text())["folds"]
    return [np.asarray([code[p] for p in fo["test"] if p in code]) for fo in folds]


def pooled(ranks: list[np.ndarray], gallery: list[int], seed=0) -> dict:
    r = np.concatenate(ranks)
    return rank_metrics(r, np.random.default_rng(seed)) | {
        "chance": float(np.mean([1 / g for g in gallery])),
        "chance_top5": float(np.mean([min(5 / g, 1.0) for g in gallery])),
        "gallery_mean": float(np.mean(gallery)), "folds": len(ranks)}


def b_folds(f, cols, seeds, folds) -> dict:
    d = both_days(f)
    runs = []
    for s in seeds[:3]:
        ranks, sizes = [], []
        for ids in folds:
            g = d[d.pid.isin(ids)]
            tr, te = g[g.day == 1], g[g.day == 2]
            if tr.pid.nunique() < 2:
                continue
            m = rf(s).fit(tr[cols].to_numpy(), tr.pid.to_numpy())
            ranks.append(session_ranks(m, te, cols))
            sizes.append(tr.pid.nunique())
        runs.append(pooled(ranks, sizes, s))
    return mean_sd(runs)


def lda_folds(f, cols, folds) -> dict:
    d = both_days(f)
    ranks, sizes = [], []
    for ids in folds:
        train = d[~d.pid.isin(ids)]
        test = d[d.pid.isin(ids)]
        if test.pid.nunique() < 2:
            continue
        n_comp = min(train.pid.nunique() - 1, len(cols))
        m = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=n_comp))
        m.fit(train[cols].to_numpy(), train.pid.to_numpy())
        z = pd.DataFrame(m.transform(test[cols].to_numpy()), index=test.index)
        gal = z[test.day == 1].groupby(test.pid[test.day == 1]).mean()
        pro = z[test.day == 2].groupby(test.pid[test.day == 2]).mean()
        gal = gal.div(np.linalg.norm(gal, axis=1), axis=0)
        pro = pro.div(np.linalg.norm(pro, axis=1), axis=0)
        ranks.append(ranks_from_scores(pd.DataFrame(pro.to_numpy() @ gal.to_numpy().T,
                                                    index=pro.index, columns=gal.index)))
        sizes.append(len(gal))
    return pooled(ranks, sizes)


def counts(f: pd.DataFrame) -> dict:
    per = f.groupby(["pid", "day"]).size()
    return {
        "templates": len(f), "participants": int(f.pid.nunique()),
        "participants_day1": int(f[f.day == 1].pid.nunique()),
        "participants_day2": int(f[f.day == 2].pid.nunique()),
        "participants_both_days": int((f.groupby("pid").day.nunique() == 2).sum()),
        "templates_per_session_median": float(per.median()),
        "templates_per_session_min": int(per.min()),
        "sessions_under_3_templates": int((per < 3).sum()),
        "templates_per_block_median": f.groupby("segment").apply(
            lambda g: float(g.groupby(["pid", "day"]).size().median())).to_dict(),
        "block_start_s_median": f.groupby("segment").apply(
            lambda g: float(g.groupby(["pid", "day"]).t_first_beat_s.min().median())).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10, help="repeats of each protocol")
    ap.add_argument("--draws", type=int, default=50, help="random galleries of 42 in B42")
    ap.add_argument("--sets", nargs="+", default=list(FEATURES))
    ap.add_argument("--anger", type=Path, help="anger_features.csv (C protocols)")
    a = ap.parse_args()
    start = time.time()
    full = load(a.features, a.anger)
    f = full[full.segment != ANG]  # rest only for A and B
    seeds = list(range(a.seeds))
    folds = fold_ids(f)
    try:
        git = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        git = ""
    report = {"model": "ecg_tanaskovic_rf", "date": datetime.now().isoformat(timespec="seconds"),
              "git": git, "config": {"seeds": a.seeds, "draws": a.draws, "test_size": 0.33,
                                     "classifier": "RandomForestClassifier(defaults)",
                                     "folds": len(folds)},
              "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
              "counts": counts(f), "results": {}}
    if a.anger is not None and (full.segment == ANG).any():
        ang = full[full.segment == ANG]
        report["counts"]["anger"] = {
            "templates": len(ang), "sessions": int(ang.groupby(["pid", "day"]).ngroups),
            "templates_per_session_median": float(ang.groupby(["pid", "day"]).size().median())}
    two_days = (f.groupby("pid").day.nunique() == 2).sum() >= 2
    for name in a.sets:
        cols = FEATURES[name]
        if not set(cols) <= set(full.columns):  # e.g. beat sets on an older features.csv
            print(f"{name}: columns missing, skipped", flush=True)
            continue
        res = {}
        if name in ONE_DAY_SETS:
            for day in sorted(f.day.unique()):
                one = f[f.day == day]
                if one.pid.nunique() < 2:
                    continue
                res[f"A1_day{day}"] = a1(one, cols, seeds)
                res[f"A2_day{day}"] = a2(one, cols, seeds)
                res[f"A2_strict_day{day}"] = a2_strict(one, cols, seeds)
                res[f"A3_day{day}"] = a3(one, cols, seeds)
                r1 = res[f"A1_day{day}"]
                print(f"{name} day {day}: A1 sessions rank 1 {r1['session_top1']:.3f} "
                      f"rank 5 {r1['session_top5']:.3f}", flush=True)
                print(f"{name} day {day}: A1 {res[f'A1_day{day}']['accuracy']:.3f} "
                      f"A2 {res[f'A2_day{day}']['accuracy']:.3f} "
                      f"A2_strict {res[f'A2_strict_day{day}']['accuracy']:.3f} "
                      f"A3 eo->ec {res[f'A3_day{day}']['eo_to_ec']['accuracy']:.3f}",
                      flush=True)
                if name in C_SETS and (one_full := full[(full.day == day)]
                                                    ).segment.eq(ANG).any():
                    res[f"C_day{day}"] = c_protocols(one_full, cols, seeds)
                    c = res[f"C_day{day}"]
                    print(f"{name} day {day}: C pooled {c['C_pooled']['accuracy']:.3f} "
                          f"rest->anger {c['C_rest_to_anger']['accuracy']:.3f} "
                          f"anger->rest {c['C_anger_to_rest']['accuracy']:.3f}; rest->anger "
                          f"sessions rank 1 {c['C_rest_to_anger']['session_top1']:.3f} "
                          f"rank 5 {c['C_rest_to_anger']['session_top5']:.3f}", flush=True)
        if two_days:
            res["B"] = b(f, cols, seeds)
            res["B42"] = b42(f, cols, a.draws)
            msg = (f"{name} B d1->d2: template {res['B']['d1_to_d2']['accuracy']:.3f}, "
                   f"session top-1 {res['B']['d1_to_d2']['session_top1']:.3f} "
                   f"top-5 {res['B']['d1_to_d2']['session_top5']:.3f} "
                   f"(shuffled {res['B']['shuffled_labels_session_top1']:.3f}); "
                   f"B42 {res['B42']['session_top1']:.3f}")
            if folds:
                res["B_folds"] = b_folds(f, cols, seeds, folds)
                msg += f"; B_folds {res['B_folds']['session_top1']:.3f}"
                if len(cols) > 1:  # cosine needs at least 2 dimensions
                    res["LDA_folds"] = lda_folds(f, cols, folds)
                    msg += f"; LDA_folds {res['LDA_folds']['session_top1']:.3f}"
            print(msg, flush=True)
        report["results"][name] = res
    report["time_s"] = round(time.time() - start, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=float))
    print(f"written {a.out} in {report['time_s']} s", flush=True)


if __name__ == "__main__":
    main()
