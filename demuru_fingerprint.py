"""
Reproduction de Demuru & Fraschini (2020) sur le dataset du HABS Neuro-AI Hackathon.

Article : « EEG fingerprinting: subject-specific signature based on the aperiodic
component of power spectrum », Computers in Biology and Medicine 120 (2020) 103748.
Préprint libre : https://arxiv.org/abs/2001.09424

Ce que fait l'article                       Ce que fait ce script
-----------------------------------------   ---------------------------------------------
109 sujets, 1 visite, 64 électrodes,         Participants HABS, jour 1 et jours suivants,
160 Hz, 1 min yeux ouverts + 1 min           62 électrodes, 1 000 Hz ramenés à 250 Hz,
yeux fermés                                  bloc resting-state : ~30 s yeux ouverts
                                             (marqueur 10002 ou 10000) + 60 s yeux fermés
                                             (marqueur 10004)
ICA (ADJUST) pour retirer les artefacts      Fichiers PREPROCESSED : déjà filtrés 0,5-48 Hz,
                                             référence moyenne, ICA (Picard) + ICLabel
5 fenêtres de 12 s                           Fenêtres de 12 s : 5 en yeux fermés, 2 en yeux
                                             ouverts (le bloc ne dure que 30 s)
Welch, FOOOF : offset et pente (exposant)    Identique (fenêtres de Welch de 2 s, FOOOF 1-45 Hz)
Puissance relative thêta, alpha, bêta,       Identique (bandes de l'article, total 1-45 Hz)
gamma
Similarité 1 / (1 + distance euclidienne)    Identique
EER sur des paires de la même visite         Test A : même jour (comme l'article)
                                             Test B : enrôlement jour 1, test jours suivants
                                             + identification au rang 1

Les paramètres exacts de Welch et de FOOOF ne sont pas donnés dans l'article :
ceux utilisés ici sont des choix standard, regroupés dans CONFIG ci-dessous.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# fooof affiche un avertissement de dépréciation à l'import : il fonctionne, on le masque
warnings.filterwarnings("ignore", message=".*fooof.*deprecated.*")

# =============================================================================
# CONFIG : tout ce qu'on peut vouloir changer est ici
# =============================================================================
CONFIG = {
    # "PREPROCESSED" (déjà nettoyé, recommandé) ou "CURATED" (brut : on filtre nous-mêmes, sans ICA)
    "data_version": "PREPROCESSED",
    # None = tous les participants ; un nombre (ex. 30) pour un test rapide
    "max_participants": None,
    # Nombre de processus en parallèle (-1 = tous les cœurs)
    "n_jobs": -1,
    # Rééchantillonnage : les données sont filtrées sous 48 Hz, 250 Hz suffit et va 4 fois plus vite
    "resample_hz": 250.0,
    # Découpe, comme l'article : fenêtres de 12 s sans recouvrement
    "window_s": 12.0,
    # Conditions : marqueurs de début (voir markers.csv) et nombre maximal de fenêtres
    "conditions": {
        "EC": {"label": "yeux fermés", "start_markers": [10004], "max_windows": 5},
        "EO": {"label": "yeux ouverts", "start_markers": [10002, 10000], "max_windows": 5},
    },
    # Électrodes retirées : T9 et T10 sont déjà exclues dans PREPROCESSED,
    # on les retire aussi en CURATED pour comparer les mêmes 62 électrodes
    "exclude_channels": ["T9", "T10"],
    # Spectre de Welch
    "welch_segment_s": 2.0,
    "welch_overlap": 0.5,
    # FOOOF
    "fit_range_hz": (1.0, 45.0),
    "fooof_settings": {
        "peak_width_limits": (1.0, 12.0),
        "max_n_peaks": 6,
        "min_peak_height": 0.05,
        "peak_threshold": 2.0,
        "aperiodic_mode": "fixed",
        "verbose": False,
    },
    # Bandes de l'article (puissance relative = bande / total 1-45 Hz)
    "bands": {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)},
    # Contrôle qualité d'une fenêtre
    "reject_peak_to_peak_uv": 300.0,   # rejet si une électrode dépasse cette amplitude crête à crête
    "min_median_r2": 0.80,             # rejet si l'ajustement FOOOF est mauvais (R² médian)
    # Jour d'enrôlement pour le test B
    "enroll_day": 1,
}

# Les jeux de features comparés, comme dans l'article
FEATURE_SETS = {
    "offset": ["offset"],
    "exposant": ["exponent"],
    "offset+exposant": ["offset", "exponent"],
    "thêta (rel.)": ["rel_theta"],
    "alpha (rel.)": ["rel_alpha"],
    "bêta (rel.)": ["rel_beta"],
    "gamma (rel.)": ["rel_gamma"],
}

# EER publiés par Demuru & Fraschini 2020 (tableau des résultats), pour comparaison
PAPER_EER = {
    ("EO", "exposant"): 0.090, ("EC", "exposant"): 0.089,
    ("EO", "offset"): 0.079, ("EC", "offset"): 0.057,
    ("EO", "offset+exposant"): 0.063, ("EC", "offset+exposant"): 0.171,
    ("EO", "thêta (rel.)"): 0.182, ("EC", "thêta (rel.)"): 0.238,
    ("EO", "alpha (rel.)"): 0.256, ("EC", "alpha (rel.)"): 0.181,
    ("EO", "bêta (rel.)"): 0.118, ("EC", "bêta (rel.)"): 0.142,
    ("EO", "gamma (rel.)"): 0.112, ("EC", "gamma (rel.)"): 0.197,
}

FILE_RE = re.compile(r"(P\d+)_day-(\d+)_task-resting-state_multimodal\.bdf$")


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# =============================================================================
# 1. Trouver les fichiers
# =============================================================================
def find_recordings(data_dir: Path, cfg: dict) -> pd.DataFrame:
    """Liste les enregistrements resting-state : participant, jour, chemin."""
    root = data_dir / cfg["data_version"]
    rows = []
    for bdf in sorted(root.glob("P*/P*_day-*_task-resting-state_multimodal.bdf")):
        m = FILE_RE.search(bdf.name)
        if m:
            rows.append({"participant": m.group(1), "day": int(m.group(2)), "path": str(bdf)})
    rec = pd.DataFrame(rows, columns=["participant", "day", "path"])
    if cfg["max_participants"] and not rec.empty:
        keep = sorted(rec.participant.unique())[: int(cfg["max_participants"])]
        rec = rec[rec.participant.isin(keep)]
    return rec.reset_index(drop=True)


# =============================================================================
# 2. Charger l'EEG d'un fichier
# =============================================================================
def load_eeg(bdf: Path, cfg: dict):
    """Charge uniquement les électrodes EEG, retire le remplissage final, prépare le signal."""
    import mne

    record_map = json.loads(bdf.with_suffix(".record_map").read_text(encoding="utf-8"))
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])

    # Électrodes EEG d'après le record_map (indices de canaux et noms)
    eeg_names = []
    for stream in record_map["stream_map"].values():
        for spec in stream.values():
            corr = spec.get("correction", {})
            if str(corr.get("RecordType", "")).lower() != "eeg":
                continue
            if "channel_ranges" in spec:
                first, last = spec["channel_ranges"]
                idx = list(range(int(first), int(last) + 1))
            else:
                idx = list(spec.get("channel_indices", []))
            for i, detail in zip(idx, corr.get("Channels", [])):
                label = raw.ch_names[i]
                if not (detail.get("name") == label or detail.get("name", "").startswith(label)):
                    raise ValueError(f"record_map et BDF ne concordent pas au canal {i}")
                eeg_names.append(label)
    if not eeg_names:
        raise ValueError("aucune électrode EEG trouvée dans le record_map")

    # Retirer le remplissage : on ne garde que TotalSamples points
    n_valid = min(int(record_map.get("TotalSamples", raw.n_times)), raw.n_times)
    raw.crop(tmax=(n_valid - 1) / sfreq)

    picks = [c for c in eeg_names if c not in cfg["exclude_channels"]]
    raw.pick(picks)
    raw.load_data(verbose="ERROR")
    raw.set_channel_types({c: "eeg" for c in picks}, verbose="ERROR")

    if cfg["data_version"] == "CURATED":
        # Données brutes : même filtrage que PREPROCESSED, mais sans ICA
        raw.notch_filter(50.0, verbose="ERROR")
        raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")

    if cfg["resample_hz"] and cfg["resample_hz"] < sfreq:
        raw.resample(cfg["resample_hz"], verbose="ERROR")
    return raw


def condition_segments(raw, bdf: Path, cfg: dict) -> dict:
    """Début et fin de chaque condition (yeux ouverts / fermés), en secondes dans le fichier."""
    ann = raw.annotations
    codes = []
    for d, o in zip(ann.description, ann.onset):
        try:
            codes.append((int(str(d).strip()), float(o)))
        except ValueError:
            pass
    t_end = raw.n_times / raw.info["sfreq"]
    segs = {}
    for cond, spec in cfg["conditions"].items():
        for start_code in spec["start_markers"]:
            starts = [o for c, o in codes if c == start_code]
            if not starts:
                continue
            start = starts[0]
            ends = [o for c, o in codes if c == start_code + 1 and o > start]
            end = min(ends[0], t_end) if ends else t_end
            segs[cond] = (start, end, start_code)
            break
    if not segs:
        segs = segments_from_metadata(bdf, cfg, t_end)
    return segs


def segments_from_metadata(bdf: Path, cfg: dict, t_end: float) -> dict:
    """Solution de secours : temps des marqueurs dans le JSON de session."""
    m = FILE_RE.search(bdf.name)
    meta_path = bdf.parent / f"{m.group(1)}_day-{m.group(2)}_metadata.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seg = meta.get("dataset_export", {}).get("segments", {}).get("resting-state", {})
    origin = float(seg.get("metadata_time_origin_s", 0.0))
    segs = {}
    for cond, spec in cfg["conditions"].items():
        for ev in meta.get("events", []):
            if ev.get("marker_id") in spec["start_markers"]:
                start = float(ev["onset_s"]) - origin
                end = start + float(ev.get("duration_s") or 0.0)
                if 0 <= start < t_end:
                    segs[cond] = (start, min(end, t_end), ev["marker_id"])
                    break
    return segs


# =============================================================================
# 3. Features d'une fenêtre : Welch, FOOOF, puissances relatives
# =============================================================================
def window_features(data_uv: np.ndarray, sfreq: float, cfg: dict) -> dict:
    from fooof import FOOOFGroup
    from scipy.integrate import trapezoid
    from scipy.signal import welch

    nper = int(round(cfg["welch_segment_s"] * sfreq))
    freqs, psd = welch(data_uv, fs=sfreq, window="hann", nperseg=nper,
                       noverlap=int(nper * cfg["welch_overlap"]), axis=-1)   # µV²/Hz

    lo, hi = cfg["fit_range_hz"]
    fg = FOOOFGroup(**cfg["fooof_settings"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg.fit(freqs, psd, [lo, hi], n_jobs=1)
    ap = fg.get_params("aperiodic_params")          # colonnes : offset, exposant
    r2 = fg.get_params("r_squared")

    total_mask = (freqs >= lo) & (freqs <= hi)
    total = trapezoid(psd[:, total_mask], freqs[total_mask], axis=1)
    feats = {"offset": ap[:, 0], "exponent": ap[:, 1], "r2": r2}
    for band, (b0, b1) in cfg["bands"].items():
        mask = (freqs >= b0) & (freqs <= b1)
        feats[f"rel_{band}"] = trapezoid(psd[:, mask], freqs[mask], axis=1) / total
    return feats


# =============================================================================
# 4. Traiter un fichier entier
# =============================================================================
def process_recording(participant: str, day: int, path: str, cfg: dict):
    """Retourne (lignes de features, ligne de journal) pour un enregistrement."""
    t0 = time.time()
    entry = {"participant": participant, "day": day, "file": Path(path).name, "status": "ok", "message": ""}
    rows = []
    try:
        raw = load_eeg(Path(path), cfg)
        sfreq = float(raw.info["sfreq"])
        chans = raw.ch_names
        segs = condition_segments(raw, Path(path), cfg)
        if not segs:
            raise ValueError("marqueurs yeux ouverts / fermés introuvables")
        data = raw.get_data() * 1e6                      # volts -> µV
        win = int(round(cfg["window_s"] * sfreq))
        for cond, (start, end, code) in segs.items():
            n_possible = int((end - start) * sfreq) // win
            n_take = min(n_possible, cfg["conditions"][cond]["max_windows"])
            kept = 0
            for w in range(n_take):
                a = int(round(start * sfreq)) + w * win
                x = data[:, a:a + win]
                p2p = float(np.max(np.ptp(x, axis=1)))
                reason = ""
                if p2p > cfg["reject_peak_to_peak_uv"]:
                    reason = f"amplitude {p2p:.0f} µV"
                else:
                    f = window_features(x, sfreq, cfg)
                    med_r2 = float(np.nanmedian(f["r2"]))
                    if not np.all(np.isfinite(f["offset"])) or med_r2 < cfg["min_median_r2"]:
                        reason = f"FOOOF R² médian {med_r2:.2f}"
                if reason:
                    entry["message"] += f"{cond} fenêtre {w} rejetée ({reason}); "
                    continue
                row = {"participant": participant, "day": day, "condition": cond, "window": w,
                       "start_s": round(a / sfreq, 3), "marker": code, "median_r2": med_r2,
                       "p2p_max_uv": round(p2p, 1)}
                for key in ["offset", "exponent"] + [f"rel_{b}" for b in cfg["bands"]]:
                    for ch, v in zip(chans, f[key]):
                        row[f"{key}__{ch}"] = float(v)
                rows.append(row)
                kept += 1
            entry[f"n_windows_{cond}"] = kept
        entry["n_channels"] = len(chans)
    except Exception as exc:  # un fichier défectueux ne doit pas arrêter tout le calcul
        entry["status"] = "erreur"
        entry["message"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc(limit=3)
    entry["seconds"] = round(time.time() - t0, 1)
    return rows, entry


# =============================================================================
# 5. Évaluation : EER (vérification) et rang 1 (identification)
# =============================================================================
def eer_auc(genuine: np.ndarray, impostor: np.ndarray):
    """EER = point où faux acceptés = faux rejets ; AUC de la courbe ROC."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    d = fnr - fpr
    j = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(j):
        j = j[0]
        t = d[j] / (d[j] - d[j + 1])
        eer = fpr[j] + t * (fpr[j + 1] - fpr[j])
        threshold = thr[j] + t * (thr[j + 1] - thr[j])
    else:
        i = int(np.argmin(np.abs(d)))
        eer, threshold = (fpr[i] + fnr[i]) / 2, thr[i]
    return float(eer), float(roc_auc_score(y, s)), float(threshold)


def similarity(dist: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + dist)          # formule de l'article


def test_same_session(X: np.ndarray, subj: np.ndarray):
    """Test A : toutes les paires de fenêtres d'un même enregistrement (comme l'article)."""
    from scipy.spatial.distance import pdist

    iu = np.triu_indices(len(subj), k=1)
    same = (subj[:, None] == subj[None, :])[iu]
    s = similarity(pdist(X))
    eer, auc, thr = eer_auc(s[same], s[~same])

    # Identification au rang 1 : chaque fenêtre contre la moyenne des AUTRES fenêtres de chaque personne
    ids = np.unique(subj)
    sums = np.vstack([X[subj == p].sum(0) for p in ids])
    counts = np.array([(subj == p).sum() for p in ids])
    correct, total = 0, 0
    for k, x in enumerate(X):
        p = np.where(ids == subj[k])[0][0]
        if counts[p] < 2:
            continue
        templates = sums / counts[:, None]
        templates[p] = (sums[p] - x) / (counts[p] - 1)
        correct += int(np.argmin(np.linalg.norm(templates - x, axis=1)) == p)
        total += 1
    rank1 = correct / total if total else np.nan
    return dict(eer=eer, auc=auc, threshold=thr, rank1=rank1, n_genuine=int(same.sum()),
                n_impostor=int((~same).sum())), s[same], s[~same]


def test_cross_day(Xe, se, Xt, st):
    """Test B : fenêtres du jour d'enrôlement contre fenêtres des jours suivants."""
    from scipy.spatial.distance import cdist

    S = similarity(cdist(Xe, Xt))
    same = se[:, None] == st[None, :]
    eer, auc, thr = eer_auc(S[same], S[~same])

    ids = np.unique(se)
    templates = np.vstack([Xe[se == p].mean(0) for p in ids])
    nearest = ids[np.argmin(cdist(Xt, templates), axis=1)]
    rank1 = float(np.mean(nearest == st))
    return dict(eer=eer, auc=auc, threshold=thr, rank1=rank1, n_genuine=int(same.sum()),
                n_impostor=int((~same).sum())), S[same], S[~same]


def evaluate(feats: pd.DataFrame, cfg: dict, out_dir: Path) -> pd.DataFrame:
    results, scores_for_plot = [], {}
    enroll = cfg["enroll_day"]
    for cond in cfg["conditions"]:
        fc = feats[feats.condition == cond]
        if fc.empty:
            continue
        both = set(fc[fc.day == enroll].participant) & set(fc[fc.day > enroll].participant)
        for fs_name, prefixes in FEATURE_SETS.items():
            # Électrodes présentes dans toutes les fenêtres (les autres sont ignorées)
            fcols = [c for c in fc.columns if c.split("__")[0] in prefixes and fc[c].notna().all()]
            sub = fc[["participant", "day"] + fcols]
            if not fcols:
                continue
            base = {"condition": cond, "features": fs_name, "n_features": len(fcols),
                    "eer_article": PAPER_EER.get((cond, fs_name), np.nan)}

            # Test A sur tous les participants (jour d'enrôlement), puis sur ceux qui ont aussi un jour suivant
            day1 = sub[sub.day == enroll]
            # Le test A demande au moins 2 fenêtres par personne (sinon aucune paire « même personne »)
            n_win = day1.groupby("participant").size()
            day1 = day1[day1.participant.isin(n_win[n_win >= 2].index)]
            for name, part in [("A_meme_jour_tous", day1),
                               ("A_meme_jour_sous_groupe", day1[day1.participant.isin(both)])]:
                if part.participant.nunique() < 2:
                    continue
                r, g, i = test_same_session(part[fcols].to_numpy(), part.participant.to_numpy())
                results.append({**base, "test": name, "n_participants": part.participant.nunique(),
                                "n_fenetres": len(part), **r})
                if fs_name == "offset" and name == "A_meme_jour_sous_groupe":
                    scores_for_plot[(cond, "A")] = (g, i, r["threshold"])
                if fs_name == "offset" and name == "A_meme_jour_tous" and (cond, "A") not in scores_for_plot:
                    scores_for_plot[(cond, "A")] = (g, i, r["threshold"])

            # Test B : enrôlement jour 1, test jours suivants
            if len(both) >= 2:
                e = sub[(sub.day == enroll) & sub.participant.isin(both)]
                t = sub[(sub.day > enroll) & sub.participant.isin(both)]
                r, g, i = test_cross_day(e[fcols].to_numpy(), e.participant.to_numpy(),
                                         t[fcols].to_numpy(), t.participant.to_numpy())
                results.append({**base, "test": "B_jour1_vers_jours_suivants", "n_participants": len(both),
                                "n_fenetres": len(e) + len(t), **r})
                if fs_name == "offset":
                    scores_for_plot[(cond, "B")] = (g, i, r["threshold"])

    res = pd.DataFrame(results)
    if not res.empty:
        res["chance_rank1"] = 1.0 / res["n_participants"]
        plot_results(res, scores_for_plot, cfg, out_dir)
    return res


# =============================================================================
# 6. Figures
# =============================================================================
def plot_results(res: pd.DataFrame, scores: dict, cfg: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    grey, teal, orange = "#9AA5B1", "#2A7F8E", "#C8613D"

    for cond in res.condition.unique():
        r = res[res.condition == cond]
        names = list(FEATURE_SETS)
        def val(test):
            d = r[r.test == test].set_index("features")["eer"]
            return [d.get(n, np.nan) for n in names]
        a_test = "A_meme_jour_sous_groupe" if (r.test == "A_meme_jour_sous_groupe").any() else "A_meme_jour_tous"
        series = [("Article (même visite)", [PAPER_EER.get((cond, n), np.nan) for n in names], grey),
                  ("Nous, même jour", val(a_test), teal),
                  ("Nous, jour 1 → jour suivant", val("B_jour1_vers_jours_suivants"), orange)]
        x = np.arange(len(names))
        w = 0.26
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for k, (lab, v, col) in enumerate(series):
            ax.bar(x + (k - 1) * w, v, width=w - 0.03, color=col, label=lab)
        ax.set_xticks(x, names)
        ax.set_ylabel("EER (plus bas = mieux)")
        ax.set_title(f"EER par jeu de features, {cfg['conditions'][cond]['label']}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6EAF0")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"eer_par_features_{cond}.png", dpi=150)
        plt.close(fig)

        panels = [(k, scores[(cond, k)]) for k in ("A", "B") if (cond, k) in scores]
        if panels:
            fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.8), squeeze=False)
            for ax, (k, (g, i, thr)) in zip(axs[0], panels):
                if len(g) == 0 or len(i) == 0:
                    continue
                bins = np.linspace(min(g.min(), i.min()), max(g.max(), i.max()), 50)
                ax.hist(i, bins=bins, density=True, color=grey, alpha=0.8, label="imposteurs (personnes différentes)")
                ax.hist(g, bins=bins, density=True, color=teal, alpha=0.7, label="même personne")
                ax.axvline(thr, color="#1D2433", lw=1, ls="--", label="seuil à l'EER")
                ax.set_title("Test A : même jour" if k == "A" else "Test B : jour 1 → jour suivant")
                ax.set_xlabel("similarité 1 / (1 + distance), offset")
                ax.spines[["top", "right"]].set_visible(False)
            axs[0][0].set_ylabel("densité")
            axs[0][0].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"scores_offset_{cond}.png", dpi=150)
            plt.close(fig)


# =============================================================================
# 7. Programme principal
# =============================================================================
def run(data_dir: Path, out_dir: Path, cfg: dict | None = None) -> None:
    cfg = {**CONFIG, **(cfg or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = find_recordings(data_dir, cfg)
    log(f"{len(rec)} enregistrement(s) resting-state trouvés dans {data_dir / cfg['data_version']}, "
        f"{rec.participant.nunique() if not rec.empty else 0} participant(s)")
    if rec.empty:
        log("Rien à traiter : vérifiez le chemin des données et data_version.")
        return

    from joblib import Parallel, delayed

    n_jobs = cfg["n_jobs"] if cfg["n_jobs"] != -1 else max(1, (os.cpu_count() or 1))
    log(f"Calcul des features sur {n_jobs} processus...")
    t0 = time.time()
    outputs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_recording)(r.participant, int(r.day), r.path, cfg) for r in rec.itertuples())
    rows = [row for rws, _ in outputs for row in rws]
    journal = pd.DataFrame([e for _, e in outputs])
    journal.to_csv(out_dir / "journal_fichiers.csv", index=False)
    log(f"Features calculées en {time.time() - t0:.0f} s ; "
        f"{(journal.status == 'erreur').sum()} fichier(s) en erreur (voir journal_fichiers.csv)")

    feats = pd.DataFrame(rows)
    if feats.empty:
        log("Aucune fenêtre valide : rien à évaluer.")
        return
    feats.to_csv(out_dir / "features_fenetres.csv.gz", index=False)

    res = evaluate(feats, cfg, out_dir)
    lines = ["Reproduction de Demuru & Fraschini 2020 sur le dataset HABS", "",
             f"Données : {cfg['data_version']}, {feats.participant.nunique()} participant(s), "
             f"{len(feats)} fenêtre(s) de {cfg['window_s']:.0f} s", ""]
    if res.empty:
        lines.append("Pas assez de participants pour évaluer (il en faut au moins 2).")
    else:
        res.to_csv(out_dir / "resultats.csv", index=False)
        show = res[["condition", "test", "features", "n_participants", "eer", "eer_article", "auc", "rank1", "chance_rank1"]]
        lines.append(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        lines += ["", "eer : taux d'erreur de vérification (plus bas = mieux) ; eer_article : valeur publiée",
                  "rank1 : part des fenêtres attribuées à la bonne personne ; chance_rank1 : hasard"]
    (out_dir / "resume.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    log(f"Terminé. Résultats dans {out_dir}")
