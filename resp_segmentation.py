"""Cycles respiratoires (ceinture thoracique) — validation sur 20 séances, puis extraction.

Décisions validées par l'équipe :
  - repos, YEUX FERMÉS uniquement : marqueur 10004 + 3 s  ->  marqueur 10005 - 7 s ;
  - polarité -1 (signal inversé : l'inspiration monte) ; seuil de phase 10 % ;
  - rejets : Ttot hors [1,5 ; 15] s ; Ti ou Te < 0,3 s ; Ti/Ttot hors [0,1 ; 0,7] ;
    amplitude < 0,3 x la médiane ; grande excursion (conservée, marquée, hors médianes) ;
  - « Ttot atypique » (> 3 écarts absolus médians) : SIGNALÉ seulement, jamais rejeté.

Le script ne modifie pas les données : lecture seule des BDF PREPROCESSED.

Deux modes :
  python resp_segmentation.py --mode validation --data ..\\data --sessions sessions_polarite.csv --out ..\\data\\validation_v3
  python resp_segmentation.py --mode extraction --data ..\\data --out ..\\data\\extraction
  (sur la machine HABS : process.py appelle extraire_tout(data_dir, out_dir), voir process_respiration.py)

Mode validation (20 séances) -> --out :
  rapport_validation.txt   LE fichier à lire en premier : tous les chiffres demandés
  planche_controle.png     vignettes des cycles récupérés, des excursions et des cycles atypiques
  synthese_segmentation.png, seance_<P>_jour<d>.png, cycles_<P>_jour<d>.csv
  bilan_validation.csv     une ligne par séance (anciennes vs nouvelles règles)
  sensibilite_seuils.csv / sensibilite_resume.csv / sensibilite_seuils.png, parametres.csv

Mode extraction (participants avec >= 2 séances de repos) -> --out :
  cycles_tous.csv          un cycle par ligne, toutes séances (statut, raison, indicateurs)
  signatures_seances.csv   caractéristiques agrégées par participant x séance x condition
  qualite_seances.csv      indicateurs de qualité par séance ; echecs.csv ; parametres.csv

Anti-fuite : chaque séance est traitée SEULE. Aucune statistique n'est calculée sur
plusieurs séances ou plusieurs participants (toutes les normalisations, ex. amplitude
relative, sont internes à la séance). Une éventuelle standardisation de population
devra être apprise plus tard, au moment de l'identification, sur les séances
d'enrôlement (1re séance) uniquement.

────────────────────────────────────────────────────────────────────────────────
Méthode (signal inversé : creux = fin d'expiration, pic = fin d'inspiration)
1. Filtrage Butterworth ordre 2 aller-retour (phase nulle), padding miroir de 20 s :
   détection 0,05-0,8 Hz ; mesure 0,05-3 Hz ; filtrage sur tout le fichier puis découpe.
2. Échelle typique, indépendante de la segmentation : 25e centile des étendues
   P90-P10 sur des tranches glissantes de 6 s. Proéminence minimale = 0,25 x échelle.
3. Extrema : 1,5 s minimum entre deux extrema de même type ; alternance stricte ;
   position affinée à ±250 ms sur le signal de mesure.
4. Phases : décollage = dernier instant avant la pente max où la pente < 10 % de cette
   pente max ; L = niveau au décollage ; A = pic - L ;
   début d'inspiration = dernier instant <= L + seuil x A entre décollage et pic ;
   Ti = pic - début ; Te = début suivant - pic ; zone de pause = signal <= L_suivant + seuil x A ;
   Tpause = début suivant - entrée dans la zone ; Te = Te_actif + Tpause ; Ttot = Ti + Te.
5. Grandes excursions, dans les DEUX sens (niveau de base = médiane des L de la séance,
   amplitude de référence = médiane des A) :
     vers le haut : amplitude d'inspiration > 3 x la médiane ;
     vers le bas  : le signal descend sous le niveau de base de plus de 1 x l'amplitude médiane.
   Catégorie (indicative, rien n'est supprimé) :
     « soupir probable »        : vers le haut seulement, montée et descente lisses
                                  (bruit > 3 Hz < 3 x la médiane), phases >= 0,3 s ;
     « artefact probable »      : bruit > 3 Hz >= 3 x la médiane (même sans grande amplitude),
                                  OU excursion vers le bas seule ;
     « excursion indéterminée » : les autres cas (ex. haut + bas : un soupir suivi du
                                  rebond du filtre ressemble à un mouvement).
   Ces cycles restent dans les tables ; ils sont exclus des médianes, et comptés
   (n_soupirs_probables, n_artefacts_probables) dans les signatures.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent / "starting_notebooks"))
from bdf_loader import load_bdf  # noqa: E402

PARAMS = {
    "canal": "Resp", "source": "PREPROCESSED", "polarite": -1, "condition": "repos_yeux_fermes",
    "marqueur_debut": "10004", "marqueur_fin": "10005",
    "exclusion_debut_s": 3.0, "exclusion_fin_s": 7.0,
    "bande_detection_hz": (0.05, 0.8), "bande_mesure_hz": (0.05, 3.0),
    "ordre_filtre": 2, "padding_s": 20.0,
    "echelle_fenetre_s": 6.0, "echelle_pas_s": 1.0, "echelle_percentiles": (10, 90), "echelle_centile": 25,
    "proeminence_frac": 0.25, "distance_meme_type_s": 1.5, "fenetre_affinage_s": 0.25,
    "lissage_pente_s": 0.10, "seuil_pente_decollage": 0.10,
    "seuil_phase": 0.10, "seuils_sensibilite": (0.05, 0.10, 0.15, 0.20),
    "ttot_bornes_s": (1.5, 15.0), "titot_bornes": (0.1, 0.7), "phase_min_s": 0.3,
    "amp_min_rel": 0.3, "excursion_haut_rel": 3.0, "excursion_bas_rel": 1.0, "bruit_artefact_rel": 3.0,
    "ttot_mad_signalement": 3.0,
    "n_harmoniques": 6, "cycles_min_signature": 5,
    "tolerance_bord": 0.05,
    # règles de la version précédente, recalculées pour comparaison uniquement
    "anciennes_regles": {"titot_bornes": (0.2, 0.7), "ttot_mad_rejet": 3.0},
}
# ── Lecture et filtrage ──────────────────────────────────────────────────────
def filtre(x, fs, bande):
    sos = butter(PARAMS["ordre_filtre"], bande, "bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, padtype="even", padlen=min(int(PARAMS["padding_s"] * fs), len(x) - 1))


def lire_seance(data: Path, participant: str, jour: int) -> dict:
    bdf = data / PARAMS["source"] / participant / f"{participant}_day-{jour}_task-resting-state_multimodal.bdf"
    raw, df, ch = load_bdf(bdf)
    marqueurs = {str(a["description"]): float(a["onset"]) for a in raw.annotations}
    if PARAMS["marqueur_debut"] not in marqueurs:
        raise ValueError(f"marqueur {PARAMS['marqueur_debut']} absent : {sorted(marqueurs)}")
    t_fin_fichier = float(df.time_s.iloc[-1])
    debut = marqueurs[PARAMS["marqueur_debut"]] + PARAMS["exclusion_debut_s"]
    fin = min(marqueurs.get(PARAMS["marqueur_fin"], t_fin_fichier), t_fin_fichier) - PARAMS["exclusion_fin_s"]
    return {"x": df[PARAMS["canal"]].to_numpy(dtype=float).copy(), "t": df.time_s.to_numpy(),
            "fs": float(raw.info["sfreq"]), "unite": ch.set_index("bdf_label").loc[PARAMS["canal"], "header_unit"],
            "debut": debut, "fin": fin, "marqueurs": marqueurs}


def preparer(s: dict) -> dict:
    """Inverse, filtre sur tout l'enregistrement, découpe la fenêtre, contrôle le bord de fin."""
    fs, t = s["fs"], s["t"]
    x = PARAMS["polarite"] * (s["x"] - np.median(s["x"]))
    det_full, mes_full = filtre(x, fs, PARAMS["bande_detection_hz"]), filtre(x, fs, PARAMS["bande_mesure_hz"])
    fen = (t >= s["debut"]) & (t <= s["fin"])
    # bord de fin : on simule une fin de fichier 10 s plus tôt et on mesure l'écart à la même distance du bord
    L = len(x) - int(10 * fs)
    tr = filtre(x[:L], fs, PARAMS["bande_detection_hz"])
    a, b = L - int((PARAMS["exclusion_fin_s"] + 3) * fs), L - int(PARAMS["exclusion_fin_s"] * fs)
    det = det_full[fen]
    return {"fs": fs, "tw": t[fen] - s["debut"], "xw": x[fen], "det": det, "mes": mes_full[fen],
            "bruit": sosfiltfilt(butter(2, 3.0, "highpass", fs=fs, output="sos"), x)[fen],   # contenu > 3 Hz
            "x_full": x, "t_full": t, "ecart_bord_fin": float(np.abs(tr[a:b] - det_full[a:b]).max() / np.ptp(det))}


# ── Détection des extrema ────────────────────────────────────────────────────
def echelle_typique(det, fs) -> float:
    """25e centile des étendues P90-P10 sur des tranches glissantes de 6 s : indépendant de la segmentation."""
    w, p = int(PARAMS["echelle_fenetre_s"] * fs), int(PARAMS["echelle_pas_s"] * fs)
    lo, hi = PARAMS["echelle_percentiles"]
    etendues = [np.subtract(*np.percentile(det[i:i + w], [hi, lo])) for i in range(0, max(len(det) - w, 1), p)]
    return float(np.percentile(etendues, PARAMS["echelle_centile"]))


def alterner_et_affiner(pics, creux, det, mes, fs):
    ev = sorted([(i, 1) for i in pics] + [(i, -1) for i in creux])
    garde = []
    for i, sgn in ev:
        if garde and garde[-1][1] == sgn:
            if sgn * det[i] > sgn * det[garde[-1][0]]:
                garde[-1] = (i, sgn)
        else:
            garde.append((i, sgn))
    w = int(PARAMS["fenetre_affinage_s"] * fs)
    out = []
    for i, sgn in garde:
        a, b = max(i - w, 0), min(i + w + 1, len(mes))
        out.append((a + (np.argmax(mes[a:b]) if sgn > 0 else np.argmin(mes[a:b])), sgn))
    return np.array([i for i, sgn in out if sgn > 0], int), np.array([i for i, sgn in out if sgn < 0], int)


def extrema_v2(det, mes, fs):
    """Nouvelle détection : échelle robuste + distance fixe de 1,5 s entre extrema de même type."""
    ech = echelle_typique(det, fs)
    kw = {"distance": int(PARAMS["distance_meme_type_s"] * fs), "prominence": PARAMS["proeminence_frac"] * ech}
    pics, _ = find_peaks(det, **kw)
    creux, _ = find_peaks(-det, **kw)
    p, c = alterner_et_affiner(pics, creux, det, mes, fs)
    return p, c, ech


# ── Phases et cycles ─────────────────────────────────────────────────────────
def montees(mes, pics, creux, fs, seuil):
    """Pour chaque montée creux -> pic : décollage, niveau L, amplitude A, début d'inspiration."""
    k = max(int(PARAMS["lissage_pente_s"] * fs), 1)
    pente = np.convolve(np.gradient(mes) * fs, np.ones(k) / k, mode="same")
    rows = []
    for c in creux:
        suivants = pics[pics > c]
        if len(suivants) == 0:
            continue
        p = suivants[0]
        if (creux[(creux > c) & (creux < p)]).size:
            continue
        seg = pente[c:p + 1]
        if len(seg) < 3:
            continue
        kmax = int(np.argmax(seg))
        bas = np.where(seg[:kmax + 1] < PARAMS["seuil_pente_decollage"] * seg[kmax])[0]
        ts = c + (int(bas[-1]) if len(bas) else 0)
        L = mes[ts]
        A = mes[p] - L
        if A <= 0:
            continue
        sous = np.where(mes[ts:p + 1] <= L + seuil * A)[0]
        rows.append({"creux": c, "decollage": ts, "L": L, "A": A, "pic": p,
                     "debut_insp": ts + (int(sous[-1]) if len(sous) else 0)})
    return pd.DataFrame(rows)



def regles_et_indicateurs(c: pd.DataFrame, mes, bruit, fs) -> pd.DataFrame:
    """Durées, rejets (nouvelles règles), signalements, excursions, et anciennes règles pour comparaison."""
    c["t_debut_s"] = c.debut_insp / fs
    c["Ti"] = (c.pic - c.debut_insp) / fs
    c["Te"] = (c.debut_insp_suiv - c.pic) / fs
    c["Te_actif"] = (c.fin_exp - c.pic) / fs
    c["Tpause"] = (c.debut_insp_suiv - c.fin_exp) / fs
    c["Ttot"] = c.Ti + c.Te
    c["Ti_Ttot"] = c.Ti / c.Ttot
    assert np.allclose(c.Te, c.Te_actif + c.Tpause) and np.allclose(c.Ttot, (c.debut_insp_suiv - c.debut_insp) / fs)
    medA, medL = c.amplitude.median(), c.L.median()
    c["amp_rel"] = c.amplitude / medA
    c["haut_rel"] = c.amp_rel
    c["bas_rel"] = [(medL - mes[int(a):int(b) + 1].min()) / medA for a, b in zip(c.debut_insp, c.debut_insp_suiv)]
    rms = np.array([np.sqrt(np.mean(bruit[int(a):int(b) + 1] ** 2)) for a, b in zip(c.debut_insp, c.debut_insp_suiv)])
    c["bruit_rel"] = rms / np.median(rms)
    # pentes max d'inspiration et d'expiration (rapport, sans unité)
    d = np.gradient(mes) * fs
    c["pente_insp_exp"] = [d[int(a):int(p) + 1].max() / max(-d[int(p):int(b) + 1].min(), 1e-9)
                           for a, p, b in zip(c.debut_insp, c.pic, c.debut_insp_suiv)]
    # excursions dans les deux sens
    haut = c.haut_rel > PARAMS["excursion_haut_rel"]
    bas = c.bas_rel > PARAMS["excursion_bas_rel"]
    bruite = c.bruit_rel >= PARAMS["bruit_artefact_rel"]
    lisse_ok = (c.Ti >= PARAMS["phase_min_s"]) & (c.Te >= PARAMS["phase_min_s"])
    cat = pd.Series("", index=c.index)
    cat[haut | bas] = "excursion indéterminée"
    cat[bruite & ~haut & ~bas] = "artefact probable"          # bruit > 3 Hz seul, sans grande amplitude
    cat[haut & ~bas & ~bruite & lisse_ok] = "soupir probable"
    cat[(haut | bas) & bruite] = "artefact probable"
    cat[bas & ~haut & ~bruite] = "artefact probable"
    c["excursion"] = cat
    c["excursion_sens"] = np.select([haut & bas, haut, bas, bruite], ["haut+bas", "haut", "bas", "bruit seul"], "")
    # nouvelles règles (une raison par cycle, la première rencontrée)
    lo, hi = PARAMS["ttot_bornes_s"]
    base = [
        ((c.Ti < PARAMS["phase_min_s"]) | (c.Te < PARAMS["phase_min_s"]), "phase < 0,3 s"),
        (~c.Ttot.between(lo, hi), "Ttot hors bornes"),
        (c.amp_rel < PARAMS["amp_min_rel"], "amplitude faible"),
    ]
    raison = pd.Series("", index=c.index)
    raison[cat != ""] = np.where(c.excursion_sens[cat != ""] == "bruit seul", "bruit : ", "grande excursion : ") + cat[cat != ""]
    for masque, nom in base + [(~c.Ti_Ttot.between(*PARAMS["titot_bornes"]), "Ti/Ttot hors bornes")]:
        raison[(raison == "") & masque] = nom
    c["raison_rejet"] = raison
    c["valide"] = raison == ""
    # signalement « Ttot atypique » (sans rejet), calculé hors grandes excursions
    ref = c.loc[cat == "", "Ttot"]
    med, mad = ref.median(), (ref - ref.median()).abs().median()
    c["atypique_Ttot"] = (c.Ttot - med).abs() > PARAMS["ttot_mad_signalement"] * 1.4826 * max(mad, 1e-9)
    # anciennes règles (version 2), pour comparaison seulement
    old = pd.Series("", index=c.index)
    old_regles = [(c.amp_rel > PARAMS["excursion_haut_rel"], "grande excursion")] + base + [
        (~c.Ti_Ttot.between(*PARAMS["anciennes_regles"]["titot_bornes"]), "Ti/Ttot hors bornes"),
        ((c.Ttot - c.Ttot.median()).abs() > PARAMS["anciennes_regles"]["ttot_mad_rejet"] * 1.4826
         * max((c.Ttot - c.Ttot.median()).abs().median(), 1e-9), "Ttot atypique")]
    for masque, nom in old_regles:
        old[(old == "") & masque] = nom
    c["raison_rejet_anciennes_regles"] = old
    c["valide_anciennes_regles"] = old == ""
    return c


def cycles(mes, pics, creux, fs, seuil, bruit=None) -> pd.DataFrame:
    m = montees(mes, pics, creux, fs, seuil)
    lignes = []
    for i in range(len(m) - 1):
        a, b = m.iloc[i], m.iloc[i + 1]
        if b.creux <= a.pic or (pics[(pics > a.pic) & (pics < b.pic)]).size:
            continue
        p, d0, d1 = int(a.pic), int(a.debut_insp), int(b.debut_insp)
        zone = np.where(mes[p:d1 + 1] <= b.L + seuil * a.A)[0]
        lignes.append({"debut_insp": d0, "pic": p, "fin_exp": p + int(zone[0]) if len(zone) else d1,
                       "debut_insp_suiv": d1, "decollage": int(a.decollage), "amplitude": a.A, "L": a.L})
    c = pd.DataFrame(lignes)
    if c.empty:
        return c
    return regles_et_indicateurs(c, mes, np.zeros_like(mes) if bruit is None else bruit, fs)


# ── Caractéristiques agrégées d'une séance ───────────────────────────────────
MESURES = ["Ti", "Te", "Ttot", "Ti_Ttot", "Tpause"]


def cycle_moyen(y, c, n=100):
    formes = []
    for _, k in c[c.valide].iterrows():
        seg = y[int(k.debut_insp):int(k.debut_insp_suiv) + 1]
        if len(seg) >= 10:
            seg = np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg)
            formes.append((seg - seg.min()) / np.ptp(seg))
    return np.mean(formes, axis=0) if formes else np.full(n, np.nan)


def signature(r: dict) -> dict:
    """Une ligne par séance. Calculée sur cette seule séance (aucune information externe)."""
    c = r["cyc"]
    v = c[c.valide] if not c.empty else c
    sig = {"n_cycles_detectes": len(c), "n_cycles_valides": len(v),
           "taux_rejet": 1 - len(v) / len(c) if len(c) else np.nan,
           "n_soupirs_probables": int((c.excursion == "soupir probable").sum()) if not c.empty else 0,
           "n_artefacts_probables": int((c.excursion == "artefact probable").sum()) if not c.empty else 0,
           "n_excursions_indeterminees": int((c.excursion == "excursion indéterminée").sum()) if not c.empty else 0,
           "n_atypiques_Ttot": int(c.atypique_Ttot.sum()) if not c.empty else 0,
           "signature_fiable": len(v) >= PARAMS["cycles_min_signature"]}
    for m in MESURES + ["Te_actif", "pente_insp_exp"]:
        x = v[m] if len(v) else pd.Series(dtype=float)
        sig[f"{m}_med"] = x.median() if len(x) else np.nan
        sig[f"{m}_iqr"] = x.quantile(.75) - x.quantile(.25) if len(x) else np.nan
    sig["Ttot_cv"] = v.Ttot.std() / v.Ttot.mean() if len(v) > 1 else np.nan
    sig["amp_cv"] = v.amplitude.std() / v.amplitude.mean() if len(v) > 1 else np.nan
    # régularité d'un cycle au suivant : autocorrélation de Ttot (cycles valides consécutifs)
    s = c.Ttot.where(c.valide) if not c.empty else pd.Series(dtype=float)
    sig["Ttot_autocorr1"] = s.autocorr(1) if s.notna().sum() > 3 else np.nan
    # morphologie (équivalent d'ASTER) : harmoniques de la forme moyenne normalisée
    f = cycle_moyen(r["mes"], c) if not c.empty else np.full(100, np.nan)
    spec = np.fft.rfft(f - np.nanmean(f)) if np.isfinite(f).all() else np.full(51, np.nan, complex)
    for h in range(2, PARAMS["n_harmoniques"] + 1):
        sig[f"harm{h}_amp_rel"] = np.abs(spec[h]) / np.abs(spec[1]) if np.isfinite(spec[1]) else np.nan
        sig[f"harm{h}_phase_rel"] = np.angle(spec[h] / spec[1] ** h) if np.isfinite(spec[1]) else np.nan
    return sig


# ── Analyse d'une séance ─────────────────────────────────────────────────────
def analyser(s: dict) -> dict:
    r = preparer(s)
    r["pics"], r["creux"], r["echelle"] = extrema_v2(r["det"], r["mes"], r["fs"])
    r["cyc"] = cycles(r["mes"], r["pics"], r["creux"], r["fs"], PARAMS["seuil_phase"], r["bruit"])
    r["amp_ref"] = float(r["cyc"].amplitude.median()) if not r["cyc"].empty else r["echelle"]
    return r


def qualite(s: dict, r: dict) -> dict:
    return {"duree_analysee_s": round(float(r["tw"][-1]), 1), "echelle_typique": round(r["echelle"], 1),
            "ecart_bord_fin": round(r["ecart_bord_fin"], 4), "bord_ok": r["ecart_bord_fin"] < PARAMS["tolerance_bord"],
            "marqueur_fin_present": PARAMS["marqueur_fin"] in s["marqueurs"]}
# ── Figures (matplotlib importé seulement en mode validation) ────────────────
COUL = {"insp": "#4C72B0", "exp": "#DD8452", "pause": "#9A9A9A"}
COUL_EXC = {"soupir probable": "#2E7D32", "artefact probable": "#C62828", "excursion indéterminée": "#6A1B9A"}
X_MAX = 50


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def phases(ax, tw, y, c, etiquettes=True):
    for _, k in c.iterrows():
        a = .35 if k.valide else .12
        ax.axvspan(tw[int(k.debut_insp)], tw[int(k.pic)], color=COUL["insp"], alpha=a, lw=0)
        ax.axvspan(tw[int(k.pic)], tw[int(k.fin_exp)], color=COUL["exp"], alpha=a, lw=0)
        ax.axvspan(tw[int(k.fin_exp)], tw[int(k.debut_insp_suiv)], color=COUL["pause"], alpha=a, lw=0)
        ax.plot(tw[int(k.debut_insp)], y[int(k.debut_insp)], "o", color=COUL["insp"], ms=3)
        ax.plot(tw[int(k.pic)], y[int(k.pic)], "^", color="k", ms=3)
        ax.plot(tw[int(k.fin_exp)], y[int(k.fin_exp)], "s", color=COUL["exp"], ms=3)
        if k.atypique_Ttot:
            ax.plot(tw[int(k.pic)], 1.2, "*", color="goldenrod", ms=8)
        if etiquettes and not k.valide:
            coul = COUL_EXC.get(k.excursion, "darkred")
            ax.text(tw[int(k.pic)], 1.42, k.raison_rejet.replace("grande excursion : ", "").replace("bruit : ", "bruit, "), fontsize=6,
                    ha="center", va="top", color=coul, bbox=dict(fc="white", ec=coul, lw=.5, pad=1))


def figure_seance(nom, s, r, out):
    plt = _plt()
    tw, a, c = r["tw"], r["amp_ref"], r["cyc"]
    fig = plt.figure(figsize=(15, 14))
    g = fig.add_gridspec(5, 3, height_ratios=[.8, .8, 1, 1.4, 1.2])
    ax = fig.add_subplot(g[0, :])
    ax.plot(r["t_full"], r["x_full"] / a, lw=.5, color="0.4")
    ax.axvspan(s["debut"], s["fin"], color="#C8E6C9", alpha=.5, label="fenêtre analysée (yeux fermés : +3 s / -7 s)")
    for k_, v in s["marqueurs"].items():
        ax.axvline(v, ls="--", color="k", lw=.7); ax.text(v, ax.get_ylim()[1], k_, fontsize=7, va="top")
    ax.set_xlim(0, 100); ax.legend(fontsize=7, loc="lower right"); ax.set_ylabel("× amp. méd.")
    ax.set_title("A. Contexte : repos complet, signal INVERSÉ", fontsize=9, loc="left")
    ax = fig.add_subplot(g[1, :])
    ax.plot(tw, -r["xw"], lw=.5, color="0.3"); ax.set_xlim(0, X_MAX); ax.set_ylabel(f"brut ({s['unite']})")
    ax.set_title("B. Signal PREPROCESSED brut, polarité d'origine", fontsize=9, loc="left")
    ax = fig.add_subplot(g[2, :])
    ax.plot(tw, r["det"] / a, lw=1, color="C0")
    ax.plot(tw[r["pics"]], r["det"][r["pics"]] / a, "r^", ms=5, label="fin d'inspiration")
    ax.plot(tw[r["creux"]], r["det"][r["creux"]] / a, "gv", ms=5, label="fin d'expiration")
    ax.set_xlim(0, X_MAX); ax.set_ylim(-1.5, 1.5); ax.set_ylabel("× amp. méd."); ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_title(f"C. Filtré {PARAMS['bande_detection_hz']} Hz, inversé, extrema", fontsize=9, loc="left")
    ax = fig.add_subplot(g[3, :])
    y = r["mes"] / a
    ax.plot(tw, y, lw=1, color="k"); phases(ax, tw, y, c)
    ax.set_xlim(0, X_MAX); ax.set_ylim(-1.5, 1.5); ax.set_ylabel("× amp. méd."); ax.set_xlabel("temps (s)")
    ax.set_title("D. Seuil 10 % : bleu = Ti, orange = expiration active, gris = pause ; pâle = rejeté (raison) ; "
                 "vert = soupir probable, rouge = artefact probable, violet = indéterminé ; ★ = Ttot atypique (signalé, conservé)",
                 fontsize=8, loc="left")
    ax = fig.add_subplot(g[4, 0])
    f = cycle_moyen(r["mes"], c) if not c.empty else np.full(100, np.nan)
    ax.plot(np.linspace(0, 1, len(f)), f, color="C0"); ax.set_ylim(-.05, 1.05)
    ax.set_title("Cycle moyen (valides)", fontsize=9)
    ax = fig.add_subplot(g[4, 1])
    if not c.empty:
        ax.hist([c.loc[c.valide, "Ttot"], c.loc[~c.valide, "Ttot"]], bins=12, stacked=True,
                color=["C0", "0.75"], label=["valides", "rejetés"]); ax.legend(fontsize=7)
    ax.set_title("Ttot des cycles (s)", fontsize=9)
    ax = fig.add_subplot(g[4, 2]); ax.axis("off")
    sig = signature(r)
    rej = c.raison_rejet[~c.valide].value_counts().to_dict() if not c.empty else {}
    va = int(c.valide_anciennes_regles.sum()) if not c.empty else 0
    txt = (f"Cycles détectés : {sig['n_cycles_detectes']} | valides : {sig['n_cycles_valides']} "
           f"(anciennes règles : {va})\nRejets : {rej if rej else 'aucun'}\n"
           f"Ttot atypiques signalés : {sig['n_atypiques_Ttot']}\n\n"
           f"Ti {sig['Ti_med']:.2f} s | Te {sig['Te_med']:.2f} s | Ttot {sig['Ttot_med']:.2f} s\n"
           f"Ti/Ttot {sig['Ti_Ttot_med']:.2f} | Tpause {sig['Tpause_med']:.2f} s | CV Ttot {sig['Ttot_cv']:.2f}\n\n"
           f"Erreur de filtrage fin de fenêtre : {100 * r['ecart_bord_fin']:.1f} %")
    ax.text(0, 1, txt, va="top", fontsize=8, family="monospace")
    fig.suptitle(f"{nom} — repos yeux fermés, polarité -1, seuil 10 %", fontsize=12)
    fig.tight_layout(); fig.savefig(out / f"seance_{nom}.png", dpi=90); plt.close(fig)


def figure_synthese(res, out):
    plt = _plt()
    n = len(res)
    fig = plt.figure(figsize=(16, 2 + 1.1 * n))
    g = fig.add_gridspec(n, 2, width_ratios=[4, 1])
    for i, (nom, r) in enumerate(res.items()):
        ax = fig.add_subplot(g[i, 0])
        y = r["mes"] / r["amp_ref"]
        ax.plot(r["tw"], y, lw=.8, color="k"); phases(ax, r["tw"], y, r["cyc"], etiquettes=False)
        for _, k in r["cyc"][r["cyc"].excursion != ""].iterrows():
            ax.axvspan(r["tw"][int(k.debut_insp)], r["tw"][int(k.debut_insp_suiv)], color=COUL_EXC[k.excursion], alpha=.25, lw=0)
        ax.set_xlim(0, X_MAX); ax.set_ylim(-1.5, 1.5); ax.set_yticks([])
        ax.set_ylabel(nom, rotation=0, ha="right", va="center", fontsize=8)
        c = r["cyc"]
        ax.text(X_MAX + .3, 0, f"{int(c.valide.sum())}/{len(c)} valides\n(avant : {int(c.valide_anciennes_regles.sum())})",
                fontsize=7, va="center")
        if i == 0:
            ax.set_title("Repos yeux fermés, polarité -1, seuil 10 % (pâle = rejeté ; vert/rouge/violet = soupir / artefact / indéterminé)", fontsize=9)
        if i < n - 1:
            ax.set_xticks([])
        ax2 = fig.add_subplot(g[i, 1]); ax2.plot(cycle_moyen(r["mes"], c), color="C0"); ax2.set_xticks([]); ax2.set_yticks([])
        if i == 0:
            ax2.set_title("cycle moyen", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "synthese_segmentation.png", dpi=80); plt.close(fig)


def planche_controle(res, out, max_vignettes=48):
    """Vignettes : cycles récupérés par les nouvelles règles, excursions, cycles atypiques."""
    plt = _plt()
    items = []
    for nom, r in res.items():
        c = r["cyc"]
        for idx, k in c.iterrows():
            motif = ("récupéré" if k.valide and not k.valide_anciennes_regles else
                     k.excursion if k.excursion else "Ttot atypique" if k.atypique_Ttot else "")
            if motif:
                prio = 0 if nom.split("_")[0] in ("P107", "P173", "P374") else (1 if k.excursion else 2)
                items.append((prio, nom, idx, motif))
    items = sorted(items)[:max_vignettes]
    if not items:
        return 0
    ncol = 6
    nrow = int(np.ceil(len(items) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(18, 2.3 * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (_, nom, idx, motif) in zip(axs.ravel(), items):
        r = res[nom]; k = r["cyc"].loc[idx]; fs = r["fs"]
        a, b = max(int(k.debut_insp - 2 * fs), 0), min(int(k.debut_insp_suiv + 2 * fs), len(r["mes"]) - 1)
        tw, y = r["tw"], r["mes"] / r["amp_ref"]
        ax.axis("on"); ax.plot(tw[a:b], y[a:b], color="k", lw=.8)
        phases(ax, tw, y, r["cyc"].loc[[idx]], etiquettes=False)
        ax.set_xlim(tw[a], tw[b]); ax.set_ylim(-2, 4); ax.set_yticks([]); ax.tick_params(labelsize=6)
        coul = COUL_EXC.get(motif, "C0" if motif == "récupéré" else "goldenrod")
        ax.set_title(f"{nom} · {motif}\nTtot {k.Ttot:.1f}s · Ti/Ttot {k.Ti_Ttot:.2f} · amp {k.amp_rel:.1f}× · "
                     f"{'VALIDE' if k.valide else 'rejeté'}", fontsize=7, color=coul)
    fig.suptitle("Planche de contrôle : cycles récupérés (bleu), soupirs (vert), artefacts (rouge), "
                 "indéterminés (violet), Ttot atypiques (or) — échelle commune en × amplitude médiane", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "planche_controle.png", dpi=80); plt.close(fig)
    return len(items)


# ── Sensibilité ──────────────────────────────────────────────────────────────
def sensibilite(res, out):
    plt = _plt()
    lignes = []
    for nom, r in res.items():
        for s_ in PARAMS["seuils_sensibilite"]:
            c = cycles(r["mes"], r["pics"], r["creux"], r["fs"], s_, r["bruit"])
            v = c[c.valide] if not c.empty else c
            lignes.append({"seance": nom, "seuil": s_, "n_valides": len(v),
                           **{f"{m}_med": (v[m].median() if len(v) else np.nan) for m in MESURES}})
    sens = pd.DataFrame(lignes); sens.round(3).to_csv(out / "sensibilite_seuils.csv", index=False)
    ref = sens[sens.seuil == 0.10].set_index("seance")
    resume = []
    for m in MESURES:
        for s_ in PARAMS["seuils_sensibilite"]:
            cur = sens[sens.seuil == s_].set_index("seance")
            resume.append({"mesure": m, "seuil": s_, "mediane_toutes_seances": cur[f"{m}_med"].median(),
                           "variation_mediane_vs_10pct": (cur[f"{m}_med"] / ref[f"{m}_med"] - 1).median(),
                           "spearman_vs_10pct": cur[f"{m}_med"].corr(ref[f"{m}_med"], method="spearman"),
                           "n_valides_median": cur.n_valides.median()})
    resume = pd.DataFrame(resume); resume.round(3).to_csv(out / "sensibilite_resume.csv", index=False)
    fig, axs = plt.subplots(1, len(MESURES), figsize=(18, 4))
    for ax, m in zip(axs, MESURES):
        for _, grp in sens.groupby("seance"):
            ax.plot(100 * grp.seuil, grp[f"{m}_med"], color="0.6", lw=.8, marker="o", ms=2)
        med = sens.groupby("seuil")[f"{m}_med"].median()
        ax.plot(100 * med.index, med.values, color="C3", lw=2.5)
        ax.set_title(m, fontsize=9); ax.set_xticks([5, 10, 15, 20]); ax.set_xlabel("seuil (%)")
    fig.tight_layout(); fig.savefig(out / "sensibilite_seuils.png", dpi=90); plt.close(fig)
    return resume


# ── Rapport texte (le fichier à envoyer) ─────────────────────────────────────
def rapport(res, bilan, resume, n_vignettes, out):
    L = []
    tous = pd.concat([r["cyc"].assign(seance=n) for n, r in res.items()], ignore_index=True)
    nv, nva = int(tous.valide.sum()), int(tous.valide_anciennes_regles.sum())
    L.append("RAPPORT DE VALIDATION — cycles respiratoires, repos yeux fermés, " + f"{len(res)} séances")
    L.append(f"Paramètres : polarité -1, seuil 10 %, Ti/Ttot [0,1 ; 0,7], Ttot atypique signalé sans rejet, "
             f"exclusion +3 s / -7 s\n")
    L.append(f"1. CYCLES : {len(tous)} détectés | valides {nva} (anciennes règles) -> {nv} (nouvelles) : "
             f"+{nv - nva} cycles récupérés ; taux de rejet {100 * (1 - nva / len(tous)):.0f} % -> {100 * (1 - nv / len(tous)):.0f} %")
    L.append("   Rejets (nouvelles règles) : " + str(tous.raison_rejet[~tous.valide].value_counts().to_dict()))
    L.append("   Ttot atypiques signalés (conservés) : " + str(int(tous.atypique_Ttot.sum())))
    L.append("   Excursions : " + str(tous.excursion[tous.excursion != ""].value_counts().to_dict())
             + " | sens : " + str(tous.excursion_sens[tous.excursion_sens != ""].value_counts().to_dict()))
    L.append("\n2. SÉANCES SURVEILLÉES (P107, P173, P374) :")
    for nom in bilan.index:
        if nom.split("_")[0] in ("P107", "P173", "P374"):
            b = bilan.loc[nom]
            L.append(f"   {nom} : valides {b.valides_avant} -> {b.valides_apres} sur {b.n_detectes} "
                     f"| rejets restants : {b.rejets_apres}")
    L.append("\n3. PAR SÉANCE (valides avant -> après / détectés, taux de rejet, excursions, atypiques) :")
    for nom, b in bilan.iterrows():
        L.append(f"   {nom:12s} {b.valides_avant:3d} -> {b.valides_apres:3d} / {b.n_detectes:3d}  "
                 f"rejet {100 * b.taux_rejet_apres:4.0f} %  soupirs {b.n_soupirs} artefacts {b.n_artefacts} "
                 f"indét. {b.n_indetermines}  atypiques {b.n_atypiques}  {'' if b.fiable else '<5 VALIDES'}")
    L.append(f"   Médiane des cycles valides par séance : {bilan.valides_avant.median():.0f} -> {bilan.valides_apres.median():.0f} ; "
             f"séances avec < 5 cycles valides : {(bilan.valides_avant < 5).sum()} -> {(bilan.valides_apres < 5).sum()}")
    L.append("\n4. CARACTÉRISTIQUES AGRÉGÉES (médiane des 20 séances ; variation médiane par séance, nouvelles vs anciennes règles) :")
    for m in MESURES:
        a, n = bilan[f"{m}_avant"], bilan[f"{m}_apres"]
        var = ((n / a) - 1).median()
        L.append(f"   {m:8s} {a.median():6.3f} -> {n.median():6.3f}   variation médiane {100 * var:+5.1f} %   "
                 f"variation max {100 * ((n / a) - 1).abs().max():5.1f} % ({((n / a) - 1).abs().idxmax()})")
    L.append("\n5. CYCLES À INSPECTER (planche_controle.png, " + str(n_vignettes) + " vignettes) :")
    recup = tous[tous.valide & ~tous.valide_anciennes_regles]
    L.append(f"   récupérés : {len(recup)} | Ti/Ttot des récupérés : min {recup.Ti_Ttot.min():.2f}, "
             f"médiane {recup.Ti_Ttot.median():.2f} ; Ttot min {recup.Ttot.min():.1f} s, max {recup.Ttot.max():.1f} s"
             if len(recup) else "   aucun cycle récupéré")
    susp = tous[tous.valide & ((tous.bruit_rel >= 2) | (tous.amp_rel > 2.5) | (tous.bas_rel > .8))]
    L.append(f"   cycles VALIDES à la limite d'une excursion (bruit >= 2x, amp > 2,5x ou descente > 0,8x) : {len(susp)}")
    for _, k in susp.iterrows():
        L.append(f"      {k.seance} t={k.t_debut_s:.1f}s amp {k.amp_rel:.1f}x bas {k.bas_rel:.1f}x bruit {k.bruit_rel:.1f}x")
    L.append("\n6. SENSIBILITÉ (médiane des séances | Spearman vs 10 %) :")
    L.append(resume.pivot(index="mesure", columns="seuil", values="mediane_toutes_seances").round(3).to_string())
    L.append(resume.pivot(index="mesure", columns="seuil", values="spearman_vs_10pct").round(2).to_string())
    L.append(f"\n7. BORD DE FIN : erreur max {100 * bilan.ecart_bord_fin.max():.1f} % "
             f"({'OK' if bilan.ecart_bord_fin.max() < PARAMS['tolerance_bord'] else 'À VÉRIFIER'})")
    (out / "rapport_validation.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def bilan_seance(r) -> dict:
    c = r["cyc"]; sig = signature(r)
    old = c[c.valide_anciennes_regles]; new = c[c.valide]
    return {"n_detectes": len(c), "valides_avant": len(old), "valides_apres": len(new),
            "taux_rejet_apres": sig["taux_rejet"], "n_soupirs": sig["n_soupirs_probables"],
            "n_artefacts": sig["n_artefacts_probables"], "n_indetermines": sig["n_excursions_indeterminees"],
            "n_atypiques": sig["n_atypiques_Ttot"], "fiable": sig["signature_fiable"],
            "rejets_apres": c.raison_rejet[~c.valide].value_counts().to_dict(), "ecart_bord_fin": r["ecart_bord_fin"],
            **{f"{m}_avant": old[m].median() for m in MESURES}, **{f"{m}_apres": new[m].median() for m in MESURES}}


def valider(data: Path, sessions: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    res, bilan = {}, {}
    for _, row in pd.read_csv(sessions).iterrows():
        nom = f"{row.participant}_jour{int(row.day)}"
        try:
            s = lire_seance(data, row.participant, int(row.day)); r = analyser(s)
        except Exception as e:  # noqa: BLE001
            print(f"{nom}: ÉCHEC ({e})"); continue
        res[nom] = r; bilan[nom] = bilan_seance(r)
        r["cyc"].round(4).to_csv(out / f"cycles_{nom}.csv", index=False)
        figure_seance(nom, s, r, out)
    bilan = pd.DataFrame(bilan).T.infer_objects()
    bilan.to_csv(out / "bilan_validation.csv")
    pd.Series({k: str(v) for k, v in PARAMS.items()}).to_csv(out / "parametres.csv", header=["valeur"])
    figure_synthese(res, out)
    n_v = planche_controle(res, out)
    resume = sensibilite(res, out)
    rapport(res, bilan, resume, n_v, out)


# ── Extraction (218 participants) ────────────────────────────────────────────
def _une_seance(args):
    data, participant, jour, rang = args
    s = lire_seance(Path(data), participant, jour)
    r = analyser(s)
    cle = {"participant": participant, "jour": jour, "rang_seance": rang, "condition": PARAMS["condition"]}
    c = r["cyc"].assign(**cle)
    return cle | signature(r), cle | qualite(s, r), c


def lister_seances(data: Path) -> pd.DataFrame:
    """Séances de repos présentes, gardées seulement pour les participants qui en ont >= 2."""
    rows = []
    for f in sorted((data / PARAMS["source"]).glob("P*/P*_day-*_task-resting-state_multimodal.bdf")):
        p, d = f.name.split("_")[:2]
        rows.append({"participant": p, "jour": int(d.removeprefix("day-"))})
    s = pd.DataFrame(rows)
    s = s[s.groupby("participant").jour.transform("nunique") >= 2].copy()
    s["rang_seance"] = s.groupby("participant").jour.rank(method="dense").astype(int)   # 1 = enrôlement, 2 = test
    return s.sort_values(["participant", "jour"])


def extraire_tout(data_dir, out_dir, workers=16):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seances = lister_seances(data)
    print(f"{seances.participant.nunique()} participants, {len(seances)} séances de repos", flush=True)
    sigs, quals, cycs, echecs = [], [], [], []
    args = [(str(data), r.participant, int(r.jour), int(r.rang_seance)) for r in seances.itertuples()]
    with ProcessPoolExecutor(workers) as pool:
        for a, fut in zip(args, [pool.submit(_une_seance, a) for a in args]):
            try:
                sg, q, c = fut.result(); sigs.append(sg); quals.append(q); cycs.append(c)
            except Exception as e:  # noqa: BLE001
                echecs.append({"participant": a[1], "jour": a[2], "erreur": repr(e)})
                print(f"ÉCHEC {a[1]} jour {a[2]} : {e}", flush=True)
    pd.DataFrame(sigs).to_csv(out / "signatures_seances.csv", index=False)
    pd.DataFrame(quals).to_csv(out / "qualite_seances.csv", index=False)
    if cycs:
        pd.concat(cycs, ignore_index=True).round(4).to_csv(out / "cycles_tous.csv", index=False)
    pd.DataFrame(echecs, columns=["participant", "jour", "erreur"]).to_csv(out / "echecs.csv", index=False)
    pd.Series({k: str(v) for k, v in PARAMS.items()}).to_csv(out / "parametres.csv", header=["valeur"])
    print(f"{len(sigs)} séances extraites, {len(echecs)} échecs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validation", "extraction"], default="validation")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--sessions", default="sessions_polarite.csv")
    ap.add_argument("--out", default="../data/validation_v3")
    a = ap.parse_args()
    if a.mode == "validation":
        valider(Path(a.data), Path(a.sessions), Path(a.out))
    else:
        extraire_tout(a.data, a.out)


if __name__ == "__main__":
    main()
