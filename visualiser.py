"""
visualiser.py — Figures pour comprendre l'identification jour 1 -> jour 2 au repos.

Entrée  : features_repos.csv (job HABS n°1, une ligne par fenêtre de 15 s)
Sortie  : un dossier d'images PNG + resume.txt

Usage (depuis le dossier habs-EEG) :
    python visualiser.py
    python visualiser.py --csv ../data/features_repos.csv --out figures

Figures produites :
    1_tsne_personnes.png            chaque fenêtre = un point, couleur = personne, forme = séance
    2_matrices_ressemblance.png     ressemblance entre profils : même jour vs jour suivant
    3_distributions_ressemblance.png  ressemblance « même personne » vs « autres personnes »
    4_stabilite_eeg_cerveau.png     stabilité de chaque électrode / bande (cartes du crâne)
    5_stabilite_autres_signaux.png  stabilité des mesures cœur, respiration, peau, yeux
    6_scores.png                    scores d'identification vs hasard
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on enregistre directement les images
import matplotlib.pyplot as plt

PREFIXES = ("eeg_", "ecg_", "resp_", "gsr_", "eye_")
EXCLUDE = {"eye_valid_samples"}  # reflète la cadence des lunettes, pas la personne
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
JUNK_TOKENS = {"log", "pow", "power", "logpow", "bp", "abs", "rel", "psd"}
SEED = 0
SIGNAL_NAMES = {"ecg_": "Cœur", "resp_": "Respiration", "gsr_": "Peau", "eye_": "Yeux"}


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def find_col(df, candidates, what):
    """Trouve la colonne participant / jour, quel que soit son nom exact."""
    meta = [c for c in df.columns if not c.lower().startswith(PREFIXES)]
    lower = {c.lower(): c for c in meta}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in meta:
        if any(cand in c.lower() for cand in candidates):
            return c
    sys.exit(f"\n[ERREUR] Colonne « {what} » introuvable.\n"
             f"Colonnes non-features du fichier : {meta}\n"
             f"Envoie cette ligne à Claude.")


def day_key(v):
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else 0


def load(csv_path):
    df = pd.read_csv(csv_path)
    pid = find_col(df, ["participant_id", "participant", "subject_id", "subject"], "participant")
    day = find_col(df, ["day", "jour"], "jour")
    feats = [c for c in df.columns
             if c.lower().startswith(PREFIXES) and c not in EXCLUDE
             and pd.api.types.is_numeric_dtype(df[c])]
    df[feats] = df[feats].fillna(df[feats].median())
    df = df.copy()
    df["_day"] = df[day].map(day_key)
    df["_win"] = df.groupby([pid, "_day"]).cumcount()

    # 1re et 2e séance de chaque personne qui en a au moins 2
    days = df.groupby(pid)["_day"].apply(lambda s: sorted(s.unique()))
    multi = days[days.map(len) >= 2]
    d = df[df[pid].isin(multi.index)].copy()
    first = d[pid].map(multi.map(lambda x: x[0]))
    second = d[pid].map(multi.map(lambda x: x[1]))
    d["_sess"] = np.where(d["_day"] == first, 1, np.where(d["_day"] == second, 2, 0))
    d = d[d["_sess"] > 0]

    print(f"Fichier          : {csv_path}")
    print(f"Colonnes clés    : participant = '{pid}', jour = '{day}'")
    print(f"Features         : {len(feats)} (dont {sum(c.startswith('eeg_') for c in feats)} EEG)")
    print(f"Fenêtres         : {len(df)} au total, {len(d)} utilisées (séances 1 et 2)")
    print(f"Participants     : {df[pid].nunique()} au total, {len(multi)} avec >= 2 séances\n")
    return d, pid, feats


# ----------------------------------------------------------------------------
# Profils et ressemblance
# ----------------------------------------------------------------------------
def profiles(sub, pid, feats):
    """Profil d'une séance = moyenne de ses fenêtres."""
    return sub.groupby(pid)[feats].mean()


def zscore_pair(A, B):
    both = pd.concat([A, B])
    mu, sd = both.mean(), both.std().replace(0, 1)
    return (A - mu) / sd, (B - mu) / sd


def corr_matrix(A, B):
    """C[i, j] = corrélation entre le profil i de A et le profil j de B."""
    a, b = zscore_pair(A, B)
    a, b = a.values, b.values
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return a @ b.T / a.shape[1]


def accuracy(C):
    return float(np.mean(C.argmax(1) == np.arange(len(C))))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def tsne_numpy(X, perplexity=15.0, n_iter=1000, seed=SEED):
    """t-SNE exact écrit en numpy (pas de fichier compilé : contourne Smart App Control).
    Suffisant pour quelques centaines de points."""
    n = X.shape[0]
    D = np.sum(X ** 2, 1)[:, None] + np.sum(X ** 2, 1)[None, :] - 2 * X @ X.T
    D = np.maximum(D, 0)
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        lo, hi, beta = 0.0, np.inf, 1.0
        di = np.delete(D[i], i)
        for _ in range(60):  # recherche de la largeur de noyau qui donne la perplexité voulue
            p = np.exp(-(di - di.min()) * beta)
            sp = p.sum()
            H = np.log(sp) + beta * np.sum((di - di.min()) * p) / sp
            if abs(H - target) < 1e-5:
                break
            if H > target:
                lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi, beta = beta, (beta + lo) / 2
        P[i, np.arange(n) != i] = p / sp
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    rng = np.random.default_rng(seed)
    Y = rng.normal(scale=1e-4, size=(n, 2))
    vel, gains = np.zeros_like(Y), np.ones_like(Y)
    lr = max(n / 48.0, 50.0)  # même règle que scikit-learn (learning_rate='auto')
    for it in range(n_iter):
        exag = 12.0 if it < 250 else 1.0
        mom = 0.5 if it < 250 else 0.8
        sq = np.sum(Y ** 2, 1)
        num = 1.0 / (1.0 + sq[:, None] + sq[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (exag * P - Q) * num
        grad = 4 * (np.diag(PQ.sum(1)) - PQ) @ Y
        gains = np.where(np.sign(grad) != np.sign(vel), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        vel = mom * vel - lr * gains * grad
        Y = Y + vel
        Y = Y - Y.mean(0)
    return Y


def fig_tsne(d, pid, feats, out, n_people=15):
    rng = np.random.default_rng(SEED)
    people = rng.choice(d[pid].unique(), size=min(n_people, d[pid].nunique()), replace=False)
    sub = d[d[pid].isin(people)].reset_index(drop=True)
    X = sub[feats].values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    # réduction préalable à 30 dimensions par PCA (numpy), comme le fait t-SNE en pratique
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    X = U[:, :30] * S[:30]
    emb = tsne_numpy(X, perplexity=min(15, len(sub) // 4))

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)
    for k, p in enumerate(people):
        col = palette[k % len(palette)]
        cents = []
        for s, marker in [(1, "o"), (2, "^")]:
            m = (sub[pid] == p).values & (sub["_sess"] == s).values
            ax.scatter(emb[m, 0], emb[m, 1], color=col, marker=marker, s=45,
                       edgecolor="k", linewidth=0.4)
            cents.append(emb[m].mean(0))
        ax.plot([cents[0][0], cents[1][0]], [cents[0][1], cents[1][1]],
                color=col, linewidth=1, alpha=0.7)
    ax.scatter([], [], marker="o", color="grey", label="Séance 1")
    ax.scatter([], [], marker="^", color="grey", label="Séance 2")
    ax.legend(loc="best")
    ax.set_title(f"t-SNE des fenêtres de repos ({len(people)} personnes au hasard)\n"
                 "couleur = personne, trait = déplacement entre les 2 séances")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "1_tsne_personnes.png", dpi=150)
    plt.close(fig)


def fig_matrices(C_same, C_cross, acc_same, acc_cross, out, n_show=30):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(C_cross), size=min(n_show, len(C_cross)), replace=False))
    vmin = min(C_same[np.ix_(idx, idx)].min(), C_cross[np.ix_(idx, idx)].min())
    vmax = max(C_same[np.ix_(idx, idx)].max(), C_cross[np.ix_(idx, idx)].max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, C, title, acc in [
        (axes[0], C_same, "Même jour (fenêtres 1–2 vs 3–5)", acc_same),
        (axes[1], C_cross, "Jour suivant (séance 1 vs séance 2)", acc_cross),
    ]:
        im = ax.imshow(C[np.ix_(idx, idx)], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\nreconnues : {acc:.1%} des {len(C)} personnes")
        ax.set_xlabel("Personne (2e profil)")
        ax.set_ylabel("Personne (1er profil)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Ressemblance (corrélation)")
    fig.suptitle(f"Ressemblance entre profils — {len(idx)} personnes au hasard "
                 "(diagonale claire = personne reconnue)")
    fig.savefig(out / "2_matrices_ressemblance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_distributions(C_same, C_cross, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    off = ~np.eye(len(C_cross), dtype=bool)
    bins = np.linspace(min(C_same.min(), C_cross.min()), max(C_same.max(), C_cross.max()), 60)
    for ax, C, title in [(axes[0], C_same, "Même jour"), (axes[1], C_cross, "Jour suivant")]:
        ax.hist(C[off], bins=bins, density=True, alpha=0.6, color="grey", label="Autres personnes")
        ax.hist(np.diag(C), bins=bins, density=True, alpha=0.7, color="tab:red", label="Même personne")
        ax.set_title(title)
        ax.set_xlabel("Ressemblance (corrélation)")
        ax.legend()
    axes[0].set_ylabel("Densité")
    fig.suptitle("Plus les deux courbes se chevauchent, plus l'identification est difficile")
    fig.tight_layout()
    fig.savefig(out / "3_distributions_ressemblance.png", dpi=150)
    plt.close(fig)


def feature_reliability(A, B):
    """Pour chaque feature : corrélation, entre personnes, de la valeur au 1er et au 2e profil."""
    return pd.Series({f: np.corrcoef(A[f], B[f])[0, 1] for f in A.columns}).fillna(0)


def parse_eeg(cols, known_channels):
    rows = []
    for c in cols:
        toks = c[4:].split("_")
        band = next((t.lower() for t in toks if t.lower() in BANDS), None)
        if known_channels is not None:
            chan = next((known_channels[t.lower()] for t in toks if t.lower() in known_channels), None)
        else:
            chan = next((t for t in toks if t.lower() not in BANDS and t.lower() not in JUNK_TOKENS), None)
        if band and chan:
            rows.append((c, chan, band))
    return pd.DataFrame(rows, columns=["col", "chan", "band"])


def fig_eeg_stability(r_same, r_cross, out):
    try:
        import mne
        mne.set_log_level("ERROR")
        name = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
        montage = mne.channels.make_standard_montage(name)
        known = {n.lower(): n for n in montage.ch_names}
    except Exception:
        mne, montage, known = None, None, None

    eeg = [c for c in r_cross.index if c.startswith("eeg_")]
    tab = parse_eeg(eeg, known)
    if tab.empty:
        print("[ATTENTION] Noms de colonnes EEG non reconnus, figure 4 ignorée. Exemple :", eeg[:3])
        return None
    tab["same"] = tab["col"].map(r_same)
    tab["cross"] = tab["col"].map(r_cross)
    bands = [b for b in BANDS if b in set(tab["band"])]

    if mne is not None:
        fig, axes = plt.subplots(2, len(bands), figsize=(3.2 * len(bands), 6.8))
        axes = np.atleast_2d(axes)
        for j, band in enumerate(bands):
            tb = tab[tab["band"] == band]
            info = mne.create_info(list(tb["chan"]), 1000.0, "eeg")
            info.set_montage(montage)
            for i, (key, label) in enumerate([("same", "Même jour"), ("cross", "Jour suivant")]):
                ax = axes[i, j]
                kw = dict(axes=ax, show=False, cmap="viridis", contours=0)
                try:
                    im, _ = mne.viz.plot_topomap(tb[key].values, info, vlim=(0, 1), **kw)
                except TypeError:  # anciennes versions de mne
                    im, _ = mne.viz.plot_topomap(tb[key].values, info, vmin=0, vmax=1, **kw)
                if i == 0:
                    ax.set_title(band.capitalize(), fontsize=13)
                if j == 0:
                    ax.text(-0.25, 0.5, label, transform=ax.transAxes, rotation=90,
                            va="center", ha="center", fontsize=12)
        fig.colorbar(im, ax=axes, shrink=0.7, label="Stabilité (corrélation entre les 2 profils)")
        fig.suptitle("Stabilité de la puissance EEG par électrode et par bande\n"
                     "jaune = la mesure garde la même valeur relative chez une personne")
    else:
        piv = tab.pivot_table(index="chan", columns="band", values="cross")[bands]
        fig, ax = plt.subplots(figsize=(5, 14))
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(bands)), bands)
        ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
        fig.colorbar(im, label="Stabilité jour suivant")
        ax.set_title("Stabilité EEG (mne absent : tableau)")
    fig.savefig(out / "4_stabilite_eeg_cerveau.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tab.groupby("band")[["same", "cross"]].mean().loc[bands]


def fig_other_stability(r_same, r_cross, out):
    other = [c for c in r_cross.index if not c.startswith("eeg_")]
    if not other:
        return
    order = sorted(other, key=lambda c: (c.split("_")[0], -r_cross[c]))
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.5))
    ax.barh(y - 0.2, [r_same[c] for c in order], height=0.4, color="lightgrey", label="Même jour")
    ax.barh(y + 0.2, [r_cross[c] for c in order], height=0.4, color="tab:blue", label="Jour suivant")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Stabilité (corrélation entre les 2 profils)")
    ax.set_title("Stabilité des mesures non-EEG")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "5_stabilite_autres_signaux.png", dpi=150)
    plt.close(fig)


def fig_scores(acc_corr, n_people, out):
    # Scores par séance (top-1) de train.py, régression logistique — voir resultats_premier_modele.md
    labels = ["Tous signaux", "EEG seul", "Tout sauf EEG", "Respiration", "Yeux", "Cœur", "Peau"]
    values = [31.2, 27.5, 17.0, 3.2, 3.2, 2.3, 2.3]
    labels.insert(0, "Corrélation\n(sans entraînement)")
    values.insert(0, 100 * acc_corr)
    colors = ["tab:orange"] + ["tab:blue"] * (len(values) - 1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(100 / n_people, color="red", linestyle="--", label=f"Hasard ({100 / n_people:.2f} %)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center")
    ax.set_ylabel("Bonne personne retrouvée (%)")
    ax.set_title(f"Identification jour 1 → jour 2 au repos ({n_people} personnes)\n"
                 "bleu = régression logistique (train.py), orange = simple corrélation de profils")
    ax.legend()
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "6_scores.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def safe(step, fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        print(f"[ATTENTION] {step} a échoué : {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../data/features_repos.csv")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    csv_path, out = Path(args.csv), Path(args.out)
    if not csv_path.exists():
        sys.exit(f"[ERREUR] Fichier introuvable : {csv_path.resolve()}\n"
                 "Indique le bon chemin avec --csv")
    out.mkdir(parents=True, exist_ok=True)

    d, pid, feats = load(csv_path)
    s1, s2 = d[d["_sess"] == 1], d[d["_sess"] == 2]
    P1 = profiles(s1, pid, feats)
    P2 = profiles(s2, pid, feats).reindex(P1.index)
    A = profiles(s1[s1["_win"] < 2], pid, feats).reindex(P1.index)
    B = profiles(s1[s1["_win"] >= 2], pid, feats).reindex(P1.index)
    keep = P2.notna().all(axis=1) & A.notna().all(axis=1) & B.notna().all(axis=1)
    P1, P2, A, B = P1[keep], P2[keep], A[keep], B[keep]
    n = len(P1)

    print("[1/6] t-SNE (peut prendre ~1 min)...")
    safe("t-SNE", fig_tsne, d, pid, feats, out)

    print("[2/6] Matrices de ressemblance...")
    C_same, C_cross = corr_matrix(A, B), corr_matrix(P1, P2)
    acc_same, acc_cross = accuracy(C_same), accuracy(C_cross)
    eeg = [f for f in feats if f.startswith("eeg_")]
    acc_cross_eeg = accuracy(corr_matrix(P1[eeg], P2[eeg])) if eeg else float("nan")
    safe("Matrices", fig_matrices, C_same, C_cross, acc_same, acc_cross, out)

    print("[3/6] Distributions...")
    safe("Distributions", fig_distributions, C_same, C_cross, out)

    print("[4/6] Stabilité EEG (cartes du crâne)...")
    r_same, r_cross = feature_reliability(A, B), feature_reliability(P1, P2)
    by_band = safe("Stabilité EEG", fig_eeg_stability, r_same, r_cross, out)

    print("[5/6] Stabilité des autres signaux...")
    safe("Stabilité autres signaux", fig_other_stability, r_same, r_cross, out)

    print("[6/6] Scores...")
    safe("Scores", fig_scores, acc_cross, n, out)

    lines = [
        f"Personnes avec 2 séances : {n} (hasard = {100 / n:.2f} %)",
        "",
        "Identification par simple corrélation de profils (aucun entraînement) :",
        f"  même jour      : {acc_same:.1%}",
        f"  jour suivant   : {acc_cross:.1%}  (tous signaux)",
        f"  jour suivant   : {acc_cross_eeg:.1%}  (EEG seul)",
        "",
        "Ressemblance moyenne même personne / autres personnes :",
        f"  même jour      : {np.diag(C_same).mean():.2f} / {C_same[~np.eye(n, dtype=bool)].mean():.2f}",
        f"  jour suivant   : {np.diag(C_cross).mean():.2f} / {C_cross[~np.eye(n, dtype=bool)].mean():.2f}",
    ]
    if by_band is not None:
        lines += ["", "Stabilité EEG moyenne par bande (même jour / jour suivant) :"]
        lines += [f"  {b:6s}: {r.same:.2f} / {r.cross:.2f}" for b, r in by_band.iterrows()]
    lines += ["", "15 features les plus stables d'un jour à l'autre :"]
    lines += [f"  {f:35s} {v:.2f}" for f, v in r_cross.sort_values(ascending=False).head(15).items()]
    lines += ["", "10 features les moins stables d'un jour à l'autre :"]
    lines += [f"  {f:35s} {v:.2f}" for f, v in r_cross.sort_values().head(10).items()]
    text = "\n".join(lines)
    (out / "resume.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nTerminé. Images dans : {out.resolve()}")


if __name__ == "__main__":
    main()
