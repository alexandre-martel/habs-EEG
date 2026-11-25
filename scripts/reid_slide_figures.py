"""Figures for the slides (``results/figures/slides/*.png``), on real data.

    python scripts/reid_open_set.py --real --npz cache/slides/open_set.npz
    python scripts/reid_slide_figures.py [--only 02_lda,08_absent]

Heavy steps are computed once and kept in ``cache/slides/`` (not committed):
the EEG LDA of fold 0 and a beat-shape encoder trained on CPU on fold 0 (same
code and settings as the final members, fewer steps). Everything else reads
the committed score dumps. People are shown as "A", "B", "C" (drawn at random
among the test people of fold 0, seeded): no participant id on any figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from reid.config import CACHE, RESULTS, SEED
from reid.final.fusion import znorm
from scripts.reid_offline_fusion import parts, test_summary
from scripts.reid_open_set import SLOT_NAMES, collect, final_slots

OUT = RESULTS / "figures" / "slides"
WORK = CACHE / "slides"

# reference palette (light mode): categorical slots 1-3, ink, recessive greys
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, CONTEXT = "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c2"
PEOPLE = (BLUE, ORANGE, AQUA)
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "figure.facecolor": "white",
    }
)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"-> {OUT / name}.png", flush=True)


def fold0():
    from reid.eval.protocol import FeatureStore, build_split, load_folds
    from reid.final.raw import RAW
    from reid.inventory import load_inventory

    inv, folds = load_inventory(), load_folds()
    feats = build_split(inv, folds[0], "d1_to_d2", store=FeatureStore())
    raw = build_split(inv, folds[0], "d1_to_d2", store=FeatureStore(RAW))
    return feats, raw


def clean_beats(r: dict) -> np.ndarray:
    """Beats of a session, each z-scored, ordered from the most typical (closest
    to the median shape) to the least: a few beats with drift or noise would
    otherwise dominate an average."""
    beats = np.vstack([r[k] for k in r if k.startswith("beats_")]).astype(float)
    beats = beats[np.isfinite(beats).all(1)]
    beats = (beats - beats.mean(1, keepdims=True)) / np.maximum(
        beats.std(1, keepdims=True), 1e-9
    )
    med = np.median(beats, 0)
    return beats[np.argsort(((beats - med) ** 2).mean(1))]


def example_people(split, n: int = 3) -> list[str]:
    pids = sorted(set(split.gallery.pids) & set(split.probe.pids))
    return list(np.random.default_rng(SEED).choice(pids, n, replace=False))


# -- 01: the signals of three people on both days ------------------------------
def fig_signals(split_f, split_r) -> None:
    from reid.final.raw import RESP_RATE

    people = example_people(split_f)
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.5), sharex="col")
    t_beat = np.linspace(-250, 450, 128)
    for row, (pid, color) in enumerate(zip(people, PEOPLE)):
        label = "ABC"[row]
        for day, style in ((1, "-"), (2, "--")):
            f = split_f.gallery.store.load(pid, day, "rest")
            r = split_r.gallery.store.load(pid, day, "rest")
            ec = f["w_condition"] == "rest_ec"
            psd = f["w_eeg_psd"][ec] if ec.any() else f["w_eeg_psd"]
            keep = (f["freqs"] >= 1) & (f["freqs"] <= 40)
            rel = psd[..., keep] / psd[..., keep].sum(-1, keepdims=True)
            spec = np.log10(np.nanmean(rel, axis=(0, 1)))
            axes[row, 0].plot(f["freqs"][keep], spec, style, color=color)
            beat = np.median(clean_beats(r), 0)
            axes[row, 1].plot(t_beat, beat, style, color=color, label=f"jour {day}")
            resp = np.asarray(r["resp_0"], float)[: 30 * RESP_RATE]
            resp = (resp - np.nanmean(resp)) / np.nanstd(resp)
            off = 0 if day == 1 else -4.5
            axes[row, 2].plot(
                np.arange(len(resp)) / RESP_RATE, resp + off, style, color=color
            )
        axes[row, 0].set_ylabel(f"Personne {label}", color=INK, fontweight="bold")
        axes[row, 2].set_yticks([0, -4.5], ["jour 1", "jour 2"])
        axes[row, 1].set_yticks([])
        axes[row, 0].set_yticks([])
    axes[0, 1].legend(loc="upper right")
    axes[0, 0].set_title("EEG : spectre (yeux fermés)")
    axes[0, 1].set_title("ECG : forme du battement (médiane)")
    axes[0, 2].set_title("Respiration : 30 s")
    axes[2, 0].set_xlabel("fréquence (Hz)")
    axes[2, 1].set_xlabel("temps autour du pic R (ms)")
    axes[2, 2].set_xlabel("temps (s)")
    for ax in axes[:, 0]:
        ax.axvspan(8, 12, color=GRID, zorder=0)
    axes[0, 0].text(
        10,
        axes[0, 0].get_ylim()[1],
        "alpha",
        ha="center",
        va="top",
        color=INK2,
        fontsize=9,
    )
    fig.suptitle(
        "Trois personnes, deux jours : la signature se retrouve, pas toujours "
        "parfaitement (plein : jour 1, pointillé : jour 2)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "01_signaux_trois_personnes")


# -- 02: LDA, windows of the test people before / after -------------------------
def lda_windows(split) -> dict:
    path = WORK / "lda_eeg_fold0_v2.npz"
    if path.is_file():
        return dict(np.load(path, allow_pickle=False))
    from reid.baselines.b3_lda_plda import LdaModel
    from reid.baselines.common import window_matrix

    model = LdaModel(("eeg",))
    model.fit(split.train)
    pipe = model.pipe
    X, who, day = [], [], []
    people = example_people(split)
    for sset, d in ((split.gallery, 1), (split.probe, 2)):
        for i, pid in enumerate(sset.pids):
            W = window_matrix(sset.load(i), **model.feat)
            X.append(W)
            who += [people.index(pid) if pid in people else -1] * len(W)
            day += [d] * len(W)
    X = np.vstack(X)
    pre = pipe[:-1].transform(X)  # scaled + PCA: the features before the LDA
    lda = pipe.transform(X)
    S = model.score(split.gallery, split.probe)
    top1 = float((S.argmax(1) == np.arange(len(S))).mean())
    WORK.mkdir(parents=True, exist_ok=True)
    out = {
        "pca": pre,
        "lda": lda,
        "who": np.array(who),
        "day": np.array(day),
        "top1": top1,
        "n": model.n_,
    }
    np.savez(path, **out)
    return out


def embed2d(X: np.ndarray) -> np.ndarray:
    """2D map of the windows (t-SNE, cosine), the same for both panels."""
    from sklearn.manifold import TSNE

    return TSNE(
        2, perplexity=30, metric="cosine", init="pca", random_state=SEED
    ).fit_transform(X)


def fig_lda(split) -> None:
    d = lda_windows(split)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    titles = (
        "Avant : spectres EEG bruts (standardisés)",
        "Après : espace appris par la LDA",
    )
    for ax, key, title in zip(axes, ("pca", "lda"), titles):
        P = embed2d(d[key])
        bg = d["who"] < 0
        ax.scatter(*P[bg].T, s=8, color=CONTEXT, lw=0, label="les autres personnes")
        for k, color in enumerate(PEOPLE):
            for day, marker in ((1, "o"), (2, "^")):
                m = (d["who"] == k) & (d["day"] == day)
                ax.scatter(
                    *P[m].T,
                    s=34,
                    color=color,
                    marker=marker,
                    edgecolor="white",
                    lw=0.8,
                    zorder=3,
                )
            m = d["who"] == k
            c = np.median(P[m], 0)
            ax.annotate(
                "ABC"[k],
                c,
                xytext=(8, 8),
                textcoords="offset points",
                color=INK,
                fontweight="bold",
                fontsize=12,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=INK2, label="jour 1"),
        plt.Line2D([], [], marker="^", ls="", color=INK2, label="jour 2"),
        plt.Line2D(
            [], [], marker="o", ls="", color=CONTEXT, label="autres personnes du pli"
        ),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle(
        "La LDA apprend les axes où les personnes diffèrent et où une même "
        "personne varie peu d'un jour à l'autre",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        f"Un point = une fenêtre de 10 s de repos ; carte 2D (t-SNE) de tout l'espace. Personnes de test du pli 0 "
        f"(jamais vues à l'entraînement). LDA EEG seule : top-1 {d['top1']:.2f} "
        f"sur ce pli ({int(d['n'])} composantes PCA).",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "02_lda_avant_apres")


# -- 03 / 04: contrastive training of the beat encoder ---------------------------
def trained_encoder(split, modality: str = "eeg") -> dict:
    """One encoder of the final model (same code and settings, CPU, fold 0,
    no refit): training history and the test people's embeddings before
    training (the initial network, as the training starts from it) and after
    (early-stopped on the inner validation)."""
    path = WORK / f"enc_{modality}_fold0.npz"
    if path.is_file():
        return dict(np.load(path, allow_pickle=False))
    import torch

    from reid.final import EEG_POOL
    from reid.final.encoders import ENCODERS
    from reid.final.model import ContrastiveModel

    kw = {**EEG_POOL, "eval_every": 50} if modality == "eeg" else {"eval_every": 50}
    model = ContrastiveModel(modality, steps=1500, device="cpu", split_seed=SEED, **kw)
    model.fit(split.train)
    gal, prb = model._pools(split.gallery), model._pools(split.probe)
    after = model._embed_pools(gal), model._embed_pools(prb)
    # the initial network, built as in ContrastiveModel._train
    model._prepare(split.train)
    torch.manual_seed(model.seed)
    model.net = ENCODERS[modality](**model.encoder_kw)
    if hasattr(model.net, "set_normalisation"):
        X, _, _ = model._batch(np.random.default_rng(model.seed))
        with torch.no_grad():
            model.net.set_normalisation(model.net.tokens(torch.as_tensor(X)))
    before = model._embed_pools(gal), model._embed_pools(prb)
    hist = np.array(
        [(s, np.nan if l is None else l, a) for s, l, a in model.config["history"]],
        float,
    )
    out = {
        "g_after": after[0],
        "p_after": after[1],
        "g_before": before[0],
        "p_before": before[1],
        "history": hist,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def sim_matrix(ax, G, P, title):
    """Cosine similarities probe x gallery (z-scored per probe), errors ringed."""
    keep = np.isfinite(G).all(1) & np.isfinite(P).all(1)
    G, P = G[keep], P[keep]
    M = znorm(P @ G.T)
    im = ax.imshow(np.clip(M, -1, 4), cmap=SEQ, vmin=-1, vmax=4)
    wrong = np.where(M.argmax(1) != np.arange(len(M)))[0]
    ax.scatter(
        M.argmax(1)[wrong], wrong, s=26, facecolor="none", edgecolor=ORANGE, lw=1.5
    )
    ax.set_title(f"{title}\n{len(M) - len(wrong)}/{len(M)} personnes reconnues")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("galerie (jour 1)")
    ax.set_ylabel("sondes (jour 2)")
    return im


def fig_training(split) -> None:
    d = trained_encoder(split, "eeg")
    fig = plt.figure(figsize=(15, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
    h = d["history"]
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h[1:, 0], h[1:, 1], color=INK2)
    ax.set_title("Perte contrastive (lots d'entraînement)")
    ax.set_ylabel("perte")
    ax.grid(axis="y")
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(h[:, 0], h[:, 2], color=BLUE, marker="o", ms=3)
    best = int(np.argmax(h[:, 2]))
    ax.scatter(h[best, 0], h[best, 2], s=60, color=BLUE, zorder=3)
    ax.annotate(
        f"retenu (arrêt précoce) : {h[best, 2]:.2f}, pas {int(h[best, 0])}",
        (h[best, 0], h[best, 2]),
        xytext=(8, 14),
        textcoords="offset points",
        fontsize=9,
        color=INK2,
    )
    ax.text(8, h[0, 2] - 0.12, f"départ : {h[0, 2]:.2f}", fontsize=9, color=INK2)
    ax.set_title("Top-1 en validation interne (33 personnes)")
    ax.set_xlabel("pas d'entraînement")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    sim_matrix(
        fig.add_subplot(gs[:, 1]), d["g_before"], d["p_before"], "Réseau au départ"
    )
    im = sim_matrix(
        fig.add_subplot(gs[:, 2]), d["g_after"], d["p_after"], "Réseau entraîné"
    )
    fig.colorbar(im, ax=fig.axes[-1], shrink=0.7, label="similarité (z, par sonde)")
    fig.suptitle(
        "Réseau EEG : l'entraînement contrastif fait ressortir la diagonale "
        "(chaque personne retrouve son autre jour)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.04,
        "Personnes de test du pli 0 (jamais vues). Cercles orange : 1er choix "
        "erroné. Un seul réseau, entraîné sur CPU (le modèle final moyenne 10 "
        "entraînements et ajoute 10 autres juges).",
        color=INK2,
        fontsize=9,
    )
    save(fig, "04_reseau_apprentissage")


def fig_contrastive_scheme(split_r) -> None:
    """Schematic of one contrastive step, with real beats of three people."""
    people = example_people(split_r)
    t = np.linspace(-250, 450, 128)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    rows = [("A", 1, 0), ("A", 2, 0), ("B", 2, 1), ("C", 2, 2)]
    ys = [4.9, 3.5, 2.1, 0.7]
    emb = {}
    for (lab, day, k), y in zip(rows, ys):
        r = split_r.gallery.store.load(people[k], day, "rest")
        beats = clean_beats(r)
        crop = beats[: len(beats) // 2]  # the more typical half
        crop = crop[np.random.default_rng(SEED).choice(len(crop), 12, replace=False)]
        sub = fig.add_axes(
            ax.transData.transform_bbox(
                matplotlib.transforms.Bbox([[0.2, y - 0.55], [2.6, y + 0.55]])
            ).transformed(fig.transFigure.inverted())
        )
        for b in crop:
            sub.plot(t, b, color=PEOPLE[k], lw=0.6, alpha=0.5)
        sub.axis("off")
        ax.text(
            0.15,
            y,
            f"{lab}\njour {day}",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
        ax.annotate(
            "",
            (4.1, y),
            (2.75, y),
            arrowprops={"arrowstyle": "->", "color": INK2, "lw": 1.2},
        )
        emb[(lab, day)] = y
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch(
            (4.15, 0.2),
            1.9,
            5.3,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            fc="#f2f1ee",
            ec=INK2,
            lw=1,
        )
    )
    ax.text(
        5.1,
        2.85,
        "même\nréseau\n(poids\npartagés)",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
    )
    # embeddings on a circle-like 2D map
    pos = {
        ("A", 1): (9.2, 3.4),
        ("A", 2): (10.1, 4.1),
        ("B", 2): (8.2, 1.4),
        ("C", 2): (11.6, 1.6),
    }
    for (lab, day), y in emb.items():
        ax.annotate(
            "",
            (pos[(lab, day)][0] - 0.25, pos[(lab, day)][1]),
            (6.15, y),
            arrowprops={"arrowstyle": "->", "color": CONTEXT, "lw": 1},
        )
    for (lab, day), (x, y) in pos.items():
        k = "ABC".index(lab)
        ax.scatter(
            x,
            y,
            s=180,
            color=PEOPLE[k],
            marker="o" if day == 1 else "^",
            zorder=3,
            edgecolor="white",
        )
        ax.text(
            x + (0.22 if (lab, day) != ("A", 1) else -0.75),
            y + 0.2,
            f"{lab} j{day}",
            fontsize=10,
            color=INK,
        )
    ax.annotate(
        "",
        pos[("A", 1)],
        pos[("A", 2)],
        arrowprops={
            "arrowstyle": "<->",
            "color": BLUE,
            "lw": 2.2,
            "shrinkA": 9,
            "shrinkB": 9,
        },
    )
    ax.text(
        10.3,
        4.6,
        "rapprocher\n(même personne, autre jour)",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
    )
    for other in (("B", 2), ("C", 2)):
        ax.annotate(
            "",
            pos[("A", 1)],
            pos[other],
            arrowprops={
                "arrowstyle": "<->",
                "color": ORANGE,
                "lw": 1.6,
                "ls": "--",
                "shrinkA": 9,
                "shrinkB": 9,
            },
        )
    ax.text(
        9.0,
        0.35,
        "éloigner (autres personnes)",
        color=ORANGE,
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        10.0,
        5.6,
        "espace des empreintes (64 dimensions)",
        ha="center",
        color=INK2,
        fontsize=10,
    )
    ax.text(
        0.2,
        5.75,
        "battements réels (12 par séance, centrés-réduits)",
        color=INK2,
        fontsize=10,
    )
    fig.suptitle(
        "Entraînement contrastif : rapprocher les deux jours d'une même "
        "personne, éloigner les autres",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    save(fig, "03_contrastif_schema")


# -- 05-07: the judges ----------------------------------------------------------
def fold_matrices(data, slots):
    """Test fold with the most probes: fused and per-slot z-scored matrices."""
    fold = max(data["test"].values(), key=lambda f: len(f["truth"]))
    Z = [znorm(np.mean([znorm(fold[m]) for m in s], 0)) for s in slots]
    return fold["truth"], Z, znorm(np.mean(Z, 0))


def fig_matrices(data, slots) -> None:
    truth, Z, F = fold_matrices(data, slots)
    pick = [0, 3, 7, 8]  # LDA EEG rest, beat net rest, resp net video, LDA EEG vid
    mats = [(SLOT_NAMES[i], Z[i]) for i in pick] + [("Fusion des 11 juges", F)]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.0), layout="constrained")
    for ax, (name, M) in zip(axes, mats):
        im = ax.imshow(np.clip(M, -1, 4), cmap=SEQ, vmin=-1, vmax=4)
        wrong = np.where(M.argmax(1) != truth)[0]
        ax.scatter(
            M.argmax(1)[wrong], wrong, s=28, facecolor="none", edgecolor=ORANGE, lw=1.6
        )
        ok = len(truth) - len(wrong)
        ax.set_title(
            f"{name}\n{ok}/{len(truth)} bien classées",
            fontsize=11,
            fontweight="bold" if "Fusion" in name else None,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("galerie (jour 1)")
    axes[0].set_ylabel("sondes (jour 2)")
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("score (z, par sonde)")
    fig.suptitle(
        "Chaque juge note toutes les paires ; la bonne personne est sur la "
        "diagonale. Cercles orange : 1er choix erroné.",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    save(fig, "05_juges_matrices")


def fig_complementary(data, slots) -> None:
    t = collect(data["test"], slots)
    rows = [collect(data["test"], [s])["present"]["correct"] for s in slots]
    fused_ok = t["present"]["correct"]
    acc = np.array([r.mean() for r in rows])
    order = np.argsort(-acc)
    M = np.array([rows[i] for i in order] + [fused_ok], float)
    cols = np.lexsort((-fused_ok.astype(int), -M[:-1].sum(0)))
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [2.4, 1]}
    )
    cmap = LinearSegmentedColormap.from_list("ok", [ORANGE, "#dcebfb"])
    ax.imshow(
        M[:, cols], aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
    )
    ax.axhline(len(slots) - 0.5, color="white", lw=4)
    labels = [SLOT_NAMES[i] for i in order] + ["FUSION"]
    ax.set_yticks(range(len(labels)), labels)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xticks([])
    ax.set_xlabel(f"les {M.shape[1]} personnes de test (triées)")
    ax.set_title("Qui se trompe sur qui ? orange = 1er choix erroné")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    names = [SLOT_NAMES[i] for i in order][::-1]
    vals = acc[order][::-1]
    y = np.arange(len(names))
    bx.barh(y, vals, color=CONTEXT, height=0.7)
    bx.barh(len(names), fused_ok.mean(), color=BLUE, height=0.7)
    for yy, v in zip(list(y) + [len(names)], list(vals) + [fused_ok.mean()]):
        bx.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=9, color=INK)
    bx.set_yticks(list(y) + [len(names)], names + ["FUSION"])
    bx.get_yticklabels()[-1].set_fontweight("bold")
    chance = test_summary(parts(data["test"], slots, "z"), np.ones(len(slots)))[
        "chance_top1"
    ][0]
    bx.axvline(chance, color=INK2, ls=":", lw=1.2)
    bx.text(chance + 0.01, -0.9, f"hasard {chance:.3f}", fontsize=8, color=INK2)
    bx.set_xlim(0, 1.08)
    bx.set_xlabel("top-1 au test (208 personnes)")
    bx.set_title("Chaque juge seul vs ensemble")
    fig.suptitle(
        "Les juges se trompent sur des personnes différentes : "
        "la moyenne corrige presque toutes les erreurs",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "06_juges_complementaires")


def fig_progress(data, slots) -> None:
    from scripts.reid_offline_fusion import variants

    members = set.intersection(*(set(f) - {"truth"} for f in data["test"].values()))
    v = variants(members, list(range(10)))
    steps = [
        ("Repos, 5 juges\n1 entraînement", v["all5 s0"]),
        ("Repos, 5 juges\nmoyenne de 10\nentraînements", v["all5 ens10"]),
        ("+ vidéo de la\nmême séance\n(11 juges)", slots),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for x, (name, s) in enumerate(steps):
        r = test_summary(parts(data["test"], s, "z"), np.full(len(s), 1 / len(s)))
        top, lo, hi = r["top1"]
        ax.plot([x, x], [lo, hi], color=BLUE, lw=2)
        ax.scatter(x, top, s=80, color=BLUE, zorder=3)
        ax.text(
            x + 0.08, top, f"{top:.3f}", va="center", fontsize=11, fontweight="bold"
        )
    ax.set_xticks(range(len(steps)), [s[0] for s in steps])
    ax.set_ylim(0.9, 1.0)
    ax.set_xlim(-0.4, len(steps) - 0.5)
    ax.grid(axis="y")
    ax.set_ylabel("top-1 au test (208 personnes)")
    ax.set_title("Progression du modèle (IC 95 % par bootstrap ; hasard 0,024)")
    save(fig, "07_progression")


# -- 08 / 09: open set ----------------------------------------------------------
def fig_absent(data, slots) -> None:
    import json

    res = json.loads((RESULTS / "open_set.json").read_text())
    t = collect(data["test"], slots)
    p, a = t["present"], t["absent"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.8))
    bins = np.linspace(0.4, 4.0, 37)
    ax.hist(p["top"], bins, color=BLUE, alpha=0.85, label="personne présente au jour 1")
    ax.hist(
        a["top"], bins, color=ORANGE, alpha=0.75, label="personne absente (simulée)"
    )
    thr = res["top"]["fpir1"]
    ax.axvline(thr["threshold"], color=INK, ls="--", lw=1.4)
    ax.text(
        thr["threshold"] + 0.08,
        ax.get_ylim()[1] * 0.92,
        f"seuil (fixé en validation)\nabsents acceptés : "
        f"{thr['fpir']:.0%}\nprésents identifiés : {thr['dir']:.0%}",
        fontsize=9,
        va="top",
    )
    ax.set_xlabel("meilleur score de la galerie (z)")
    ax.set_ylabel("personnes")
    ax.set_title(f"Critère 1 : le meilleur score (AUC {res['top']['auc_test']:.2f})")
    ax.legend(loc="upper right", fontsize=9)
    k = np.arange(len(slots) + 1)
    w = 0.4
    bx.bar(
        k - w / 2,
        [np.mean(p["agree"] == i) for i in k],
        w,
        color=BLUE,
        label="présente",
    )
    bx.bar(
        k + w / 2,
        [np.mean(a["agree"] == i) for i in k],
        w,
        color=ORANGE,
        label="absente",
    )
    bx.set_xticks(k)
    bx.set_xlabel("nombre de juges (sur 11) d'accord avec le choix final")
    bx.set_ylabel("part des personnes")
    bx.set_title(f"Critère 2 : le vote des juges (AUC {res['agree']['auc_test']:.2f})")
    bx.legend(loc="upper center", fontsize=9)
    fig.suptitle(
        "Personne absente au jour 1 : peut-on voir qu'aucune ne correspond ?",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.04,
        "Test, 208 personnes : chacune est sondée deux fois, avec puis sans "
        "sa séance du jour 1 dans la galerie (44 personnes par galerie).",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "08_personne_absente")


def fig_absent_real() -> None:
    path = WORK / "open_set.npz"
    if not path.is_file():
        print(f"skip 09: run reid_open_set.py --real --npz {path}")
        return
    d = np.load(path)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bins = np.linspace(0.4, 5.0, 33)
    for key, color, label in (
        ("lda_present_top", BLUE, "présente (208)"),
        ("lda_simulated_top", ORANGE, "absente simulée (208)"),
        (
            "lda_real_top",
            AQUA,
            f"absente réelle : jour 2 seul ({len(d['lda_real_top'])})",
        ),
    ):
        x = d[key]
        ax.hist(
            x,
            bins,
            weights=np.full(len(x), 1 / len(x)),
            histtype="step",
            lw=2.2,
            color=color,
            label=label,
        )
    ax.set_xlabel("meilleur score de la galerie (z) — LDA EEG + battement, repos")
    ax.set_ylabel("part des sondes")
    ax.set_title("La simulation reproduit les vraies personnes absentes")
    ax.legend(fontsize=9)
    save(fig, "09_absent_reel_vs_simule")


FIGS = {
    "01_signaux": lambda c: fig_signals(*c["fold0"]),
    "02_lda": lambda c: fig_lda(c["fold0"][0]),
    "03_contrastif": lambda c: fig_contrastive_scheme(c["fold0"][1]),
    "04_reseau": lambda c: fig_training(c["fold0"][1]),
    "05_matrices": lambda c: fig_matrices(*c["final"]),
    "06_juges": lambda c: fig_complementary(*c["final"]),
    "07_progression": lambda c: fig_progress(*c["final"]),
    "08_absent": lambda c: fig_absent(*c["final"]),
    "09_absent_reel": lambda c: fig_absent_real(),
}


class Ctx(dict):
    def __missing__(self, key):
        self[key] = fold0() if key == "fold0" else final_slots()
        return self[key]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default=",".join(FIGS))
    args = ap.parse_args()
    ctx = Ctx()
    for name in args.only.split(","):
        FIGS[name](ctx)


if __name__ == "__main__":
    main()
