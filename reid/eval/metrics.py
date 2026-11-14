"""Identification and verification metrics with person-level bootstrap CIs.

Every metric is computed on a list of *blocks* (one score matrix per fold), so
the same code gives per-fold values (one block) and pooled values (all blocks).
The bootstrap resamples probe people (one probe per person) with replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from reid.config import SEED
from reid.eval import matching


@dataclass
class Block:
    """One score matrix with the index of each probe's true gallery entry."""

    S: np.ndarray
    truth: np.ndarray
    one_to_one: bool  # gallery and probe hold exactly the same people


def per_probe(block: Block) -> dict[str, np.ndarray]:
    """Per-probe identification outcomes (probes without a true match dropped)."""
    keep = block.truth >= 0
    S, truth = block.S[keep], block.truth[keep]
    greater, ties = matching.rank_counts(S, truth)
    n_valid = np.isfinite(S).sum(axis=1)
    if block.one_to_one:
        hung = (matching.hungarian(block.S)[keep] == truth).astype(float)
    else:
        hung = np.full(len(truth), np.nan)
    return {
        "top1": matching.topk(greater, ties, 1),
        "top5": matching.topk(greater, ties, 5),
        "mrr": matching.reciprocal_rank(greater, ties),
        "rank": greater + ties / 2 + 1,
        "chance_top1": 1.0 / np.maximum(n_valid, 1),
        "hungarian": hung,
    }


def pair_scores(block: Block, rows: np.ndarray | None = None):
    """Genuine and impostor scores (verification pairs) of the selected probes."""
    rows = np.flatnonzero(block.truth >= 0) if rows is None else rows
    genuine, impostor = [], []
    for i in rows:
        s = block.S[i]
        genuine.append(s[block.truth[i]])
        mask = np.isfinite(s)
        mask[block.truth[i]] = False
        impostor.append(s[mask])
    return np.asarray(genuine), np.concatenate(impostor) if impostor else np.array([])


def auc_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """ROC AUC and equal error rate of same- vs different-person pairs."""
    genuine = genuine[np.isfinite(genuine)]
    if len(genuine) == 0 or len(impostor) == 0:
        return np.nan, np.nan
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    fpr, tpr, _ = roc_curve(y, s)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return float(roc_auc_score(y, s)), float((fpr[i] + 1 - tpr[i]) / 2)


def _point(blocks: list[Block], outcomes: list[dict], picks) -> dict[str, float]:
    """Metric values for a selection ``picks`` = list of (block, probe) indices."""
    out = {}
    for key in ("top1", "top5", "mrr", "chance_top1", "hungarian"):
        vals = np.array([outcomes[b][key][i] for b, i in picks])
        out[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    out["median_rank"] = float(np.median([outcomes[b]["rank"][i] for b, i in picks]))
    gen, imp = [], []
    for b in range(len(blocks)):
        rows = np.array([i for bb, i in picks if bb == b], dtype=int)
        valid = np.flatnonzero(blocks[b].truth >= 0)[rows] if len(rows) else rows
        g, m = pair_scores(blocks[b], valid)
        gen.append(g)
        imp.append(m)
    out["auc"], out["eer"] = auc_eer(np.concatenate(gen), np.concatenate(imp))
    return out


def summarize(blocks: list[Block], n_boot: int = 1000, seed: int = SEED) -> dict:
    """Point estimates, 95 % bootstrap CIs over probe people, and probe count."""
    outcomes = [per_probe(b) for b in blocks]
    picks = [(b, i) for b, o in enumerate(outcomes) for i in range(len(o["top1"]))]
    if not picks:
        return {"n_probes": 0}
    point = _point(blocks, outcomes, picks)
    result = {k: {"value": v} for k, v in point.items()}
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = [
            _point(
                blocks,
                outcomes,
                [picks[j] for j in rng.integers(0, len(picks), len(picks))],
            )
            for _ in range(n_boot)
        ]
        for k in point:
            vals = np.array([b[k] for b in boots], dtype=float)
            if np.isfinite(vals).any():
                result[k]["ci"] = np.nanpercentile(vals, [2.5, 97.5]).tolist()
    result["n_probes"] = len(picks)
    return result


def curves(blocks: list[Block], n_points: int = 101, n_bins: int = 40) -> dict:
    """Aggregated ROC points and genuine/impostor histograms (no per-person data)."""
    pairs = [pair_scores(b) for b in blocks]
    gen = np.concatenate([g for g, _ in pairs])
    imp = np.concatenate([m for _, m in pairs])
    gen, imp = gen[np.isfinite(gen)], imp[np.isfinite(imp)]
    if not len(gen) or not len(imp):
        return {}
    fpr, tpr, _ = roc_curve(
        np.r_[np.ones(len(gen)), np.zeros(len(imp))], np.r_[gen, imp]
    )
    grid = np.linspace(0, 1, n_points)
    edges = np.linspace(
        min(gen.min(), imp.min()), max(gen.max(), imp.max()), n_bins + 1
    )
    return {
        "roc_fpr": grid.tolist(),
        "roc_tpr": np.interp(grid, fpr, tpr).tolist(),
        "hist_edges": edges.tolist(),
        "hist_genuine": np.histogram(gen, edges, density=True)[0].tolist(),
        "hist_impostor": np.histogram(imp, edges, density=True)[0].tolist(),
    }


def gallery_size_curve(
    blocks: list[Block],
    sizes=(5, 10, 25, 50, 100),
    n_draws: int = 200,
    seed: int = SEED,
) -> dict:
    """Top-1 vs number of gallery people: random sub-galleries inside each block
    (only sizes that fit in a block; 'all' = full block)."""
    rng = np.random.default_rng(seed)
    out = {}
    max_n = min(int((b.truth >= 0).sum()) for b in blocks)
    for n in [s for s in sizes if s < max_n] + [max_n]:
        draws = []
        for _ in range(n_draws):
            accs = []
            for b in blocks:
                probes = np.flatnonzero(b.truth >= 0)
                chosen = rng.choice(probes, n, replace=False)
                cols = b.truth[chosen]
                sub = Block(b.S[np.ix_(chosen, cols)], np.arange(n), False)
                accs.append(per_probe(sub)["top1"])
            draws.append(np.concatenate(accs).mean())
        label = str(n) if n < max_n else f"all ({n})"
        out[label] = {
            "n": n,
            "top1": float(np.mean(draws)),
            "ci": np.percentile(draws, [2.5, 97.5]).tolist(),
            "chance": 1.0 / n,
        }
    return out
