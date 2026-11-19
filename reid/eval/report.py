"""Leaderboard and figures from results/runs/*.json (aggregated data only)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reid.config import RESULTS

#: Reference categorical palette (light mode), slots 1-3, plus neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def load_runs(runs_dir: Path = RESULTS / "runs") -> list[dict]:
    """All runs, latest first; only the latest run per (model, scenario, tasks)."""
    runs = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]
    latest = {}
    for r in sorted(runs, key=lambda r: r["date"]):
        latest[
            (r["model"], r["scenario"], r["task"], r["probe_task"], r.get("part"))
        ] = r
    return list(latest.values())


def _fmt(metric: dict | None) -> tuple[float, float, float]:
    if not metric:
        return (float("nan"),) * 3
    lo, hi = metric.get("ci", [float("nan")] * 2)
    return metric["value"], lo, hi


def leaderboard(runs: list[dict]) -> pd.DataFrame:
    """One row per run: pooled metrics with CIs, chance and controls."""
    rows = []
    for r in runs:
        p, c = r["pooled"], r.get("controls", {})
        row = {
            "model": r["model"],
            "scenario": r["scenario"],
            "task": r["task"]
            if r["task"] == r["probe_task"]
            else f"{r['task']}->{r['probe_task']}",
            "n_probes": p["n_probes"],
        }
        for key in ("top1", "top5", "mrr", "hungarian", "auc", "eer"):
            row[key], row[f"{key}_lo"], row[f"{key}_hi"] = _fmt(p.get(key))
        row["chance_top1"] = p["chance_top1"]["value"]
        row["demographics_top1"] = (
            c.get("demographics", {}).get("top1", {}).get("value")
        )
        row["shuffled_top1"] = c.get("shuffled", {}).get("top1", {}).get("value")
        cohort = c.get("cohort_prediction", {})
        row["cohort_acc"], row["cohort_majority"] = (
            cohort.get("accuracy"),
            cohort.get("majority"),
        )
        row["top1_fold_sd"] = pd.Series(
            [f["top1"]["value"] for f in r["per_fold"]]
        ).std()
        row["ablation"], row["label"] = r.get("ablation"), r.get("label")
        row["seconds"] = r.get("seconds")
        row["date"], row["git"] = r["date"], r["git"][:8]
        rows.append(row)
    board = pd.DataFrame(rows)
    return board.sort_values(
        ["scenario", "task", "top1"], ascending=[True, True, False]
    )


def _bars(sub: pd.DataFrame, labels: str, title: str, out: Path) -> None:
    """Horizontal top-1 bars with 95 % CI, chance and demographic-control lines."""
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(sub) + 1.2))
    y = range(len(sub))
    err = [sub.top1 - sub.top1_lo, sub.top1_hi - sub.top1]
    ax.barh(
        y,
        sub.top1,
        height=0.55,
        color=SERIES[0],
        xerr=err,
        error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2},
    )
    ax.set_yticks(list(y), sub[labels])
    ax.axvline(sub.chance_top1.mean(), color=INK_2, ls="--", lw=1, label="chance")
    demo = sub.demographics_top1.dropna()
    if len(demo):
        ax.axvline(demo.mean(), color=INK_2, ls=":", lw=1.2, label="demographics only")
    ax.set_xlim(0, 1)
    ax.set_xlabel("top-1 accuracy (95 % CI over people)")
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    # legend below the axes: never on top of a bar or its error bar
    fig.legend(
        *ax.get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2
    )
    fig.tight_layout(rect=(0, 0.35 / fig.get_figheight(), 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def bar_top1(board: pd.DataFrame, scenario: str, task: str, out: Path) -> None:
    """Top-1 per model (ablation runs and controls excluded)."""
    sub = board[(board.scenario == scenario) & (board.task == task)]
    sub = sub[~sub.model.str.startswith(("control_", "abl_"))].sort_values("top1")
    _bars(sub, "model", f"{scenario} · {task}", out)


def ablation_figures(board: pd.DataFrame, figures: Path) -> None:
    """One figure per (ablation kind, method): heatmap for modalities, curve for
    durations, bars otherwise. Ablation model names are abl_<method>_<label>."""
    abl = board.dropna(subset=["ablation"])
    if abl.empty:
        return
    method = [
        m[len("abl_") : -len(lab) - 1]
        for m, lab in zip(abl.model, abl.label, strict=True)
    ]
    for (kind, meth), sub in abl.assign(method=method).groupby(["ablation", "method"]):
        out = figures / f"ablation_{kind}_{meth}.png"
        if kind == "modality":
            _modality_heatmap(sub, f"{meth} · modality ablation", out)
        elif kind == "duration":
            _duration_curve(sub, f"{meth} · top-1 vs recording duration", out)
        else:
            _bars(sub.sort_values("top1"), "label", f"{meth} · {kind}", out)


def _modality_heatmap(sub: pd.DataFrame, title: str, out: Path) -> None:
    """Rows: modality alone / all modalities but this one; cells: top-1."""
    parts = sub.label.str.split("_", n=1, expand=True)
    table = sub.assign(mode=parts[0], modality=parts[1]).pivot_table(
        index="mode", columns="modality", values="top1"
    )
    fig, ax = plt.subplots(figsize=(1.2 * table.shape[1] + 1.8, 2.4))
    im = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    names = {"only": "this modality only", "minus": "all but this one"}
    ax.set_yticks(range(table.shape[0]), [names.get(m, m) for m in table.index])
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v > 0.6 else INK,
            )
    ax.grid(False)
    ax.set_title(title, loc="left", color=INK)
    fig.colorbar(im, ax=ax, label="top-1")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _duration_curve(sub: pd.DataFrame, title: str, out: Path) -> None:
    """Top-1 vs seconds kept per condition, one line per task, with CI band."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for color, (task, g) in zip(SERIES, sub.groupby("task"), strict=False):
        g = g.sort_values("seconds")
        ax.plot(g.seconds, g.top1, color=color, lw=2, marker="o", ms=4, label=task)
        ax.fill_between(g.seconds, g.top1_lo, g.top1_hi, color=color, alpha=0.15, lw=0)
    ax.axhline(sub.chance_top1.mean(), color=INK_2, ls="--", lw=1, label="chance")
    ax.set_xscale("log")
    ticks = sorted(sub.seconds.unique())
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("seconds per condition (gallery and probe)")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(title, loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def gallery_curve(runs: list[dict], out: Path, max_models: int = 3) -> None:
    """Top-1 vs gallery size: best d1_to_d2 run of the ``max_models`` best
    families (b1, b2, ...; ablations excluded), demographic control in grey."""
    runs = [
        r
        for r in runs
        if r["scenario"] == "d1_to_d2"
        and r.get("gallery_size")
        and not r["model"].startswith("abl_")
    ]
    demo = [r for r in runs if r["model"].startswith("control_")]
    best: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: -r["pooled"]["top1"]["value"]):
        if not r["model"].startswith("control_"):
            best.setdefault(r["model"].split("_")[0], r)
    runs = list(best.values())[:max_models]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for r in demo[:1]:
        pts = sorted(r["gallery_size"].values(), key=lambda p: p["n"])
        ax.plot(
            [p["n"] for p in pts],
            [p["top1"] for p in pts],
            color=INK_2,
            ls=":",
            lw=1.5,
            label="demographics only",
        )
    for color, r in zip(SERIES, runs):
        pts = sorted(r["gallery_size"].values(), key=lambda p: p["n"])
        n = [p["n"] for p in pts]
        ax.plot(
            n,
            [p["top1"] for p in pts],
            color=color,
            lw=2,
            marker="o",
            ms=4,
            label=f"{r['model']} ({r['task']})",
        )
        ax.fill_between(
            n,
            [p["ci"][0] for p in pts],
            [p["ci"][1] for p in pts],
            color=color,
            alpha=0.15,
            lw=0,
        )
    ax.plot(n, [1 / k for k in n], color=INK_2, ls="--", lw=1, label="chance")
    ax.set_xscale("log")
    ax.set_xticks(n, [str(k) for k in n])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("gallery size (people)")
    ax.set_ylabel("top-1 accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def roc_and_scores(runs: list[dict], out: Path, max_models: int = 3) -> None:
    """ROC curves (left) and genuine/impostor score histograms of the best model."""
    runs = [r for r in runs if r["scenario"] == "d1_to_d2" and r.get("curves")]
    runs = sorted(runs, key=lambda r: -r["pooled"]["top1"]["value"])[:max_models]
    if not runs:
        return
    fig, (roc, hist) = plt.subplots(1, 2, figsize=(9, 3.5))
    for color, r in zip(SERIES, runs):
        roc.plot(
            r["curves"]["roc_fpr"],
            r["curves"]["roc_tpr"],
            color=color,
            lw=2,
            label=f"{r['model']} (EER {r['pooled']['eer']['value']:.2f})",
        )
    roc.plot([0, 1], [0, 1], color=INK_2, ls="--", lw=1)
    roc.set_xlabel("false match rate")
    roc.set_ylabel("true match rate")
    roc.legend(frameon=False, loc="lower right")
    best = runs[0]["curves"]
    edges = best["hist_edges"]
    for key, color, label in (
        ("hist_impostor", SERIES[1], "different person"),
        ("hist_genuine", SERIES[0], "same person"),
    ):
        hist.stairs(best[key], edges, color=color, lw=2, label=label)
    hist.set_xlabel(f"score · {runs[0]['model']}")
    hist.set_ylabel("density")
    hist.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_report(results: Path = RESULTS) -> pd.DataFrame:
    """Regenerate results/leaderboard.csv and results/figures/*.png."""
    runs = load_runs(results / "runs")
    if not runs:
        print("no run in results/runs")
        return pd.DataFrame()
    board = leaderboard(runs)
    board.to_csv(results / "leaderboard.csv", index=False)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for (scenario, task), _ in board.groupby(["scenario", "task"]):
        bar_top1(board, scenario, task, figures / f"top1_{scenario}_{task}.png")
    gallery_curve(runs, figures / "gallery_size.png")
    roc_and_scores(runs, figures / "roc_scores.png")
    ablation_figures(board, figures)
    return board
