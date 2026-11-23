"""Pretraining of the EEG encoder on the EEG of the emotion-video task.

The video task holds ~13 times more EEG per session than rest (~14 min vs
~90 s), so it can teach the encoder what EEG looks like before it learns who
it belongs to. Objectives (``objective``):

* ``mae`` -- spatial world model: half of the electrode tokens are replaced by
  a learned mask token; the transformer predicts their (standardised) spectra
  from the other electrodes (MSE on masked tokens only).
* ``jepa`` -- temporal world model: from a crop at time t, a predictor maps the
  online embedding, plus the time offset, to the embedding of a crop of the
  same recording 4-60 s later (or earlier) given by a slowly updated (EMA)
  copy of the encoder (BYOL/JEPA; cosine loss, symmetric).
* ``session`` -- instance contrast without labels: two crops of the same
  recording are positives, crops of other recordings negatives.
* ``xday`` -- the fine-tuning objective on more data: SupCon with positives =
  same person on another day (training people's identities only).

Which people: the caller passes the people the pretraining may see -- in
``ContrastiveModel`` the *inner* training people of the fold (neither the test
people nor the inner validation people), so neither the test score nor the
early stopping that the fusion relies on sees anything of those people.

Results are memoised on disk under ``CACHE/pretrain`` (key: config + people),
with a file lock, so that the fine-tuning seeds of a job, run in parallel
processes, share one pretraining per fold.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from reid.baselines.b5_siamese import supcon_loss
from reid.config import CACHE, SEED
from reid.final.crops import EEGCrops
from reid.final.encoders import ENCODERS, augment
from reid.final.raw import EEG_RATE
from reid.final.video import VIDEO, load_video, video_sessions

OBJECTIVES = ("mae", "jepa", "session", "xday")
DEFAULTS = {
    "objective": "session",
    "steps": 3000,
    "max_s": 600.0,  # wall-clock cap of one pretraining
    "batch": 128,  # sessions (session/jepa), people (xday) or crops/2 (mae)
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "warmup": 100,
    "temperature": 0.1,
    "mask_ratio": 0.5,
    "ema": 0.996,
    "dt_s": [4.0, 60.0],
    "seed": SEED,
    "encoder_kw": {},
}


def config(**kw) -> dict:
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown pretraining options {sorted(unknown)}")
    cfg = DEFAULTS | kw
    if cfg["objective"] not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    return cfg


def _key(pids, cfg: dict, root: Path) -> str:
    sessions = [s for s in video_sessions(root) if s[0] in set(pids)]
    blob = json.dumps([sorted(pids), sessions, cfg], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class _Data:
    """Crop pools of the video sessions of ``pids``."""

    def __init__(self, pids, root: Path, crops: EEGCrops) -> None:
        self.crops = crops
        self.pools, self.pid, self.day = [], [], []
        wanted = set(pids)
        for pid, day in video_sessions(root):
            if pid not in wanted:
                continue
            feats = load_video(pid, day, root)
            pool = crops.prepare(feats, ("video",)) if feats else None
            if pool:
                self.pools.append(pool)
                self.pid.append(pid)
                self.day.append(day)
        self.pid, self.day = np.array(self.pid), np.array(self.day)
        by_pid: dict = {}
        for i, p in enumerate(self.pid):
            by_pid.setdefault(p, []).append(i)
        self.multi = [
            idx for idx in by_pid.values() if len(set(self.day[idx].tolist())) >= 2
        ]

    def crop(self, i: int, rng) -> np.ndarray:
        return self.crops.sample(self.pools[i], 1, rng)[0]

    def pair(self, i: int, dt: tuple[float, float], rng):
        """Two crops of session i, ``dt`` seconds apart (signed offset)."""
        size = round(self.crops.crop_s * EEG_RATE)
        seg, starts = self.pools[i][rng.integers(len(self.pools[i]))]
        a = int(rng.choice(starts))
        lo, hi = (round(d * EEG_RATE) for d in dt)
        gap = np.abs(starts - a)
        ok = starts[(gap >= lo) & (gap <= hi)]
        b = int(rng.choice(ok)) if len(ok) else int(rng.choice(starts))
        return seg[:, a : a + size], seg[:, b : b + size], (b - a) / EEG_RATE


def _lr_lambda(warmup: int, steps: int):
    def f(s):
        return (
            min(1.0, (s + 1) / warmup)
            * 0.5
            * (1 + math.cos(math.pi * min(s, steps) / steps))
        )

    return f


def _train(data: _Data, cfg: dict, device: str, log=print) -> tuple[dict, dict]:
    import torch
    import torch.nn.functional as F
    from torch import nn

    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    obj, B = cfg["objective"], cfg["batch"]
    net = ENCODERS["eeg"](**cfg["encoder_kw"]).to(device)
    sample = np.stack(
        [data.crop(i, rng) for i in rng.integers(len(data.pools), size=1024)]
    )
    with torch.no_grad():
        net.set_normalisation(net.tokens(torch.as_tensor(sample, device=device)))
    dim, out = net.proj.out_features, net.head.out_features
    extra = nn.ModuleDict()
    if obj == "mae":
        extra["mask"] = nn.Embedding(1, dim)
        extra["decoder"] = nn.Linear(dim, net.proj.in_features)
    if obj == "jepa":
        extra["predictor"] = nn.Sequential(
            nn.Linear(out + 1, 2 * out), nn.GELU(), nn.Linear(2 * out, out)
        )
        target = copy.deepcopy(net).requires_grad_(False).eval()  # no dropout/mask
    extra = extra.to(device)
    params = list(net.parameters()) + list(extra.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(cfg["warmup"], cfg["steps"])
    )
    as_t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
    start, history, step, n = time.time(), [], 0, len(data.pools)
    for step in range(1, cfg["steps"] + 1):
        net.train()
        if obj == "mae":
            x = as_t([data.crop(i, rng) for i in rng.integers(n, size=2 * B)])
            x = augment("eeg", x)
            tok = (net.tokens(x) - net.mu) / net.sd
            h = net.proj(tok)
            m = torch.rand(h.shape[:2], device=device) < cfg["mask_ratio"]
            h = torch.where(m[..., None], extra["mask"].weight[0], h) + net.channel
            pred = extra["decoder"](net.norm(net.body(h)))
            loss = (pred - tok).pow(2).mean(-1)[m].mean()
        elif obj == "jepa":
            idx = rng.integers(n, size=B)
            pairs = [data.pair(i, cfg["dt_s"], rng) for i in idx]
            xa = augment("eeg", as_t([p[0] for p in pairs]))
            xb = augment("eeg", as_t([p[1] for p in pairs]))
            dt = as_t([[p[2] / cfg["dt_s"][1]] for p in pairs])
            za, zb = net(xa), net(xb)
            with torch.no_grad():
                ta, tb = target(xa), target(xb)
            pa = extra["predictor"](torch.cat([za, dt], 1))
            pb = extra["predictor"](torch.cat([zb, -dt], 1))
            loss = (
                2
                - F.cosine_similarity(pa, tb).mean()
                - F.cosine_similarity(pb, ta).mean()
            )
        elif obj == "session":  # distinct sessions: no false negatives
            idx = rng.choice(n, min(B, n), replace=False)
            x = as_t([data.crop(i, rng) for i in idx for _ in range(2)])
            z = F.normalize(net(augment("eeg", x)), dim=1)
            ids = torch.as_tensor(np.repeat(np.arange(len(idx)), 2), device=device)
            views = torch.as_tensor(np.tile([0, 1], len(idx)), device=device)
            loss = supcon_loss(z, ids, views, cfg["temperature"])
        else:  # xday
            people = rng.choice(len(data.multi), min(B, len(data.multi)), replace=False)
            x, person, day = [], [], []
            for k, p in enumerate(people):
                sessions = np.array(data.multi[p])
                days = data.day[sessions]
                for d in rng.choice(np.unique(days), 2, replace=False):
                    i = int(rng.choice(sessions[days == d]))
                    x += [data.crop(i, rng) for _ in range(2)]
                    person += [k, k]
                    day += [int(d)] * 2
            z = F.normalize(net(augment("eeg", as_t(x))), dim=1)
            loss = supcon_loss(
                z,
                torch.as_tensor(person, device=device),
                torch.as_tensor(day, device=device),
                cfg["temperature"],
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if obj == "jepa":
            tau = (
                1 - (1 - cfg["ema"]) * (math.cos(math.pi * step / cfg["steps"]) + 1) / 2
            )
            with torch.no_grad():
                for pt, po in zip(target.parameters(), net.parameters(), strict=True):
                    pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
                for bt, bo in zip(target.buffers(), net.buffers(), strict=True):
                    bt.copy_(bo)
        stop = time.time() - start > cfg["max_s"] or step == cfg["steps"]
        if step % 100 == 0 or stop:  # and always the last step
            entry = {"step": step, "loss": round(loss.item(), 4)}
            if obj == "jepa":  # collapse check: spread of normalised embeddings
                entry["z_std"] = round(F.normalize(za, dim=1).std(0).mean().item(), 4)
            history.append(entry)
        if stop:
            break
    info = {
        "objective": obj,
        "sessions": n,
        "people": len(set(data.pid.tolist())),
        "multi_day_people": len(data.multi),
        "steps": step,
        "seconds": round(time.time() - start, 1),
        "last": history[-1] if history else None,
        "history": history,
    }
    log(
        f"  pretrain {obj}: {n} sessions / {info['people']} people, {step} steps "
        f"in {info['seconds']} s, last {info['last']}"
    )
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return state, info


def pretrained(pids, cfg: dict, device: str, root: Path = VIDEO, cache=None):
    """(encoder state dict, info) pretrained on the video EEG of ``pids``;
    memoised on disk (shared by the processes of a job)."""
    import torch

    cfg = config(**cfg)
    folder = Path(cache or CACHE / "pretrain")
    folder.mkdir(parents=True, exist_ok=True)
    key = _key(pids, cfg, root)
    target = folder / f"{cfg['objective']}_{key}.pt"
    with open(folder / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_file():
            saved = torch.load(target, map_location="cpu", weights_only=False)
            return saved["state"], saved["info"] | {"memo": True}
        data = _Data(pids, root, EEGCrops())
        if not data.pools:
            raise ValueError(f"no video EEG for these people under {root}")
        if cfg["objective"] == "xday" and len(data.multi) < 4:
            raise ValueError("xday needs people with video on two days")
        state, info = _train(data, cfg, device)
        torch.save({"state": state, "info": info, "config": cfg}, target)
        return state, info
