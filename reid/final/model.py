"""A single-modality encoder trained by cross-day contrast, as a ``ReIDModel``.

Training (``fit``), per fold, on the training people only:
* the same inner validation people as the LDA (20 % of the people with days 1
  and 2, same seed) serve for early stopping on their day 1 -> day 2 top-1;
* each step draws ``people`` people with two days x ``crops`` random crops per
  day, plus ``negatives`` single-day people x ``crops`` crops (they have no
  cross-day positive, but are useful negatives);
* supervised contrastive loss whose positives are the same person on *another*
  day only (same-day pairs are left out: they would teach the session);
* AdamW, warm-up then cosine decay, best validation state kept.

Embedding: mean of the L2-normalised embeddings of every crop of the session
(see ``reid.final.crops``), re-normalised; scores are cosine similarities.
"""

from __future__ import annotations

import math
import time

import numpy as np

from reid.baselines.b5_siamese import supcon_loss
from reid.config import SEED
from reid.eval.interface import cosine
from reid.eval.matching import rank_counts, topk, truth_index
from reid.eval.protocol import SessionSet
from reid.final.crops import CROPS
from reid.final.encoders import ENCODERS, augment

REST = ("rest_ec", "rest_eo")


def _device(name: str) -> str:
    """``auto``: CUDA (HABS H100), else Apple MPS, else CPU."""
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def inner_split(train: SessionSet, seed: int):
    """(inner training set, validation gallery day 1, validation probe day 2):
    identical to the LDA's inner split for the same seed."""
    df = train.sessions
    both = df.groupby("participant_id").day.agg(lambda d: {1, 2} <= set(d))
    pool = np.array(sorted(both.index[both]))
    rng = np.random.default_rng(seed)
    val = set(rng.choice(pool, max(len(pool) // 5, 2), replace=False))
    in_val = df.participant_id.isin(val)
    return (
        train.subset(~in_val.to_numpy()),
        train.subset((in_val & (df.day == 1)).to_numpy()),
        train.subset((in_val & (df.day == 2)).to_numpy()),
    )


class ContrastiveModel:
    def __init__(
        self,
        modality: str = "eeg",
        conditions=REST,
        seed: int = SEED,
        steps: int = 3000,
        eval_every: int = 100,
        patience: int = 800,
        people: int = 32,
        crops: int = 4,
        negatives: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        temperature: float = 0.1,
        warmup: int = 100,
        device: str = "auto",
        crop_kw: dict | None = None,
        encoder_kw: dict | None = None,
        tag: str = "",
        pool_crops: bool = False,
    ) -> None:
        self.modality, self.conditions, self.seed = modality, tuple(conditions), seed
        self.steps, self.eval_every, self.patience = steps, eval_every, patience
        self.people, self.crops, self.negatives = people, crops, negatives
        self.lr, self.weight_decay, self.temperature = lr, weight_decay, temperature
        self.warmup, self.device = warmup, _device(device)
        self.pool_crops = pool_crops
        self.crop_kw, self.encoder_kw = crop_kw or {}, encoder_kw or {}
        self.cropper = CROPS[modality](**self.crop_kw)
        self.name = f"f_{modality}{tag}_s{seed - SEED}"
        self.config = {
            "modality": modality,
            "conditions": list(self.conditions),
            "seed": seed,
            "steps": steps,
            "people": people,
            "crops": crops,
            "negatives": negatives,
            "lr": lr,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "pool_crops": pool_crops,
            "device": self.device,
            "crop_kw": self.crop_kw,
            "encoder_kw": self.encoder_kw,
        }

    # -- data -----------------------------------------------------------------
    def _pools(self, sessions: SessionSet) -> list:
        if (
            sessions.sessions.get("part") is not None
            and sessions.sessions.part.notna().any()
        ):
            raise ValueError("window selection (part) is not supported on raw crops")
        return [self.cropper.prepare(f, self.conditions) for f in sessions]

    def _batch(self, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crops, person codes and days of one training batch."""
        chosen = rng.choice(
            len(self.multi), min(self.people, len(self.multi)), replace=False
        )
        x, person, day = [], [], []
        for p in chosen:
            sessions = self.multi[p]
            for s in rng.choice(len(sessions), 2, replace=False):
                i = sessions[s]
                x.append(self.cropper.sample(self.pool[i], self.crops, rng))
                person += [p] * self.crops
                day += [self.day[i]] * self.crops
        if self.single and self.negatives:
            extra = rng.choice(
                len(self.single), min(self.negatives, len(self.single)), replace=False
            )
            for k, j in enumerate(extra):
                i = self.single[j]
                x.append(self.cropper.sample(self.pool[i], self.crops, rng))
                person += [len(self.multi) + k] * self.crops
                day += [self.day[i]] * self.crops
        return np.concatenate(x), np.array(person), np.array(day)

    # -- model ----------------------------------------------------------------
    def _encode(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(X), batch):
                x = torch.as_tensor(X[k : k + batch], device=self.device)
                out.append(F.normalize(self.net(x), dim=1).cpu().numpy())
        return np.concatenate(out)

    def _embed_pools(self, pools: list) -> np.ndarray:
        dim = self.encoder_kw.get("out", 64)
        out = []
        for pool in pools:
            if pool is None:
                out.append(None)
                continue
            z = self._encode(self.cropper.all_crops(pool)).mean(0)
            out.append(z / max(np.linalg.norm(z), 1e-12))
        return np.array([z if z is not None else np.full(dim, np.nan) for z in out])

    def embed(self, sessions: SessionSet) -> np.ndarray:
        return self._embed_pools(self._pools(sessions))

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        return cosine(self.embed(probe), self.embed(gallery))

    def _val_top1(self) -> float:
        S = cosine(self._embed_pools(self.val_prb), self._embed_pools(self.val_gal))
        return float(topk(*rank_counts(S, self.val_truth), 1).mean())

    def fit(self, train: SessionSet) -> None:
        import torch
        import torch.nn.functional as F

        start = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        inner, gal, prb = inner_split(train, self.seed)
        self.pool = self._pools(inner)
        self.val_gal, self.val_prb = self._pools(gal), self._pools(prb)
        self.val_truth = truth_index(gal.pids, prb.pids)

        ok = np.array([p is not None for p in self.pool])
        self.day = inner.sessions.day.to_numpy()
        groups = [
            idx[ok[idx]]
            for idx in inner.sessions.groupby("participant_id").indices.values()
        ]
        self.multi = [i for i in groups if len(np.unique(self.day[i])) >= 2]
        self.single = [i[0] for i in groups if len(np.unique(self.day[i])) == 1]

        self.net = ENCODERS[self.modality](**self.encoder_kw).to(self.device)
        if hasattr(self.net, "set_normalisation"):
            X, _, _ = self._batch(rng)
            with torch.no_grad():
                self.net.set_normalisation(
                    self.net.tokens(torch.as_tensor(X, device=self.device))
                )
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: (
                min(1.0, (s + 1) / self.warmup)
                * 0.5
                * (1 + math.cos(math.pi * min(s, self.steps) / self.steps))
            ),
        )
        best, best_state, best_step, history = -1.0, None, 0, []
        for step in range(1, self.steps + 1):
            self.net.train()
            X, person, day = self._batch(rng)
            x = augment(self.modality, torch.as_tensor(X, device=self.device))
            z = F.normalize(self.net(x), dim=1)
            if self.pool_crops:  # one unit = mean of a (person, day)'s crops
                z = F.normalize(z.reshape(-1, self.crops, z.shape[1]).mean(1), dim=1)
                person, day = person[:: self.crops], day[:: self.crops]
            loss = supcon_loss(
                z,
                torch.as_tensor(person, device=self.device),
                torch.as_tensor(day, device=self.device),
                self.temperature,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % self.eval_every == 0:
                acc = self._val_top1()
                history.append((step, round(loss.item(), 4), round(acc, 4)))
                if acc > best:
                    best, best_step = acc, step
                    best_state = {
                        k: v.clone() for k, v in self.net.state_dict().items()
                    }
                elif step - best_step >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        del self.pool, self.val_gal, self.val_prb
        print(
            f"  {self.name}: val top-1 {best:.3f} at step {best_step} "
            f"(stopped {step}, {time.time() - start:.0f} s)",
            flush=True,
        )
        self.config |= {
            "val_top1": best,
            "best_step": best_step,
            "stopped_step": step,
            "history": history,
            "fit_s": round(time.time() - start, 1),
        }
