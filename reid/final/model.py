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

After ``fit``, ``val_scores_`` / ``val_pids_`` hold the early-stopped network's
scores on the inner validation people (used by ``reid.final.fusion``).

``pretrain`` (options of ``reid.final.pretrain``): the network starts from an
encoder pretrained on the video EEG of the *inner* training people of the
fold (never the test or inner validation people); input normalisation is
then recomputed on rest crops. ``val_top1_step0`` is the inner validation
top-1 of the network before any fine-tuning (pretrained, or random: the
reference for what pretraining alone brings).

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


def inner_split(train: SessionSet, seed: int, val_people=None):
    """(inner training set, validation gallery day 1, validation probe day 2):
    identical to the LDA's inner split for the same seed. ``val_people``
    imposes the validation people instead (those of another task's split, so
    that members of several tasks share one validation: ``TaskView``)."""
    df = train.sessions
    both = df.groupby("participant_id").day.agg(lambda d: {1, 2} <= set(d))
    pool = np.array(sorted(both.index[both]))
    if val_people is not None:
        val = set(pool) & set(val_people)
    else:
        rng = np.random.default_rng(seed)
        val = set(rng.choice(pool, max(len(pool) // 5, 2), replace=False))
    in_val = df.participant_id.isin(val)
    return (
        train.subset(~in_val.to_numpy()),
        train.subset((in_val & (df.day == 1)).to_numpy()),
        train.subset((in_val & (df.day == 2)).to_numpy()),
    )


_POOLS: dict = {}


def _pool(cropper, conditions, sessions: SessionSet, i: int):
    """Crop pool of session ``i``, memoised per process (finding the clean
    crops of a 14 min video session takes ~0.5 s, x 400 sessions x 5 folds)."""
    row = sessions.sessions.iloc[i]
    root = getattr(sessions.store, "root", None)
    key = (
        str(root),
        row.participant_id,
        int(row.day),
        row.task,
        tuple(conditions),
        type(cropper).__name__,
        tuple(sorted(vars(cropper).items())),
    )
    if root is None:
        return cropper.prepare(sessions.load(i), conditions)
    if key not in _POOLS:
        _POOLS[key] = cropper.prepare(sessions.load(i), conditions)
    return _POOLS[key]


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
        refit: bool = False,
        split_seed: int | None = None,
        pretrain: dict | None = None,
    ) -> None:
        self.modality, self.conditions, self.seed = modality, tuple(conditions), seed
        self.steps, self.eval_every, self.patience = steps, eval_every, patience
        self.people, self.crops, self.negatives = people, crops, negatives
        self.lr, self.weight_decay, self.temperature = lr, weight_decay, temperature
        self.warmup, self.device = warmup, _device(device)
        self.pool_crops, self.refit = pool_crops, refit
        self.split_seed = seed if split_seed is None else split_seed
        self.pretrain, self.init_state = pretrain, None
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
            "refit": refit,
            "split_seed": self.split_seed,
            "pretrain": pretrain,
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
        return [
            _pool(self.cropper, self.conditions, sessions, i)
            for i in range(len(sessions))
        ]

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

    def _prepare(self, sessions: SessionSet) -> None:
        """Crop pools of the training sessions and their grouping by person."""
        self.pool = self._pools(sessions)
        ok = np.array([p is not None for p in self.pool])
        self.day = sessions.sessions.day.to_numpy()
        groups = [
            idx[ok[idx]]
            for idx in sessions.sessions.groupby("participant_id").indices.values()
        ]
        self.multi = [i for i in groups if len(np.unique(self.day[i])) >= 2]
        self.single = [i[0] for i in groups if len(np.unique(self.day[i])) == 1]

    def _train(self, rng, n_steps: int, validate: bool) -> tuple:
        """Fresh network trained for up to ``n_steps`` on the prepared pools.
        With ``validate``: early stopping on the inner validation people, best
        state restored. Returns (best val top-1, best step, last step, history)."""
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        self.net = ENCODERS[self.modality](**self.encoder_kw).to(self.device)
        if self.init_state is not None:
            self.net.load_state_dict(self.init_state)
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
        best, best_state, best_step, history, step = -1.0, None, 0, [], 0
        if validate:  # before any fine-tuning (random or pretrained network)
            self.val_top1_step0 = self._val_top1()
            history.append((0, None, round(self.val_top1_step0, 4)))
        for step in range(1, n_steps + 1):
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
            if validate and step % self.eval_every == 0:
                acc = self._val_top1()
                history.append((step, round(loss.item(), 4), round(acc, 4)))
                if acc > best:
                    best, best_step = acc, step
                    best_state = {
                        k: v.clone() for k, v in self.net.state_dict().items()
                    }
                elif step - best_step >= self.patience:
                    break
        if validate:
            self.net.load_state_dict(best_state)
        return best, best_step, step, history

    def fit(self, train: SessionSet) -> None:
        """Early stopping on the inner split; with ``refit``, a fresh network is
        then trained on *all* training people for the chosen number of steps
        (as the LDA is refitted on everyone once its dimension is chosen)."""
        start = time.time()
        rng = np.random.default_rng(self.seed)
        inner, gal, prb = inner_split(
            train, self.split_seed, getattr(self, "val_people", None)
        )
        if self.pretrain:
            from reid.final.pretrain import pretrained

            pids = sorted(set(inner.sessions.participant_id))
            self.init_state, info = pretrained(pids, self.pretrain, self.device)
            self.config |= {
                f"pretrain_{k}": v
                for k, v in info.items()
                if k in ("sessions", "people", "steps", "seconds", "memo")
            }
        self._prepare(inner)
        self.val_gal, self.val_prb = self._pools(gal), self._pools(prb)
        self.val_truth = truth_index(gal.pids, prb.pids)
        best, best_step, step, history = self._train(rng, self.steps, validate=True)
        # inner-validation scores of the early-stopped network (score fusion)
        self.val_scores_ = cosine(
            self._embed_pools(self.val_prb), self._embed_pools(self.val_gal)
        )
        self.val_pids_ = (gal.pids, prb.pids)
        del self.val_gal, self.val_prb
        if self.refit:
            self._prepare(train)
            self._train(rng, best_step, validate=False)
        del self.pool
        print(
            f"  {self.name}: val top-1 {best:.3f} at step {best_step} "
            f"(stopped {step}{', refitted on all' if self.refit else ''}, "
            f"{time.time() - start:.0f} s)",
            flush=True,
        )
        self.config |= {
            "val_top1": best,
            "best_step": best_step,
            "stopped_step": step,
            "history": history,
            "fit_s": round(time.time() - start, 1),
        }
        self.config["val_top1_step0"] = round(self.val_top1_step0, 4)
