"""One model for the three signals: the EEG, ECG-beat and respiration encoders
and a fusion head trained together, end to end, as a ``ReIDModel``.

Unlike score fusion (``reid.final.fusion``: separately trained members whose
scores are added), the joint model learns *one* embedding of the person:

* unit of the loss = one (person, day): the mean of ``crops[m]`` normalised
  crop embeddings of each modality m (as the session-level EEG encoder);
* fused embedding = MLP(concatenated modality means, presence flags), so a
  session without ECG or respiration still gets an embedding;
* loss = SupCon on the fused embedding (positives: same person, another day)
  + ``branch_weight`` x the same loss on each modality alone (keeps every
  encoder useful by itself);
* modality dropout: each available modality of a unit is hidden from the head
  with probability ``drop_p`` (at least one kept), so that the head does not
  rely on one signal only and tolerates missing ones;
* same inner split, early stopping and refit on all training people as
  ``reid.final.model.ContrastiveModel``; ``eeg_pretrain`` starts the EEG
  branch from a video-pretrained encoder (``reid.final.pretrain``).

Embedding of a session: per modality, mean of the normalised embeddings of all
its crops, normalised; then the head. Scores are cosine similarities.
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
from reid.final.model import REST, _device, inner_split

MODALITIES = ("eeg", "ecg_beat", "resp")


class JointModel:
    def __init__(
        self,
        seed: int = SEED,
        split_seed: int = SEED,
        conditions=REST,
        steps: int = 3000,
        eval_every: int = 50,
        patience: int = 800,
        people: int = 64,
        negatives: int = 16,
        crops: dict | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        temperature: float = 0.1,
        warmup: int = 100,
        branch_weight: float = 0.5,
        drop_p: float = 0.2,
        dim: int = 64,
        refit: bool = True,
        eeg_pretrain: dict | None = None,
        encoder_kw: dict | None = None,
        crop_kw: dict | None = None,
        device: str = "auto",
        tag: str = "",
    ) -> None:
        self.seed, self.split_seed, self.conditions = (
            seed,
            split_seed,
            tuple(conditions),
        )
        self.steps, self.eval_every, self.patience = steps, eval_every, patience
        self.people, self.negatives = people, negatives
        self.k = {"eeg": 8, "ecg_beat": 4, "resp": 4} | (crops or {})
        self.lr, self.weight_decay, self.temperature = lr, weight_decay, temperature
        self.warmup, self.branch_weight, self.drop_p = warmup, branch_weight, drop_p
        self.dim, self.refit, self.eeg_pretrain = dim, refit, eeg_pretrain
        self.device = _device(device)
        self.encoder_kw = {m: {} for m in MODALITIES} | (encoder_kw or {})
        self.crop_kw = {m: {} for m in MODALITIES} | (crop_kw or {})
        self.croppers = {m: CROPS[m](**self.crop_kw[m]) for m in MODALITIES}
        self.init_state = None
        self.name = f"j_all3{tag}_s{seed - SEED}"
        self.config = {
            "seed": seed,
            "split_seed": split_seed,
            "conditions": list(self.conditions),
            "steps": steps,
            "people": people,
            "negatives": negatives,
            "crops": self.k,
            "lr": lr,
            "temperature": temperature,
            "branch_weight": branch_weight,
            "drop_p": drop_p,
            "refit": refit,
            "eeg_pretrain": eeg_pretrain,
            "device": self.device,
        }

    # -- data -----------------------------------------------------------------
    def _pools(self, sessions: SessionSet) -> list[dict]:
        out = []
        for f in sessions:
            out.append(
                {m: self.croppers[m].prepare(f, self.conditions) for m in MODALITIES}
            )
        return out

    def _prepare(self, sessions: SessionSet) -> None:
        self.pool = self._pools(sessions)
        ok = np.array([any(p[m] is not None for m in MODALITIES) for p in self.pool])
        self.day = sessions.sessions.day.to_numpy()
        groups = [
            idx[ok[idx]]
            for idx in sessions.sessions.groupby("participant_id").indices.values()
        ]
        self.multi = [i for i in groups if len(np.unique(self.day[i])) >= 2]
        self.single = [
            i[0] for i in groups if len(i) and len(np.unique(self.day[i])) == 1
        ]

    def _batch(self, rng):
        """Units (person, day): crops per modality, unit index per crop group,
        presence [U, 3], person and day codes."""
        units = []
        chosen = rng.choice(
            len(self.multi), min(self.people, len(self.multi)), replace=False
        )
        for p in chosen:
            sessions = self.multi[p]
            days = self.day[sessions]
            for d in rng.choice(np.unique(days), 2, replace=False):
                units.append((int(rng.choice(sessions[days == d])), p, int(d)))
        if self.single and self.negatives:
            extra = rng.choice(
                len(self.single), min(self.negatives, len(self.single)), replace=False
            )
            for k, j in enumerate(extra):
                i = self.single[j]
                units.append((i, len(self.multi) + k, int(self.day[i])))
        x = {m: [] for m in MODALITIES}
        where = {m: [] for m in MODALITIES}
        present = np.zeros((len(units), len(MODALITIES)), bool)
        for u, (i, _, _) in enumerate(units):
            for j, m in enumerate(MODALITIES):
                pool = self.pool[i][m]
                if pool is None:
                    continue
                x[m].append(self.croppers[m].sample(pool, self.k[m], rng))
                where[m].append(u)
                present[u, j] = True
        person = np.array([u[1] for u in units])
        day = np.array([u[2] for u in units])
        return x, where, present, person, day

    # -- model ----------------------------------------------------------------
    def _build(self):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        n = len(MODALITIES)
        net = nn.ModuleDict(
            {
                "enc": nn.ModuleDict(
                    {m: ENCODERS[m](**self.encoder_kw[m]) for m in MODALITIES}
                ),
                "head": nn.Sequential(
                    nn.Linear(n * self.dim + n, 2 * self.dim),
                    nn.GELU(),
                    nn.Linear(2 * self.dim, self.dim),
                ),
            }
        ).to(self.device)
        if self.init_state is not None:
            net["enc"]["eeg"].load_state_dict(self.init_state)
        return net

    def _unit_means(self, m: str, crops: np.ndarray, k: int, train: bool):
        """[n_units, dim]: normalised mean of the k normalised crop embeddings."""
        import torch
        import torch.nn.functional as F

        x = torch.as_tensor(crops, device=self.device)
        if train:
            x = augment(m, x)
        z = F.normalize(self.net["enc"][m](x), dim=1)
        return F.normalize(z.reshape(-1, k, z.shape[1]).mean(1), dim=1)

    def _fuse(self, z: dict, present):
        import torch
        import torch.nn.functional as F

        p = torch.as_tensor(present, device=self.device, dtype=torch.float32)
        parts = [z[m] * p[:, j : j + 1] for j, m in enumerate(MODALITIES)]
        return F.normalize(self.net["head"](torch.cat([*parts, p], 1)), dim=1)

    def _embed_pools(self, pools: list[dict], batch: int = 512) -> np.ndarray:
        """Session embeddings (eval mode, all crops)."""
        import torch
        import torch.nn.functional as F

        self.net.eval()
        n = len(pools)
        present = np.array([[p[m] is not None for m in MODALITIES] for p in pools])
        z = {}
        with torch.no_grad():
            for j, m in enumerate(MODALITIES):
                out = torch.zeros(n, self.dim, device=self.device)
                for i, p in enumerate(pools):
                    if not present[i, j]:
                        continue
                    crops = self.croppers[m].all_crops(p[m])
                    if not len(crops):
                        present[i, j] = False
                        continue
                    e = [
                        F.normalize(
                            self.net["enc"][m](
                                torch.as_tensor(
                                    crops[a : a + batch], device=self.device
                                )
                            ),
                            dim=1,
                        )
                        for a in range(0, len(crops), batch)
                    ]
                    out[i] = F.normalize(torch.cat(e).mean(0), dim=0)
                z[m] = out
            fused = self._fuse(z, present).cpu().numpy()
        fused[~present.any(1)] = np.nan
        return fused

    def embed(self, sessions: SessionSet) -> np.ndarray:
        return self._embed_pools(self._pools(sessions))

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        return cosine(self.embed(probe), self.embed(gallery))

    def _val_top1(self) -> float:
        S = cosine(self._embed_pools(self.val_prb), self._embed_pools(self.val_gal))
        return float(topk(*rank_counts(S, self.val_truth), 1).mean())

    def _train(self, rng, n_steps: int, validate: bool):
        import torch

        self.net = self._build()
        eeg = self.net["enc"]["eeg"]
        if hasattr(eeg, "set_normalisation"):  # input statistics from rest crops
            x, _, _, _, _ = self._batch(rng)
            with torch.no_grad():
                eeg.set_normalisation(
                    eeg.tokens(
                        torch.as_tensor(np.concatenate(x["eeg"]), device=self.device)
                    )
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
        if validate:
            self.val_top1_step0 = self._val_top1()
            history.append((0, None, round(self.val_top1_step0, 4)))
        for step in range(1, n_steps + 1):
            self.net.train()
            x, where, present, person, day = self._batch(rng)
            z = {}
            person_t = torch.as_tensor(person, device=self.device)
            day_t = torch.as_tensor(day, device=self.device)
            branch = []
            for m in MODALITIES:
                full = torch.zeros(len(person), self.dim, device=self.device)
                if where[m]:
                    zm = self._unit_means(m, np.concatenate(x[m]), self.k[m], True)
                    idx = torch.as_tensor(where[m], device=self.device)
                    full = full.index_copy(0, idx, zm)
                    branch.append(
                        supcon_loss(zm, person_t[idx], day_t[idx], self.temperature)
                    )
                z[m] = full
            seen = present.copy()  # modality dropout, at least one modality kept
            drop = (rng.random(seen.shape) < self.drop_p) & seen
            keep_one = drop.sum(1) == seen.sum(1)
            drop[keep_one] = False
            seen &= ~drop
            fused = self._fuse(z, seen)
            loss = supcon_loss(fused, person_t, day_t, self.temperature)
            branch = [b for b in branch if torch.isfinite(b)]
            if branch:
                loss = loss + self.branch_weight * torch.stack(branch).mean()
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
        if validate and best_state is not None:
            self.net.load_state_dict(best_state)
        return best, best_step, step, history

    def fit(self, train: SessionSet) -> None:
        start = time.time()
        rng = np.random.default_rng(self.seed)
        inner, gal, prb = inner_split(train, self.split_seed)
        if self.eeg_pretrain:
            from reid.final.pretrain import pretrained

            pids = sorted(set(inner.sessions.participant_id))
            self.init_state, info = pretrained(pids, self.eeg_pretrain, self.device)
            self.config["pretrain_steps"] = info.get("steps")
        self._prepare(inner)
        self.val_gal, self.val_prb = self._pools(gal), self._pools(prb)
        self.val_truth = truth_index(gal.pids, prb.pids)
        best, best_step, step, history = self._train(rng, self.steps, validate=True)
        self.val_scores_ = cosine(
            self._embed_pools(self.val_prb), self._embed_pools(self.val_gal)
        )
        self.val_pids_ = (gal.pids, prb.pids)
        del self.val_gal, self.val_prb
        if self.refit and best_step > 0:
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
            "val_top1_step0": round(self.val_top1_step0, 4),
            "best_step": best_step,
            "stopped_step": step,
            "history": history,
            "fit_s": round(time.time() - start, 1),
        }
