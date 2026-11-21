"""Baseline 5 (optional): small siamese MLP trained with a supervised
contrastive (SupCon) loss.

Same idea as the LDA, but non-linear: an MLP (256 -> 128 -> 64, dropout, L2
output) maps a window's standardised features to an embedding. Positive pairs
are windows of the same person from *different days* only (two windows of the
same day would teach the session, not the person); same-person same-day pairs
are left out of the loss. Augmentations mimic what changes between sessions:
Gaussian noise, masked modality blocks (a missing sensor), per-electrode offsets
(impedance). Early stopping on the same inner validation people as the LDA.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reid.baselines.b3_lda_plda import ALL
from reid.baselines.common import stack_windows, window_matrix
from reid.config import SEED
from reid.eval.interface import cosine
from reid.eval.matching import rank_counts, topk, truth_index
from reid.eval.protocol import SessionSet


def supcon_loss(z, person, day, temperature: float = 0.1):
    """SupCon with positives = same person, different day. ``z`` is L2-normalised."""
    import torch

    sim = z @ z.T / temperature
    same = person[:, None] == person[None]
    pos = same & (day[:, None] != day[None])
    logits = sim.masked_fill(same & ~pos, float("-inf"))  # self and same-day pairs out
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    has_pos = pos.any(1)
    per_anchor = -(log_prob.masked_fill(~pos, 0).sum(1)) / pos.sum(1).clamp(min=1)
    return per_anchor[has_pos].mean()


class SiameseModel:
    def __init__(
        self,
        modalities=ALL,
        conditions=("rest_ec", "rest_eo"),
        eeg_kind: str = "rel",
        seed: int = SEED,
        epochs: int = 300,
        people_per_batch: int = 32,
        windows_per_day: int = 4,
        patience: int = 30,
        lr: float = 1e-3,
    ) -> None:
        self.feat = {
            "modalities": modalities,
            "conditions": conditions,
            "eeg_kind": eeg_kind,
        }
        self.seed, self.epochs, self.patience, self.lr = seed, epochs, patience, lr
        self.people_per_batch, self.windows_per_day = people_per_batch, windows_per_day
        self.name = f"b5_siamese_s{seed - SEED}"
        self.config = {
            **self.feat,
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
            "people_per_batch": people_per_batch,
            "patience": patience,
        }

    # -- data ---------------------------------------------------------------
    def _blocks(self, feats: dict) -> list[slice]:
        """Column slice of each modality block (used by the masking augmentation)."""
        sizes = [
            window_matrix(
                feats, (m,), self.feat["conditions"], self.feat["eeg_kind"]
            ).shape[1]
            for m in self.feat["modalities"]
        ]
        edges = np.cumsum([0, *sizes])
        return [slice(a, b) for a, b in itertools.pairwise(edges)]

    def _augment(self, x, rng):
        import torch

        x = x + 0.1 * torch.randn_like(x)
        for block in self.blocks[1:] if len(self.blocks) > 1 else []:
            drop = torch.rand(len(x), 1) < 0.2
            x[:, block] = x[:, block].masked_fill(drop, 0.0)
        if "eeg" in self.feat["modalities"]:  # per-electrode offset ~ impedance
            eeg = self.blocks[0]
            n_ch = self.n_ch
            offset = 0.2 * torch.randn(len(x), n_ch, 1)
            width = (eeg.stop - eeg.start) // n_ch
            x[:, eeg] = (x[:, eeg].reshape(len(x), n_ch, width) + offset).reshape(
                len(x), -1
            )
        return x

    def _batch(self, person, day, rng) -> np.ndarray:
        """``people_per_batch`` people x 2 days x ``windows_per_day`` windows."""
        chosen = rng.choice(
            self.trainable,
            min(self.people_per_batch, len(self.trainable)),
            replace=False,
        )
        rows = []
        for p in chosen:
            days = rng.choice(np.unique(day[person == p]), 2, replace=False)
            for d in days:
                idx = np.flatnonzero((person == p) & (day == d))
                rows.extend(
                    rng.choice(
                        idx,
                        self.windows_per_day,
                        replace=len(idx) < self.windows_per_day,
                    )
                )
        return np.array(rows)

    # -- model ----------------------------------------------------------------
    def _net(self, d: int):
        from torch import nn

        return nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
        )

    def _forward(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.net.eval()
        with torch.no_grad():
            return F.normalize(
                self.net(torch.as_tensor(X, dtype=torch.float32)), dim=1
            ).numpy()

    def embed(self, sessions: SessionSet) -> np.ndarray:
        out = []
        for f in sessions:
            W = window_matrix(f, **self.feat)
            out.append(
                self._forward(self.prep.transform(W)).mean(0)
                if len(W)
                else np.full(64, np.nan)
            )
        return np.array(out)

    def _val_top1(self, gal: SessionSet, prb: SessionSet) -> float:
        S = cosine(self.embed(prb), self.embed(gal))
        return float(topk(*rank_counts(S, truth_index(gal.pids, prb.pids)), 1).mean())

    def fit(self, train: SessionSet) -> None:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        first = train.load(0)
        self.blocks, self.n_ch = self._blocks(first), first["w_eeg_psd"].shape[1]

        df = train.sessions  # inner validation people, as for the LDA
        both = df.groupby("participant_id").day.agg(lambda d: {1, 2} <= set(d))
        pool = np.array(sorted(both.index[both]))
        val = set(rng.choice(pool, max(len(pool) // 5, 2), replace=False))
        inner = train.subset(~df.participant_id.isin(val).to_numpy())
        gal = train.subset((df.participant_id.isin(val) & (df.day == 1)).to_numpy())
        prb = train.subset((df.participant_id.isin(val) & (df.day == 2)).to_numpy())

        X, idx = stack_windows(inner, **self.feat)
        self.prep = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler()
        ).fit(X)
        X = torch.as_tensor(self.prep.transform(X), dtype=torch.float32)
        person, day = inner.pids[idx], inner.sessions.day.to_numpy()[idx]
        n_days = inner.sessions.groupby("participant_id").day.nunique()
        self.trainable = np.array(n_days.index[n_days >= 2])
        codes = {p: i for i, p in enumerate(np.unique(person))}

        self.net = self._net(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, best_state, waited = -1.0, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            rows = self._batch(person, day, rng)
            z = F.normalize(self.net(self._augment(X[rows].clone(), rng)), dim=1)
            loss = supcon_loss(
                z,
                torch.as_tensor([codes[p] for p in person[rows]]),
                torch.as_tensor(day[rows]),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 5 == 0:
                acc = self._val_top1(gal, prb)
                if acc > best:
                    best, waited = acc, 0
                    best_state = {
                        k: v.clone() for k, v in self.net.state_dict().items()
                    }
                else:
                    waited += 5
                    if waited >= self.patience:
                        break
        self.net.load_state_dict(best_state)
        self.config["val_top1"], self.config["stopped_epoch"] = best, epoch

    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray:
        return cosine(self.embed(probe), self.embed(gallery))
