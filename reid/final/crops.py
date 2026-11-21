"""Crops of the raw cache, one ``Modality`` per encoder.

A modality turns a session's raw dict into a *pool* (prepared once), then draws
random crops from it for training (``sample``) or every crop on a fixed grid for
embedding (``all_crops``). Crops are float32 arrays of a fixed shape, so that
any number of sessions can be batched.

Windows (defaults):
* EEG: 4 s crops (512 samples at 128 Hz), random start on a 0.25 s grid; a crop
  is kept only if every 2 s sub-epoch passes the session's peak-to-peak
  threshold. Embedding: one crop every 2 s.
* ECG: sets of 16 consecutive beats (~13 s), each beat z-normalised (shape only:
  units and electrode gain vary between days); beats far from the session's
  median beat (correlation < 0.5: ectopic or noisy) are dropped. Embedding: one
  set every 8 beats.
* Respiration: 20 s crops (500 samples at 25 Hz, 4-8 breaths), z-normalised.
  Embedding: one crop every 5 s.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from reid.final.raw import EEG_RATE, RESP_RATE
from reid.preprocess import peak_to_peak


def _segments(feats: dict, prefix: str, conditions) -> list[np.ndarray]:
    return [
        feats[f"{prefix}_{k}"]
        for k, c in enumerate(feats["seg_condition"])
        if c in conditions and f"{prefix}_{k}" in feats
    ]


class EEGCrops:
    name = "eeg"

    def __init__(self, crop_s: float = 4.0, stride_s: float = 2.0, grid_s=0.25):
        self.crop_s, self.stride_s, self.grid_s = crop_s, stride_s, grid_s

    def prepare(self, feats: dict, conditions) -> list | None:
        """[(segment, valid crop starts)], starts in samples."""
        rate = EEG_RATE
        size, grid = round(self.crop_s * rate), round(self.grid_s * rate)
        epoch = 2 * rate
        pool = []
        for seg in _segments(feats, "eeg", conditions):
            if seg.shape[-1] < size:
                continue
            ep = sliding_window_view(seg, epoch, axis=-1)[:, ::grid]  # [C, nE, 2 s]
            ok = peak_to_peak(ep.transpose(1, 0, 2)) <= feats["eeg_reject_uv"]
            span = (size - epoch) // grid + 1  # sub-epochs per crop
            starts = np.arange(0, seg.shape[-1] - size + 1, grid)
            cum = np.concatenate([[0], np.cumsum(~ok)])
            j = starts // grid
            n_bad = cum[np.minimum(j + span, len(ok))] - cum[j]
            # no clean crop at all: keep the least contaminated quarter
            valid = starts[
                n_bad <= (0 if (n_bad == 0).any() else np.quantile(n_bad, 0.25))
            ]
            pool.append((seg, valid))
        return pool or None

    def sample(self, pool: list, n: int, rng) -> np.ndarray:
        size = round(self.crop_s * EEG_RATE)
        out = []
        for _ in range(n):
            seg, starts = pool[rng.integers(len(pool))]
            s = rng.choice(starts)
            out.append(seg[:, s : s + size])
        return np.asarray(out, np.float32)

    def all_crops(self, pool: list) -> np.ndarray:
        size = round(self.crop_s * EEG_RATE)
        stride = round(self.stride_s * EEG_RATE)
        out = []
        for seg, starts in pool:
            grid = np.arange(0, seg.shape[-1] - size + 1, stride)
            keep = np.intersect1d(grid, starts)
            for s in keep if len(keep) else grid:
                out.append(seg[:, s : s + size])
        return np.asarray(out, np.float32)


class BeatCrops:
    name = "ecg_beat"

    def __init__(self, n_beats: int = 16, stride: int = 8, min_corr: float = 0.5):
        self.n_beats, self.stride, self.min_corr = n_beats, stride, min_corr

    def prepare(self, feats: dict, conditions) -> np.ndarray | None:
        segs = [b for b in _segments(feats, "beats", conditions) if len(b)]
        if not segs:
            return None
        beats = np.concatenate(segs).astype(np.float64)
        beats = (beats - beats.mean(1, keepdims=True)) / (
            beats.std(1, keepdims=True) + 1e-9
        )
        template = np.median(beats, axis=0)
        corr = (
            beats
            @ (template - template.mean())
            / (len(template) * (template.std() + 1e-9))
        )
        beats = beats[corr >= self.min_corr]
        return beats.astype(np.float32) if len(beats) >= 4 else None

    def _set(self, beats: np.ndarray, start: int) -> np.ndarray:
        idx = (start + np.arange(self.n_beats)) % len(beats)  # wraps if too few
        return beats[idx]

    def sample(self, pool: np.ndarray, n: int, rng) -> np.ndarray:
        starts = rng.integers(len(pool), size=n)
        return np.asarray([self._set(pool, s) for s in starts], np.float32)

    def all_crops(self, pool: np.ndarray) -> np.ndarray:
        last = max(len(pool) - self.n_beats, 0)
        starts = range(0, last + 1, self.stride)
        return np.asarray([self._set(pool, s) for s in starts], np.float32)


class RespCrops:
    name = "resp"

    def __init__(self, crop_s: float = 20.0, stride_s: float = 5.0):
        self.size = round(crop_s * RESP_RATE)
        self.stride = round(stride_s * RESP_RATE)

    def prepare(self, feats: dict, conditions) -> list | None:
        segs = [
            s
            for s in _segments(feats, "resp", conditions)
            if len(s) >= self.size and np.isfinite(s).all() and s.std() > 0
        ]
        return segs or None

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9)

    def sample(self, pool: list, n: int, rng) -> np.ndarray:
        out = []
        for _ in range(n):
            seg = pool[rng.integers(len(pool))]
            s = rng.integers(len(seg) - self.size + 1)
            out.append(seg[s : s + self.size])
        return self._norm(np.asarray(out, np.float32))

    def all_crops(self, pool: list) -> np.ndarray:
        out = [
            seg[s : s + self.size]
            for seg in pool
            for s in range(0, len(seg) - self.size + 1, self.stride)
        ]
        return self._norm(np.asarray(out, np.float32))


CROPS = {"eeg": EEGCrops, "ecg_beat": BeatCrops, "resp": RespCrops}
