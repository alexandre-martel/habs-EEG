"""One encoder per modality: crop -> embedding (not normalised).

* ``SpectralTokenEEG``: Welch log-spectrum of each electrode (fixed front-end,
  1-45 Hz, 0.5 Hz bins: the representation the LDA already exploits), one token
  per electrode (relative spectrum + log total power) plus a learned electrode
  embedding, a small transformer across electrodes, masked mean pooling.
  Random electrode masking during training (a bad electrode on another day).
* ``BeatSetEncoder``: a 1D CNN applied to each beat of a set, mean over beats.
* ``RespEncoder``: a 1D CNN over a respiration crop, mean + max pooling.

All are small (tens of thousands of parameters): there are ~170 training people
per fold with two days, so capacity is not the bottleneck.
"""

from __future__ import annotations

import torch
from torch import nn

from reid.final.raw import EEG_RATE


class SpectralTokenEEG(nn.Module):
    def __init__(
        self,
        n_channels: int = 62,
        dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        out: int = 64,
        fmin: float = 1.0,
        fmax: float = 45.0,
        seg: int = 2 * EEG_RATE,
        mask_p: float = 0.1,
        total_power: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seg, self.mask_p = seg, mask_p
        freqs = torch.fft.rfftfreq(seg, 1 / EEG_RATE)
        self.register_buffer(
            "bins", torch.nonzero((freqs >= fmin) & (freqs <= fmax))[:, 0]
        )
        self.register_buffer("window", torch.hann_window(seg))
        self.total_power = total_power
        n_feat = len(self.bins) + int(total_power)
        self.register_buffer("mu", torch.zeros(n_channels, n_feat))
        self.register_buffer("sd", torch.ones(n_channels, n_feat))
        self.proj = nn.Linear(n_feat, dim)
        self.channel = nn.Parameter(0.02 * torch.randn(n_channels, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] µV -> [B, C, n_bins + 1]: log10 relative spectrum and log10
        total power of each electrode (Welch, 2 s Hann segments, 50 % overlap)."""
        segs = x.unfold(-1, self.seg, self.seg // 2) * self.window
        power = torch.fft.rfft(segs, dim=-1).abs().pow(2).mean(-2)[..., self.bins]
        total = power.sum(-1, keepdim=True) + 1e-12
        rel = torch.log10(power / total + 1e-12)
        return torch.cat([rel, torch.log10(total)], -1) if self.total_power else rel

    def set_normalisation(self, tokens: torch.Tensor) -> None:
        """Standardise each (electrode, feature) with training statistics."""
        self.mu.copy_(tokens.mean(0))
        self.sd.copy_(tokens.std(0).clamp_min(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj((self.tokens(x) - self.mu) / self.sd) + self.channel
        keep = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if self.training and self.mask_p > 0:
            keep = torch.rand(h.shape[:2], device=h.device) >= self.mask_p
        h = self.norm(self.body(h, src_key_padding_mask=~keep))
        w = keep.unsqueeze(-1).float()
        return self.head((h * w).sum(1) / w.sum(1).clamp_min(1))


def _conv(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.GELU(),
        nn.MaxPool1d(2),
    )


class BeatSetEncoder(nn.Module):
    def __init__(self, points: int = 128, dim: int = 64, out: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(_conv(1, 16, 7), _conv(16, 32, 5), _conv(32, 64, 5))
        self.beat = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * (points // 8), dim), nn.GELU()
        )
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_beats, points] -> [B, out]."""
        b, n, p = x.shape
        h = self.beat(self.cnn(x.reshape(b * n, 1, p))).reshape(b, n, -1)
        return self.head(h.mean(1))


class RespEncoder(nn.Module):
    def __init__(self, dim: int = 64, out: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _conv(1, 16, 9), _conv(16, 32, 9), _conv(32, 64, 9), _conv(64, dim, 9)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, out]."""
        h = self.cnn(x.unsqueeze(1))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], -1))


def augment(modality: str, x: torch.Tensor) -> torch.Tensor:
    """What changes between days: electrode gains, noise, small time shifts."""
    if modality == "eeg":  # per-electrode gain (impedance), ~10 %
        gain = torch.exp(0.1 * torch.randn(x.shape[0], x.shape[1], 1, device=x.device))
        return x * gain
    if modality == "ecg_beat":  # R-peak jitter and noise
        shift = int(torch.randint(-2, 3, ()).item())
        return torch.roll(x, shift, dims=-1) + 0.05 * torch.randn_like(x)
    if modality == "resp":
        return x + 0.05 * torch.randn_like(x)
    return x


ENCODERS = {"eeg": SpectralTokenEEG, "ecg_beat": BeatSetEncoder, "resp": RespEncoder}
