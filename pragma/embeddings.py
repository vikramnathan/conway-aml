"""Token embeddings, within-field positional encoding, RoPE, and calendar MLP.

Implements the tokenisation-side of the architecture (paper §2.2, §2.3.1):

  x = PosEmb(E(key) + E(value))          (Eq. 1)

where E is a single shared embedding table, PosEmb is a *within-field*
sine/cosine positional encoding (positions index values within a field, not
across fields), and RoPE encodes the temporal coordinates inside attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import PAD_ID, PRAGMAConfig, VocabConfig


class TokenEmbedding(nn.Module):
    """Shared key+value embedding table with within-field positional encoding.

    Given key ids, value ids, and within-field positions, returns
        E(key) + E(value) + PosEmb(field_pos)
    (paper Eq. 1). The same table is used for profile-state and event tokens,
    and is later reused by the MLM head to produce logits (weight tying).
    """

    def __init__(self, vocab: VocabConfig, cfg: PRAGMAConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.table = nn.Embedding(vocab.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        # Static (non-learned) sine/cosine within-field positional encodings.
        pos = self._build_sinusoidal(vocab.max_field_pos, cfg.d_model)
        self.register_buffer("field_pos_emb", pos, persistent=False)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.table.weight[PAD_ID].zero_()

    @staticmethod
    def _build_sinusoidal(n_pos: int, d: int) -> torch.Tensor:
        pe = torch.zeros(n_pos, d)
        position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe  # (n_pos, d)

    def forward(
        self,
        key_ids: torch.Tensor,   # (..., T)
        val_ids: torch.Tensor,   # (..., T)
        field_pos: torch.Tensor,  # (..., T)
    ) -> torch.Tensor:
        x = self.table(key_ids) + self.table(val_ids)
        x = x + self.field_pos_emb[field_pos]
        return self.dropout(x)


class RotaryTimeEmbedding(nn.Module):
    """RoPE driven by a continuous temporal coordinate (not integer position).

    The paper uses RoPE to encode temporal coordinates t (log-seconds): the
    Profile Encoder uses time-since-lifelong-events, and the History Encoder
    uses time-to-last-event. We rotate q/k by angles t * inv_freq, which
    generalises standard RoPE (where the "position" is the time value itself).
    """

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (head_dim/2,)

    def cos_sin(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """t: (B, S) float coordinate -> cos, sin each (B, 1, S, head_dim)."""
        freqs = t[..., None].float() * self.inv_freq  # (B, S, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (B, S, head_dim)
        return emb.cos()[:, None, :, :], emb.sin()[:, None, :, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: (B, H, S, hd). cos/sin: (B, 1, S, hd)."""
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


class CalendarEmbedding(nn.Module):
    """Cyclical calendar features -> d-dim vector (paper §2.3.3).

    hour of day, day of week, day of month are each mapped to (sin, cos) with
    *fixed* periods (24, 7, 31), then embedded with a 2-layer MLP. Added to the
    aggregated [EVT] representation of each event.
    """

    PERIODS = (24.0, 7.0, 31.0)  # hour, day-of-week, day-of-month

    def __init__(self, cfg: PRAGMAConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * len(self.PERIODS), cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, calendar: torch.Tensor) -> torch.Tensor:
        """calendar: (B, E, 3) int -> (B, E, d)."""
        periods = torch.tensor(self.PERIODS, device=calendar.device, dtype=torch.float)
        angles = 2 * math.pi * calendar.float() / periods  # (B, E, 3)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, E, 6)
        return self.mlp(feats)
