"""Pre-norm bidirectional Transformer encoder blocks with continuous-time RoPE.

All three PRAGMA encoders (Profile State, Event, History) share this block. It
is a standard pre-norm Transformer encoder layer (GELU FFN, dropout 0.1) whose
attention is optionally rotated by RoPE driven by a temporal coordinate
(paper §2.3.2/§2.3.4). The Event Encoder uses no time RoPE (it relies only on
within-field PosEmb); the Profile and History encoders do.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PRAGMAConfig
from .embeddings import RotaryTimeEmbedding, apply_rope


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: PRAGMAConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x, key_padding_mask=None, cos=None, sin=None):
        """x: (B, S, d). key_padding_mask: (B, S) bool, True = real token."""
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, S, H, hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, S, hd)

        if cos is not None:
            q, k = apply_rope(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, S): disallow attending to padded keys.
            attn_mask = key_padding_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.proj(out)


class EncoderLayer(nn.Module):
    """Pre-norm: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, cfg: PRAGMAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ffn),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ffn, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None, cos=None, sin=None):
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask, cos, sin))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm layers + final LayerNorm, with optional time-RoPE.

    If `use_time_rope` is set, `forward` expects a temporal coordinate `t`
    (B, S) and rotates attention by it; otherwise `t` is ignored.
    """

    def __init__(self, cfg: PRAGMAConfig, depth: int, use_time_rope: bool):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(depth))
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.use_time_rope = use_time_rope
        self.rope = RotaryTimeEmbedding(cfg.head_dim, cfg.rope_theta) if use_time_rope else None

    def forward(self, x, key_padding_mask=None, t=None):
        cos = sin = None
        if self.use_time_rope:
            assert t is not None, "time-RoPE encoder requires temporal coordinate t"
            cos, sin = self.rope.cos_sin(t)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, cos=cos, sin=sin)
        return self.final_ln(x)
