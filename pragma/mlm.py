"""MLM head and masking strategy (paper §2.3.5).

MLM head. For each (masked) event value token, concatenate three d-dim vectors:
  1. zhat_e  -- Event Encoder output at that token position (local within-event)
  2. z_h[EVT] -- History Encoder output at that event's [EVT] (cross-event)
  3. z_h[USR] -- History Encoder output at the [USR] position (user-level)
Project 3d -> d, then match against the shared embedding table E to produce
logits over the vocabulary. Loss is cross-entropy with label smoothing.

Masking. Three sources combined (probabilities from the paper):
  - token-level    15%  : individual value tokens
  - event-level    10%  : all value tokens of a whole event
  - key-level      10%  : all value tokens whose key was selected
Of the selected positions, a fraction (unk_fraction) are replaced with [UNK]
(excluded from the loss -> input dropout); the rest become [MASK] and are
supervised. Only *value* tokens are masked; keys are always visible so the
model predicts values given the key (§2.2/§2.3.5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch import PragmaBatch
from .config import MASK_ID, UNK_ID, MaskingConfig, PRAGMAConfig, VocabConfig
from .model import PRAGMA, PragmaOutput

IGNORE_INDEX = -100


class MLMHead(nn.Module):
    def __init__(self, vocab: VocabConfig, cfg: PRAGMAConfig, embedding_weight: nn.Parameter):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        # Weight-tied output: match projected repr against the shared table E.
        self.embedding_weight = embedding_weight  # (vocab, d)  -- tied, not re-registered
        self.bias = nn.Parameter(torch.zeros(embedding_weight.shape[0]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (N, 3d) -> logits (N, vocab)."""
        h = self.proj(feats)
        return F.linear(h, self.embedding_weight, self.bias)


class PRAGMAForMLM(nn.Module):
    def __init__(self, vocab: VocabConfig, cfg: PRAGMAConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.backbone = PRAGMA(vocab, cfg)
        self.head = MLMHead(vocab, cfg, self.backbone.embed.table.weight)

    def forward(self, batch: PragmaBatch) -> dict:
        out: PragmaOutput = self.backbone(batch)
        labels = batch.mlm_labels  # (B, E, Te), IGNORE_INDEX where not supervised

        B, E, Te, d = out.event_token_repr.shape
        sel = labels != IGNORE_INDEX  # (B, E, Te)
        if sel.sum() == 0:
            zero = out.z_h_usr.sum() * 0.0
            return {"loss": zero, "logits": None, "n_masked": 0}

        b_idx, e_idx, t_idx = sel.nonzero(as_tuple=True)
        local = out.event_token_repr[b_idx, e_idx, t_idx]  # (N, d)
        evt = out.z_h_evt[b_idx, e_idx]                     # (N, d)
        usr = out.z_h_usr[b_idx]                            # (N, d)
        feats = torch.cat([local, evt, usr], dim=-1)        # (N, 3d)

        logits = self.head(feats)                           # (N, vocab)
        target = labels[b_idx, e_idx, t_idx]                # (N,)
        loss = F.cross_entropy(
            logits, target, label_smoothing=self.cfg.label_smoothing
        )
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return {"loss": loss, "logits": logits, "n_masked": int(sel.sum()), "acc": acc}


@torch.no_grad()
def apply_masking(
    batch: PragmaBatch,
    vocab: VocabConfig,
    mcfg: MaskingConfig,
    generator: torch.Generator | None = None,
) -> PragmaBatch:
    """Return a *masked copy* of `batch` with `mlm_labels` populated.

    Only real value tokens (event_token_mask=True) are eligible. Sets labels to
    the original value id at supervised positions and IGNORE_INDEX elsewhere;
    rewrites event_val_ids in place on the copy with [MASK]/[UNK] where selected.
    """
    device = batch.event_val_ids.device
    val = batch.event_val_ids.clone()
    key = batch.event_key_ids
    tok_mask = batch.event_token_mask       # (B, E, Te)
    evt_mask = batch.event_mask             # (B, E)
    B, E, Te = val.shape

    def rand(shape):
        return torch.rand(shape, device=device, generator=generator)

    # --- select positions from three sources ---
    # token-level: per-token Bernoulli over real tokens
    sel = (rand((B, E, Te)) < mcfg.token_prob) & tok_mask

    # event-level: pick whole events, mask all their real tokens
    evt_pick = (rand((B, E)) < mcfg.event_prob) & evt_mask
    sel = sel | (evt_pick[:, :, None] & tok_mask)

    # key-level: per (record, key) selection. All tokens whose key was selected
    # within that record are masked together (paper: "all values of the selected
    # keys are masked"). Keys live in [key_id_lo, key_id_hi).
    for b in range(B):
        # unique real key ids in this record
        rk = key[b][tok_mask[b]]
        if rk.numel() == 0:
            continue
        uniq = torch.unique(rk)
        chosen = uniq[rand((uniq.numel(),)) < mcfg.key_prob]
        if chosen.numel() > 0:
            match = torch.isin(key[b], chosen) & tok_mask[b]
            sel[b] = sel[b] | match

    # --- split selected into [MASK] (supervised) vs [UNK] (input dropout) ---
    labels = torch.full_like(val, IGNORE_INDEX)
    labels[sel] = batch.event_val_ids[sel]

    is_unk = sel & (rand((B, E, Te)) < mcfg.unk_fraction)
    is_mask = sel & ~is_unk

    val[is_mask] = MASK_ID
    val[is_unk] = UNK_ID
    labels[is_unk] = IGNORE_INDEX  # UNK positions get no gradient (§2.3.5)

    masked = PragmaBatch(**{**batch.__dict__})
    masked.event_val_ids = val
    masked.mlm_labels = labels
    return masked
