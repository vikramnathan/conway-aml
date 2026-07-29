"""PRAGMA backbone: Profile State Encoder + Event Encoder + History Encoder.

Data flow (paper §2.3, Figure 4):

  Profile State Encoder (bidir + RoPE on time-since-lifelong):
      x_a = Emb(profile tokens)  -> [USR:x_a] -> encoder -> z_a = out[:, 0]   (B, d)

  Event Encoder (bidir, no time RoPE, each event independent):
      x_e,i = Emb(event i tokens) -> [EVT:x_e,i] -> encoder
          -> zhat_e   token-level outputs (used by MLM head, local context)
          -> z'_e = out[:, 0]  aggregated [EVT] per event
      z_e = z'_e + calendar_mlp(calendar)                                   (B, E, d)

  History Encoder (bidir + RoPE on time-to-last-event):
      z = [z_a : z_e]  -> encoder
          -> z_h        (B, 1+E, d): z_h[:,0]=[USR], z_h[:,1:]=[EVT] per event

Outputs bundled in `PragmaOutput` for the MLM head / downstream probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .batch import PragmaBatch
from .config import EVT_ID, USR_ID, PRAGMAConfig, VocabConfig
from .embeddings import CalendarEmbedding, TokenEmbedding
from .encoder import TransformerEncoder


@dataclass
class PragmaOutput:
    event_token_repr: torch.Tensor   # (B, E, Te, d)  zhat_e: per-token event-encoder output
    event_token_mask: torch.Tensor   # (B, E, Te) bool
    z_h_evt: torch.Tensor            # (B, E, d)     history-encoder output at [EVT] positions
    z_h_usr: torch.Tensor            # (B, d)        history-encoder output at [USR] position
    event_mask: torch.Tensor         # (B, E) bool


class PRAGMA(nn.Module):
    def __init__(self, vocab: VocabConfig, cfg: PRAGMAConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab

        self.embed = TokenEmbedding(vocab, cfg)
        self.calendar = CalendarEmbedding(cfg)

        # Learnable [USR] / [EVT] prepend tokens (paper §2.3.1).
        self.usr_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.evt_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.usr_token, std=0.02)
        nn.init.normal_(self.evt_token, std=0.02)

        if cfg.use_profile:
            self.profile_encoder = TransformerEncoder(cfg, cfg.profile_depth, use_time_rope=True)
        self.event_encoder = TransformerEncoder(cfg, cfg.event_depth, use_time_rope=False)
        self.history_encoder = TransformerEncoder(cfg, cfg.history_depth, use_time_rope=True)

    # ---- branches --------------------------------------------------------
    def encode_profile(self, batch: PragmaBatch) -> torch.Tensor:
        """Returns z_a: (B, d). Zeros if profile disabled/absent."""
        B = batch.event_key_ids.shape[0]
        if not self.cfg.use_profile or batch.profile_key_ids is None:
            return self.usr_token.expand(B, -1)

        x = self.embed(batch.profile_key_ids, batch.profile_val_ids, batch.profile_field_pos)
        usr = self.usr_token.expand(B, 1, -1)
        x = torch.cat([usr, x], dim=1)  # (B, 1+Ta, d)

        # [USR] slot always attended; its time coord is 0.
        real = batch.profile_token_mask  # (B, Ta)
        kpm = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), real], dim=1)
        t = torch.cat([torch.zeros(B, 1, device=x.device), batch.profile_time], dim=1)

        out = self.profile_encoder(x, key_padding_mask=kpm, t=t)
        return out[:, 0]  # [USR] aggregate

    def encode_events(self, batch: PragmaBatch):
        """Returns (zhat_e (B,E,Te,d), z_e (B,E,d)). Each event encoded independently."""
        B, E, Te = batch.event_key_ids.shape
        flat = B * E
        key = batch.event_key_ids.reshape(flat, Te)
        val = batch.event_val_ids.reshape(flat, Te)
        fpos = batch.event_field_pos.reshape(flat, Te)
        tok_mask = batch.event_token_mask.reshape(flat, Te)

        x = self.embed(key, val, fpos)                       # (flat, Te, d)
        evt = self.evt_token.expand(flat, 1, -1)
        x = torch.cat([evt, x], dim=1)                       # (flat, 1+Te, d)
        kpm = torch.cat([torch.ones(flat, 1, dtype=torch.bool, device=x.device), tok_mask], dim=1)

        out = self.event_encoder(x, key_padding_mask=kpm, t=None)  # (flat, 1+Te, d)
        z_evt = out[:, 0].reshape(B, E, -1)                  # aggregated [EVT] per event
        zhat = out[:, 1:].reshape(B, E, Te, -1)              # per-token local context

        z_e = z_evt + self.calendar(batch.event_calendar)    # add calendar features
        return zhat, z_e

    def encode_history(self, batch: PragmaBatch, z_a: torch.Tensor, z_e: torch.Tensor):
        """Returns z_h: (B, 1+E, d)."""
        B, E, _ = z_e.shape
        z = torch.cat([z_a[:, None, :], z_e], dim=1)         # (B, 1+E, d)
        kpm = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=z.device), batch.event_mask], dim=1)
        # time-to-last-event; 0 for the [USR] (z_a) slot (paper §2.3.4).
        t = torch.cat([torch.zeros(B, 1, device=z.device), batch.event_time_to_last], dim=1)
        return self.history_encoder(z, key_padding_mask=kpm, t=t)

    # ---- full forward ----------------------------------------------------
    def forward(self, batch: PragmaBatch) -> PragmaOutput:
        z_a = self.encode_profile(batch)
        zhat_e, z_e = self.encode_events(batch)
        z_h = self.encode_history(batch, z_a, z_e)
        return PragmaOutput(
            event_token_repr=zhat_e,
            event_token_mask=batch.event_token_mask,
            z_h_evt=z_h[:, 1:],
            z_h_usr=z_h[:, 0],
            event_mask=batch.event_mask,
        )
