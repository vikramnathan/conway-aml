"""Downstream AML head + lag-penalty loss.

This is a *downstream* objective (not part of MLM pretraining). It attaches a
per-event binary classifier on top of the History Encoder's contextualised
[EVT] outputs (z_h[EVT]) and trains it with an asymmetric, time-windowed loss.

Everything is derived from e = elapsed_since_mark (seconds), the single source
of truth:

    e < 0            -> account is NOT laundering at this event: label y = 0.
                        A false positive here costs weight 1.
    0 <= e < T       -> account IS laundering (within the detection window):
                        label y = 1. A miss (false negative) costs weight W.
    e >= T           -> penalty resets (we no longer care): event excluded.

Per-event loss, with L = binary cross-entropy:
    y == 1 (0 <= e < T):  weight = W        (default 10)
    y == 0 (e < 0):       weight = 1
    e >= T:               excluded from the loss

W up-weights the laundering window so misses are penalised heavily. e == 0 (the
mark event, is_laundering=1) is included in the positive window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch import PragmaBatch
from .config import PRAGMAConfig
from .model import PRAGMA


def aml_targets(e: torch.Tensor, T: float, W: float):
    """Derive (label y, per-event weight, in_loss mask) from elapsed-since-mark e.

    e: (…) seconds. Returns tensors of the same shape:
      y       : 1.0 where 0 <= e < T (laundering window), else 0.0
      weight  : W where y==1, 1.0 where e<0, 0.0 where e>=T (excluded)
      in_loss : bool, True where the event participates in the loss (e < T)
    """
    in_window = (e >= 0) & (e < T)          # laundering, detect -> y=1
    innocent = e < 0                         # not laundering -> y=0
    y = in_window.to(e.dtype)
    weight = torch.where(in_window, torch.full_like(e, W),
                         torch.where(innocent, torch.ones_like(e), torch.zeros_like(e)))
    in_loss = e < T                          # e>=T excluded (penalty reset)
    return y, weight, in_loss


class PRAGMAForAML(nn.Module):
    """PRAGMA backbone + per-event laundering classifier with windowed weighting."""

    def __init__(self, backbone: PRAGMA, cfg: PRAGMAConfig,
                 T: float = 30 * 24 * 3600.0, W: float = 10.0):
        super().__init__()
        self.backbone = backbone
        self.T = T
        self.W = W
        self.classifier = nn.Linear(cfg.d_model, 1)

    def forward(self, batch: PragmaBatch) -> dict:
        """Dispatch: decision-point mode if `decision_index` is present, else per-event."""
        if batch.decision_index is not None:
            return self.forward_decision(batch)
        return self.forward_per_event(batch)

    def forward_decision(self, batch: PragmaBatch) -> dict:
        """One prediction per record, read at the decision event (the 'now' event).

        Input history is already truncated to end at the decision point, so
        bidirectional attention over the prefix is leakage-free. Label is the
        windowed AML label at that event; positives (0<=e<T) are weighted by W.
        """
        out = self.backbone(batch)                          # z_h_evt: (B, E, d)
        B = out.z_h_evt.shape[0]
        idx = batch.decision_index                           # (B,)
        rep = out.z_h_evt[torch.arange(B, device=out.z_h_evt.device), idx]  # (B, d)
        logits = self.classifier(rep).squeeze(-1)            # (B,)

        y = batch.aml_label                                  # (B,) in {0,1}
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        weight = torch.where(y > 0.5, torch.full_like(y, self.W), torch.ones_like(y))
        loss = (bce * weight).sum() / weight.sum().clamp(min=1e-6)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
        return {"loss": loss, "logits": logits, "probs": probs, "labels": y,
                "n_pos": int(y.sum()), "n_in_loss": B, "n_valid": B}

    def forward_per_event(self, batch: PragmaBatch) -> dict:
        out = self.backbone(batch)
        logits = self.classifier(out.z_h_evt).squeeze(-1)  # (B, E)

        e = batch.aml_elapsed          # (B, E) seconds, <0 if no mark
        valid = out.event_mask         # (B, E) real events

        y, weight, in_loss = aml_targets(e, self.T, self.W)
        mask = valid & in_loss         # participate in loss

        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # (B, E)
        weighted = bce * weight

        denom = (weight * mask).sum().clamp(min=1e-6)  # weighted normaliser
        loss = (weighted * mask).sum() / denom

        with torch.no_grad():
            m = mask
            probs = torch.sigmoid(logits)[m]
            labels = y[m]
            n_pos = int(labels.sum())
        return {
            "loss": loss,
            "logits": logits,
            "probs": probs,
            "labels": labels,
            "n_pos": n_pos,
            "n_in_loss": int(mask.sum()),
            "n_valid": int(valid.sum()),
        }
