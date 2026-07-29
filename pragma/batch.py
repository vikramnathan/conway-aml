"""The batch contract between the (external) event builder and the model.

A single *record* is one account's history: an ordered list of events plus an
optional profile-state block. The model consumes a `PragmaBatch` of records.

Tokenisation recap (paper §2.2, Fig 3). Every field of an event is decomposed
into (key, value, time). A field with multiple value tokens (e.g. a BPE-split
text value) replicates its key once per value token. So one event is a flat
list of (key_id, value_id, field_pos) triples, where `field_pos` indexes the
value *within its field* (0 for single-valued fields; 0,1,2,... for multi-valued).

Shapes below use:
  B  = records (accounts) in the batch
  E  = max #events per record (padded)
  Te = max #tokens per event (padded)         -- paper truncates events to <=24
  Ta = max #profile-state tokens (padded)      -- paper truncates profile to <=200

All id tensors are torch.long; masks are torch.bool (True = real, False = pad);
time tensors are torch.float (log-seconds, already soft-log transformed by the
builder OR raw seconds — see `soft_log` note). Calendar is integer hour/dow/dom.

The model never assumes a specific number of keys/events; it only relies on the
padding masks and the id layout described in `config.VocabConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PragmaBatch:
    # ---- Events ----------------------------------------------------------
    # (B, E, Te) token ids: key ids and value ids for every within-event token.
    event_key_ids: torch.Tensor
    event_val_ids: torch.Tensor
    # (B, E, Te) within-field position of each value token (for within-field PosEmb).
    event_field_pos: torch.Tensor
    # (B, E, Te) bool: True where the (key,value) token is real (not padding).
    event_token_mask: torch.Tensor
    # (B, E) bool: True where the event slot is a real event (not padding).
    event_mask: torch.Tensor
    # (B, E) float: log-seconds from each event to the most recent event in the
    # record (0 for the most recent). Used by the History Encoder via RoPE.
    event_time_to_last: torch.Tensor
    # (B, E, 3) int: calendar features (hour_of_day, day_of_week, day_of_month).
    event_calendar: torch.Tensor

    # ---- Profile state (optional; may be None when use_profile=False) ----
    profile_key_ids: torch.Tensor | None = None      # (B, Ta)
    profile_val_ids: torch.Tensor | None = None      # (B, Ta)
    profile_field_pos: torch.Tensor | None = None    # (B, Ta)
    profile_token_mask: torch.Tensor | None = None    # (B, Ta) bool
    # (B, Ta) float: log-seconds since the lifelong event for that pair
    # (0 for non-lifelong profile pairs). Used by the Profile Encoder via RoPE.
    profile_time: torch.Tensor | None = None

    # ---- MLM supervision (filled by the masking collator, not the builder) --
    # (B, E, Te) long: original value id at each masked position, else -100.
    # -100 positions are ignored by cross-entropy. [UNK]-corrupted positions are
    # also -100 (they receive no gradient, per §2.3.5).
    mlm_labels: torch.Tensor | None = None

    # ---- Downstream AML supervision (labels only; never model inputs) -----
    # (B, E) float: per-event laundering label y in {0,1}.
    aml_label: torch.Tensor | None = None
    # (B, E) float: elapsed_since_mark e (seconds; <0 means mark not yet occurred),
    # drives the lag-penalty f(e).
    aml_elapsed: torch.Tensor | None = None
    # (B,) long: for decision-point AML, the event index whose contextual [EVT]
    # is read as the record representation (the "now" event). None for per-event.
    decision_index: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.event_key_ids.device

    def to(self, device) -> "PragmaBatch":
        def mv(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return PragmaBatch(**{k: mv(v) for k, v in self.__dict__.items()})


def soft_log_seconds(t: torch.Tensor, scale: float = 8.0) -> torch.Tensor:
    """Soft-log temporal transform from §2.2: 8 * ln(1 + t/8).

    Compresses the dynamic range of large gaps (lifelong events) while keeping
    near-linear resolution for recent events. `t` is elapsed seconds (>= 0).
    """
    return scale * torch.log1p(t / scale)
