"""Dataset + collate: encoded records -> padded PragmaBatch tensors.

Applies the paper's truncation (§2.4): events truncated to <=24 tokens, records
subsampled to the most-recent MAX_EVENTS. Pads ragged event/token dims and
builds the boolean masks the model relies on. Joins profile state by account.

Also carries per-event downstream supervision for the AML lag-penalty loss:
`is_laundering` (label y) and `elapsed_since_mark` (e, drives f(e)). These are
NOT fed as model inputs (no leakage) — they live only in the label tensors.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .batch import PragmaBatch, soft_log_seconds
from .tokenizer import (
    FittedTokenizer, encode_event, encode_profile, event_time_features, _parse_ts,
)

MAX_TOKENS_PER_EVENT = 24  # paper §2.4
MAX_EVENTS = 512           # paper subsamples to <=6500; smaller default here
MAX_PROFILE_TOKENS = 200   # paper §2.4


class PragmaDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tok: FittedTokenizer,
        profiles: dict | None = None,   # account -> profile dict
        max_events: int = MAX_EVENTS,
    ):
        self.records = records
        self.tok = tok
        self.profiles = profiles or {}
        self.max_events = max_events

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        # keep most recent events (temporal recency) and ensure chronological order
        events = sorted(rec["events"], key=lambda e: e["timestamp"])[-self.max_events:]

        enc = [encode_event(self.tok, ev) for ev in events]
        enc = [(k[:MAX_TOKENS_PER_EVENT], v[:MAX_TOKENS_PER_EVENT], p[:MAX_TOKENS_PER_EVENT])
               for k, v, p in enc]
        ttl, cal = event_time_features(events)

        # downstream AML supervision (per event)
        y = [int(ev.get("is_laundering", 0)) for ev in events]
        esm = [float(ev.get("elapsed_since_mark", -1)) for ev in events]

        # profile state (eval point = timestamp of most recent event)
        prof_enc = None
        acct = rec.get("account")
        if acct in self.profiles:
            eval_ts = _parse_ts(events[-1]["timestamp"])
            pk, pv, pp, pt = encode_profile(self.tok, self.profiles[acct], eval_ts)
            prof_enc = (pk[:MAX_PROFILE_TOKENS], pv[:MAX_PROFILE_TOKENS],
                        pp[:MAX_PROFILE_TOKENS], pt[:MAX_PROFILE_TOKENS])

        return {"enc": enc, "ttl": ttl, "cal": cal, "y": y, "esm": esm, "prof": prof_enc}


def collate(samples: list[dict]) -> PragmaBatch:
    B = len(samples)
    E = max(len(s["enc"]) for s in samples)
    Te = max((max((len(k) for k, _, _ in s["enc"]), default=1) for s in samples), default=1)
    Te = max(Te, 1)

    key_ids = torch.zeros(B, E, Te, dtype=torch.long)
    val_ids = torch.zeros(B, E, Te, dtype=torch.long)
    field_pos = torch.zeros(B, E, Te, dtype=torch.long)
    tok_mask = torch.zeros(B, E, Te, dtype=torch.bool)
    evt_mask = torch.zeros(B, E, dtype=torch.bool)
    ttl = torch.zeros(B, E, dtype=torch.float)
    cal = torch.zeros(B, E, 3, dtype=torch.long)
    y = torch.zeros(B, E, dtype=torch.float)
    esm = torch.full((B, E), -1.0, dtype=torch.float)

    for b, s in enumerate(samples):
        for e, (k, v, p) in enumerate(s["enc"]):
            evt_mask[b, e] = True
            n = len(k)
            key_ids[b, e, :n] = torch.tensor(k, dtype=torch.long)
            val_ids[b, e, :n] = torch.tensor(v, dtype=torch.long)
            field_pos[b, e, :n] = torch.tensor(p, dtype=torch.long)
            tok_mask[b, e, :n] = True
            ttl[b, e] = s["ttl"][e]
            cal[b, e] = torch.tensor(s["cal"][e], dtype=torch.long)
            y[b, e] = s["y"][e]
            esm[b, e] = s["esm"][e]

    # ---- profile state (padded); None if no sample has a profile ----
    has_prof = any(s["prof"] is not None for s in samples)
    p_key = p_val = p_pos = p_mask = p_time = None
    if has_prof:
        Ta = max((len(s["prof"][0]) if s["prof"] else 0) for s in samples)
        Ta = max(Ta, 1)
        p_key = torch.zeros(B, Ta, dtype=torch.long)
        p_val = torch.zeros(B, Ta, dtype=torch.long)
        p_pos = torch.zeros(B, Ta, dtype=torch.long)
        p_mask = torch.zeros(B, Ta, dtype=torch.bool)
        p_time = torch.zeros(B, Ta, dtype=torch.float)
        for b, s in enumerate(samples):
            if not s["prof"]:
                continue
            pk, pv, pp, pt = s["prof"]
            n = len(pk)
            p_key[b, :n] = torch.tensor(pk, dtype=torch.long)
            p_val[b, :n] = torch.tensor(pv, dtype=torch.long)
            p_pos[b, :n] = torch.tensor(pp, dtype=torch.long)
            p_mask[b, :n] = True
            p_time[b, :n] = torch.tensor(pt, dtype=torch.float)
        p_time = soft_log_seconds(p_time)  # soft-log transform (§2.2)

    batch = PragmaBatch(
        event_key_ids=key_ids,
        event_val_ids=val_ids,
        event_field_pos=field_pos,
        event_token_mask=tok_mask,
        event_mask=evt_mask,
        event_time_to_last=soft_log_seconds(ttl),
        event_calendar=cal,
        profile_key_ids=p_key,
        profile_val_ids=p_val,
        profile_field_pos=p_pos,
        profile_token_mask=p_mask,
        profile_time=p_time,
        aml_label=y,
        aml_elapsed=esm,
    )
    return batch
