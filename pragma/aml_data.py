"""Decision-point dataset for the downstream AML head.

The AML use case is "predict fraud from the event representation available *now*"
(cf. an MLM-pretrained encoder used for classification): we feed the event
prefix up to a decision point i and read the representation at that point. The
backbone stays bidirectional (as pretrained) but the *input* ends at i, so there
is no future-event leakage.

Construction (per account, per user spec):
  - a decision point at event i feeds events[0..i] and is labelled by e_i:
        0 <= e_i < T   -> positive (y=1, in the laundering window)
        e_i < 0 or >=T -> negative (y=0)
  - keep ALL positive decision points;
  - keep ~`neg_frac` (default 1%) of negative decision points, sampled
    per account (so innocent accounts are still represented).

Each decision point reuses the event encoding but truncates the history to end
at i (and to the most-recent MAX_EVENTS before i). The AML head reads the LAST
event's contextual [EVT] output as the record representation at that point.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from .batch import PragmaBatch, soft_log_seconds
from .data import MAX_EVENTS, MAX_TOKENS_PER_EVENT, MAX_PROFILE_TOKENS
from .tokenizer import (
    FittedTokenizer, encode_event, encode_profile, event_time_features, _parse_ts,
)


def build_decision_points(
    records: list[dict],
    T_seconds: float,
    neg_frac: float = 0.01,
    seed: int = 0,
) -> list[tuple[int, int, int]]:
    """Return a list of (record_index, cut_index, label).

    cut_index i means: use events[0..i] inclusive; label from elapsed_since_mark[i].
    Positives (0<=e<T) all kept; negatives sampled per account at `neg_frac`.
    """
    rng = random.Random(seed)
    points: list[tuple[int, int, int]] = []
    for ridx, rec in enumerate(records):
        events = rec["events"]
        neg_idx = []
        for i, ev in enumerate(events):
            e = float(ev.get("elapsed_since_mark", -1))
            if 0.0 <= e < T_seconds:
                points.append((ridx, i, 1))          # positive: keep all
            else:
                neg_idx.append(i)
        # per-account negative subsample (at least keep the account represented
        # if it has any negatives: expected count = neg_frac * len, but Bernoulli)
        for i in neg_idx:
            if rng.random() < neg_frac:
                points.append((ridx, i, 0))
    rng.shuffle(points)
    return points


class AMLDecisionDataset(Dataset):
    """Yields one masked-free sample per decision point (prefix + single label)."""

    def __init__(
        self,
        records: list[dict],
        tok: FittedTokenizer,
        points: list[tuple[int, int, int]],
        profiles: dict | None = None,
        max_events: int = MAX_EVENTS,
    ):
        self.records = records
        self.tok = tok
        self.points = points
        self.profiles = profiles or {}
        self.max_events = max_events

    def __len__(self):
        return len(self.points)

    def __getitem__(self, k):
        ridx, cut, label = self.points[k]
        rec = self.records[ridx]
        # prefix up to and including `cut`, then most-recent-max_events window
        events = rec["events"][: cut + 1][-self.max_events:]

        enc = [encode_event(self.tok, ev) for ev in events]
        enc = [(kk[:MAX_TOKENS_PER_EVENT], vv[:MAX_TOKENS_PER_EVENT], pp[:MAX_TOKENS_PER_EVENT])
               for kk, vv, pp in enc]
        ttl, cal = event_time_features(events)

        prof_enc = None
        acct = rec.get("account")
        if acct in self.profiles:
            eval_ts = _parse_ts(events[-1]["timestamp"])  # decision-point time
            pk, pv, pp, pt = encode_profile(self.tok, self.profiles[acct], eval_ts)
            prof_enc = (pk[:MAX_PROFILE_TOKENS], pv[:MAX_PROFILE_TOKENS],
                        pp[:MAX_PROFILE_TOKENS], pt[:MAX_PROFILE_TOKENS])

        return {"enc": enc, "ttl": ttl, "cal": cal, "label": float(label), "prof": prof_enc}


def collate_decision(samples: list[dict]) -> PragmaBatch:
    """Pad decision-point samples. Carries a single per-record label in aml_label
    at the LAST real event position (the decision point); other positions are
    ignored by the head via the returned `decision_index`."""
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
    label = torch.zeros(B, dtype=torch.float)
    decision_index = torch.zeros(B, dtype=torch.long)  # last real event per record

    for b, s in enumerate(samples):
        n_ev = len(s["enc"])
        for e, (k, v, p) in enumerate(s["enc"]):
            evt_mask[b, e] = True
            n = len(k)
            key_ids[b, e, :n] = torch.tensor(k, dtype=torch.long)
            val_ids[b, e, :n] = torch.tensor(v, dtype=torch.long)
            field_pos[b, e, :n] = torch.tensor(p, dtype=torch.long)
            tok_mask[b, e, :n] = True
            ttl[b, e] = s["ttl"][e]
            cal[b, e] = torch.tensor(s["cal"][e], dtype=torch.long)
        label[b] = s["label"]
        decision_index[b] = n_ev - 1

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
        p_time = soft_log_seconds(p_time)

    batch = PragmaBatch(
        event_key_ids=key_ids, event_val_ids=val_ids, event_field_pos=field_pos,
        event_token_mask=tok_mask, event_mask=evt_mask,
        event_time_to_last=soft_log_seconds(ttl), event_calendar=cal,
        profile_key_ids=p_key, profile_val_ids=p_val, profile_field_pos=p_pos,
        profile_token_mask=p_mask, profile_time=p_time,
        aml_label=label,
        decision_index=decision_index,  # (B,) index of the decision event
    )
    return batch
