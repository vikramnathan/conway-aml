"""Tokeniser: raw K/V event records (preprocessed.json) -> model token ids.

Bridges the event builder's output to the `PragmaBatch` contract. Implements
the value-typing scheme from §2.2:

  - numerical fields  -> percentile buckets (boundaries fit from training data),
                         with a dedicated zero bucket; one token per bucket.
  - categorical fields -> single token per value (low cardinality).
  - textual fields     -> (not present in SAML-D) would BPE-split; we treat any
                         string field as categorical here given the schema.

Keys are single tokens (§2.2). The tokeniser is *fit* once on a data sample to
learn the key vocab, categorical value vocab, and numeric bucket edges, then
`encode_record` maps a raw record to per-event token id lists.

Input record schema (from preprocessed.sample.json):
    { "account": str, "num_events": int,
      "events": [ { "timestamp": "YYYY-MM-DD HH:MM:SS", "feature": "TRANS_SEND"/..,
                    "amount": float, "payment_currency": str, ... }, ... ] }

Fields `is_laundering` / `laundering_type` are DOWNSTREAM LABELS and are
excluded from pretraining inputs to prevent leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import NUM_SPECIAL, VocabConfig

# Fields consumed as event K/V pairs. `feature` is the event type (TRANS_SEND/RECEIVE).
CATEGORICAL_FIELDS = [
    "feature", "direction", "payment_currency", "received_currency",
    "sender_bank_location", "receiver_bank_location", "payment_type",
    "counterparty_account",
]
NUMERICAL_FIELDS = ["amount"]
# Excluded from inputs: timestamp (-> time features), is_laundering, laundering_type
# (labels), num_events, account, elapsed_since_mark (builder bookkeeping).
LABEL_FIELDS = {"is_laundering", "laundering_type"}

# ---- Profile state schema (profiles.json) ------------------------------------
# Static attributes at the evaluation point (§2.1.2), encoded like event K/V but
# processed by the Profile State Encoder.
PROFILE_CATEGORICAL = ["home_location", "home_currency"]
PROFILE_NUMERICAL = ["num_events", "num_send", "num_receive"]
# Lifelong events (§2.1.2): each carries a first-occurrence timestamp used to
# compute time-since-eval-point (via RoPE in the Profile Encoder). Value is a
# single "present" token; the signal is the key identity + its time coordinate.
PROFILE_LIFELONG = ["first_transaction", "first_send", "first_receive"]

N_BUCKETS = 64  # percentile buckets for numeric fields (+1 reserved zero bucket)


@dataclass
class FittedTokenizer:
    key_to_id: dict            # field name -> key token id (events AND profile)
    cat_val_to_id: dict        # (field, str value) -> local value token id
    num_bucket_base: dict      # field -> base local value id for its buckets
    num_edges: dict            # field -> np.ndarray of percentile edges
    n_keys: int
    n_values: int
    high_card_fields: set = field(default_factory=set)
    # profile lifelong keys carry a shared "present" value token
    lifelong_present_id: int = 0  # local value id used for lifelong-event pairs

    def vocab_config(self, max_field_pos: int = 8) -> VocabConfig:
        return VocabConfig(n_keys=self.n_keys, n_values=self.n_values, max_field_pos=max_field_pos)


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def fit_tokenizer(
    records: list[dict],
    profiles: list[dict] | None = None,
    high_card_threshold: int = 5000,
) -> FittedTokenizer:
    """Learn key vocab, categorical value vocab, and numeric bucket edges.

    Fits over both event fields and (optionally) profile-state fields so that
    keys and values share one embedding table (§2.3.1). Categorical fields whose
    cardinality exceeds `high_card_threshold` (e.g. counterparty_account) are
    hashed into a fixed number of buckets to bound the value vocabulary.
    """
    profiles = profiles or []

    # ---- keys (events + profile categorical/numerical + lifelong) ----
    all_keys = (
        CATEGORICAL_FIELDS + NUMERICAL_FIELDS
        + PROFILE_CATEGORICAL + PROFILE_NUMERICAL + PROFILE_LIFELONG
    )
    key_to_id = {name: NUM_SPECIAL + i for i, name in enumerate(all_keys)}
    n_keys = len(key_to_id)

    cat_fields = CATEGORICAL_FIELDS + PROFILE_CATEGORICAL
    num_fields = NUMERICAL_FIELDS + PROFILE_NUMERICAL

    # ---- gather categorical values + numeric samples ----
    cat_values: dict[str, set] = {f: set() for f in cat_fields}
    num_samples: dict[str, list] = {f: [] for f in num_fields}
    for rec in records:
        for ev in rec["events"]:
            for f in CATEGORICAL_FIELDS:
                if ev.get(f) is not None:
                    cat_values[f].add(str(ev[f]))
            for f in NUMERICAL_FIELDS:
                if ev.get(f) is not None:
                    num_samples[f].append(float(ev[f]))
    for prof in profiles:
        for f in PROFILE_CATEGORICAL:
            if prof.get(f) is not None:
                cat_values[f].add(str(prof[f]))
        for f in PROFILE_NUMERICAL:
            if prof.get(f) is not None:
                num_samples[f].append(float(prof[f]))

    HASH_BUCKETS = 4096
    high_card = {f for f in cat_fields if len(cat_values[f]) > high_card_threshold}

    cat_val_to_id: dict = {}
    next_val = 0  # value ids are 0-based here; global offset applied at encode time
    for f in cat_fields:
        if f in high_card:
            for h in range(HASH_BUCKETS):
                cat_val_to_id[(f, f"__hash_{h}")] = next_val
                next_val += 1
        else:
            for v in sorted(cat_values[f]):
                cat_val_to_id[(f, v)] = next_val
                next_val += 1

    # ---- numeric bucket edges (percentiles), + reserved zero bucket ----
    num_bucket_base: dict = {}
    num_edges: dict = {}
    for f in num_fields:
        arr = np.asarray(num_samples[f], dtype=np.float64)
        nz = arr[arr != 0.0]
        qs = np.linspace(0, 100, N_BUCKETS + 1)[1:-1]
        edges = np.quantile(nz, qs / 100.0) if nz.size else np.array([0.0])
        num_edges[f] = edges
        num_bucket_base[f] = next_val
        next_val += (N_BUCKETS + 1)

    # ---- shared "present" value token for lifelong-event pairs ----
    lifelong_present_id = next_val
    next_val += 1

    return FittedTokenizer(
        key_to_id=key_to_id, cat_val_to_id=cat_val_to_id,
        num_bucket_base=num_bucket_base, num_edges=num_edges,
        n_keys=n_keys, n_values=next_val, high_card_fields=high_card,
        lifelong_present_id=lifelong_present_id,
    )


def _cat_value_id(tok: FittedTokenizer, f: str, v: str) -> int:
    if f in tok.high_card_fields:
        h = (hash(v) % 4096 + 4096) % 4096
        return tok.cat_val_to_id[(f, f"__hash_{h}")]
    # unseen categorical value -> hash-free fallback: map to bucket 0 of field
    return tok.cat_val_to_id.get((f, v), next(iter(
        (vid for (ff, _), vid in tok.cat_val_to_id.items() if ff == f)), 0))


def _num_value_id(tok: FittedTokenizer, f: str, x: float) -> int:
    base = tok.num_bucket_base[f]
    if x == 0.0:
        return base + N_BUCKETS  # reserved zero bucket (last slot)
    b = int(np.searchsorted(tok.num_edges[f], x, side="right"))
    return base + min(b, N_BUCKETS - 1)


def encode_event(tok: FittedTokenizer, ev: dict) -> tuple[list[int], list[int], list[int]]:
    """Return (key_ids, val_ids, field_pos) for one event.

    Value ids are shifted into the global table: value_id_lo + local_value_id.
    All fields here are single-valued (field_pos = 0). Multi-valued (text) fields
    would emit multiple (key,value) with field_pos 0,1,2,...
    """
    val_lo = NUM_SPECIAL + tok.n_keys
    keys, vals, pos = [], [], []
    for f in CATEGORICAL_FIELDS:
        if f in ev and ev[f] is not None:
            keys.append(tok.key_to_id[f])
            vals.append(val_lo + _cat_value_id(tok, f, str(ev[f])))
            pos.append(0)
    for f in NUMERICAL_FIELDS:
        if f in ev and ev[f] is not None:
            keys.append(tok.key_to_id[f])
            vals.append(val_lo + _num_value_id(tok, f, float(ev[f])))
            pos.append(0)
    return keys, vals, pos


def encode_profile(
    tok: FittedTokenizer, prof: dict, eval_ts: datetime
) -> tuple[list[int], list[int], list[int], list[float]]:
    """Encode one profile record -> (key_ids, val_ids, field_pos, time_seconds).

    Categorical/numerical attributes get time 0 (they describe state at the eval
    point). Lifelong events get a shared "present" value token and a time
    coordinate = seconds from the lifelong-event timestamp back to `eval_ts`
    (>= 0), consumed by the Profile Encoder via RoPE (§2.1.2/§2.3.2).
    """
    val_lo = NUM_SPECIAL + tok.n_keys
    keys, vals, pos, times = [], [], [], []

    for f in PROFILE_CATEGORICAL:
        if prof.get(f) is not None:
            keys.append(tok.key_to_id[f])
            vals.append(val_lo + _cat_value_id(tok, f, str(prof[f])))
            pos.append(0); times.append(0.0)
    for f in PROFILE_NUMERICAL:
        if prof.get(f) is not None:
            keys.append(tok.key_to_id[f])
            vals.append(val_lo + _num_value_id(tok, f, float(prof[f])))
            pos.append(0); times.append(0.0)

    lifelong = prof.get("lifelong", {}) or {}
    for f in PROFILE_LIFELONG:
        ts = lifelong.get(f)
        if ts is None:
            continue
        keys.append(tok.key_to_id[f])
        vals.append(val_lo + tok.lifelong_present_id)
        pos.append(0)
        times.append(max(0.0, (eval_ts - _parse_ts(ts)).total_seconds()))
    return keys, vals, pos, times


def event_time_features(events: list[dict]) -> tuple[list[float], list[tuple[int, int, int]]]:
    """Compute (time_to_last_seconds, calendar[hour,dow,dom]) per event.

    Assumes events are chronologically ordered. time_to_last = seconds from the
    event to the most-recent (last) event in the record (>= 0).
    """
    ts = [_parse_ts(e["timestamp"]) for e in events]
    last = ts[-1]
    ttl = [max(0.0, (last - t).total_seconds()) for t in ts]
    cal = [(t.hour, t.weekday(), t.day - 1) for t in ts]
    return ttl, cal


def load_records(path: str, limit: int | None = None) -> list[dict]:
    """Load records, auto-detecting JSON Lines vs a top-level JSON array.

    The full corpus (preprocessed.json) is JSONL: one record object per line.
    The sample file is a JSON array. We sniff the first non-whitespace bytes to
    pick a streaming line reader (memory-friendly for the 6.8 GB file) or a
    plain json.load for the array form.
    """
    with open(path, "rb") as f:
        head = f.read(64).lstrip()
    is_array = head[:1] == b"["

    if is_array:
        with open(path) as f:
            data = json.load(f)
        return data[:limit] if limit else data

    # JSON Lines: parse line by line (skip blanks).
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def load_profiles(path: str) -> dict:
    """Load profiles (JSONL or array) into an {account: profile} dict."""
    return {p["account"]: p for p in load_records(path)}
