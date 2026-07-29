"""Stratified train/test split by account.

Accounts are stratified into two groups and each is split independently so the
(rare) laundering accounts are represented in both partitions at the same ratio:

  - guilty  : the account has at least one laundering event
              (is_laundering == 1, equivalently elapsed_since_mark >= 0)
  - innocent: no laundering event ever

Both strata are split `train_frac` / (1 - train_frac). The split is deterministic
given `seed` (hash-free: we sort account ids and shuffle with a seeded RNG), so
it is stable across machines and reruns.
"""

from __future__ import annotations

import random


def is_guilty(record: dict) -> bool:
    for ev in record["events"]:
        if int(ev.get("is_laundering", 0)) == 1 or float(ev.get("elapsed_since_mark", -1)) >= 0:
            return True
    return False


def stratified_split(
    records: list[dict],
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Return (train_records, test_records), stratified by guilty/innocent account."""
    guilty, innocent = [], []
    for rec in records:
        (guilty if is_guilty(rec) else innocent).append(rec)

    rng = random.Random(seed)

    def split_group(group: list[dict]) -> tuple[list, list]:
        # sort by account for determinism, then shuffle with the seeded RNG
        group = sorted(group, key=lambda r: str(r.get("account")))
        rng.shuffle(group)
        n_train = int(round(len(group) * train_frac))
        return group[:n_train], group[n_train:]

    g_tr, g_te = split_group(guilty)
    i_tr, i_te = split_group(innocent)

    train = g_tr + i_tr
    test = g_te + i_te
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def split_summary(train: list[dict], test: list[dict]) -> str:
    def counts(recs):
        g = sum(is_guilty(r) for r in recs)
        return len(recs), g, len(recs) - g
    ntr, gtr, itr = counts(train)
    nte, gte, ite = counts(test)
    return (f"train: {ntr} accts (guilty={gtr}, innocent={itr}) | "
            f"test: {nte} accts (guilty={gte}, innocent={ite})")
