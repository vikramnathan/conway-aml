"""
Build lightweight per-account PROFILE STATE from the per-account event stream,
following PRAGMA (aml_conway.pdf §2.1.2): static profile attributes plus
"life-long events" that carry the timestamp of a first occurrence.

Derived per account from its own events (which are already sorted ascending):
  - home_location : most frequent bank location for this account
                    (sender_bank_location on SEND, receiver_bank_location on RECEIVE)
  - home_currency : most frequent currency for this account
                    (payment_currency on SEND, received_currency on RECEIVE)
  - lifelong first-occurrence timestamps:
        first_transaction, first_send, first_receive  (null if that type never occurs)

Reads preprocessed.json (JSON Lines, one account per line) and writes
profiles.json (JSON Lines). Also builds profiles.sample.json (a JSON array)
for exactly the accounts present in preprocessed.sample.json.
"""

import json
import sys
from collections import Counter

EVENTS_FULL = "preprocessed.json"
EVENTS_SAMPLE = "preprocessed.sample.json"
OUT_FULL = "profiles.json"
OUT_SAMPLE = "profiles.sample.json"


def build_profile(acct):
    """acct: {'account', 'num_events', 'events':[...]} -> profile dict."""
    events = acct["events"]
    loc = Counter()
    cur = Counter()
    first_transaction = first_send = first_receive = None
    n_send = n_receive = 0

    for e in events:
        ts = e["timestamp"]
        if first_transaction is None:
            first_transaction = ts  # events are pre-sorted ascending
        if e["feature"] == "TRANS_SEND":
            n_send += 1
            if first_send is None:
                first_send = ts
            loc[e["sender_bank_location"]] += 1
            cur[e["payment_currency"]] += 1
        else:  # TRANS_RECEIVE
            n_receive += 1
            if first_receive is None:
                first_receive = ts
            loc[e["receiver_bank_location"]] += 1
            cur[e["received_currency"]] += 1

    return {
        "account": acct["account"],
        "home_location": loc.most_common(1)[0][0] if loc else None,
        "home_currency": cur.most_common(1)[0][0] if cur else None,
        "lifelong": {
            "first_transaction": first_transaction,
            "first_send": first_send,
            "first_receive": first_receive,
        },
        "num_events": acct["num_events"],
        "num_send": n_send,
        "num_receive": n_receive,
    }


def build_full():
    n = 0
    with open(EVENTS_FULL) as fin, open(OUT_FULL, "w") as fout:
        for line in fin:
            prof = build_profile(json.loads(line))
            fout.write(json.dumps(prof, separators=(",", ":")))
            fout.write("\n")
            n += 1
            if n % 200_000 == 0:
                print(f"  {n:,} profiles", file=sys.stderr)
    print(f"Done: {n:,} profiles -> {OUT_FULL}", file=sys.stderr)


def build_sample():
    with open(EVENTS_SAMPLE) as f:
        accts = json.load(f)  # JSON array
    profs = [build_profile(a) for a in accts]
    with open(OUT_SAMPLE, "w") as f:
        json.dump(profs, f, indent=2)
    print(f"Done: {len(profs)} profiles -> {OUT_SAMPLE}", file=sys.stderr)


if __name__ == "__main__":
    build_full()
    build_sample()
