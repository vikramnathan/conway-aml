"""
Preprocess SAML-D transaction data into per-account event sequences, following
the PRAGMA event model (aml_conway.pdf): each account has an ordered event
history, where every event is a timestamp + a set of key/value pairs.

Each row in SAML-D.csv is a single transaction between a sender and a receiver.
We expand it into TWO events:
  - a TRANS_SEND event in the sender account's history
  - a TRANS_RECEIVE event in the receiver account's history

For every event we attach the transaction's is_laundering flag and
laundering_type, plus a derived "elapsed_since_mark": the time in seconds
between this event and the most recent laundering transaction in the SAME
account at or before this event (causal look-back).
  * A laundering event is its own most recent mark  -> elapsed_since_mark = 0
  * No laundering event at/before this one           -> elapsed_since_mark = -1

Output: preprocessed.json, written as JSON Lines (one account per line).
"""

import csv
import json
import sys
from datetime import date

INPUT = "SAML-D.csv"
OUTPUT = "preprocessed.json"

# CSV column indices
C_TIME, C_DATE = 0, 1
C_SENDER, C_RECEIVER = 2, 3
C_AMOUNT = 4
C_PAY_CUR, C_RCV_CUR = 5, 6
C_SENDER_LOC, C_RECEIVER_LOC = 7, 8
C_PAY_TYPE = 9
C_IS_LAUND, C_LAUND_TYPE = 10, 11

# Compact per-event tuple layout (stored in memory to limit overhead).
# (epoch, feature, direction, amount, pay_cur, rcv_cur, counterparty,
#  sender_loc, receiver_loc, pay_type, is_laundering, laundering_type)
E_EPOCH = 0

_date_cache = {}  # "YYYY-MM-DD" -> seconds at midnight (ordinal-based, UTC-agnostic)


def to_epoch(date_str, time_str):
    """Return integer seconds usable for ordering and differences.

    Uses proleptic-ordinal days * 86400 + seconds-of-day. We only ever take
    differences, so no timezone handling is required; results are consistent.
    """
    base = _date_cache.get(date_str)
    if base is None:
        y, m, d = date_str.split("-")
        base = date(int(y), int(m), int(d)).toordinal() * 86400
        _date_cache[date_str] = base
    h, mi, s = time_str.split(":")
    return base + int(h) * 3600 + int(mi) * 60 + int(s)


def epoch_to_str(epoch):
    days, secs = divmod(epoch, 86400)
    d = date.fromordinal(days)
    h, rem = divmod(secs, 3600)
    mi, s = divmod(rem, 60)
    return f"{d.isoformat()} {h:02d}:{mi:02d}:{s:02d}"


def main(limit=None):
    intern = sys.intern
    accounts = {}  # account_id -> list of event tuples
    n = 0

    with open(INPUT, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            n += 1
            if limit and n > limit:
                break

            epoch = to_epoch(row[C_DATE], row[C_TIME])
            amount = float(row[C_AMOUNT])
            # Intern repeated categorical strings to keep memory bounded.
            pay_cur = intern(row[C_PAY_CUR])
            rcv_cur = intern(row[C_RCV_CUR])
            sender_loc = intern(row[C_SENDER_LOC])
            receiver_loc = intern(row[C_RECEIVER_LOC])
            pay_type = intern(row[C_PAY_TYPE])
            laund_type = intern(row[C_LAUND_TYPE])
            is_laund = 1 if row[C_IS_LAUND] == "1" else 0
            sender = row[C_SENDER]
            receiver = row[C_RECEIVER]

            # SEND event -> sender's history
            accounts.setdefault(sender, []).append(
                (epoch, "TRANS_SEND", "out", amount, pay_cur, rcv_cur, receiver,
                 sender_loc, receiver_loc, pay_type, is_laund, laund_type)
            )
            # RECEIVE event -> receiver's history
            accounts.setdefault(receiver, []).append(
                (epoch, "TRANS_RECEIVE", "in", amount, pay_cur, rcv_cur, sender,
                 sender_loc, receiver_loc, pay_type, is_laund, laund_type)
            )

            if n % 1_000_000 == 0:
                print(f"  read {n:,} rows, {len(accounts):,} accounts", file=sys.stderr)

    print(f"Read {n:,} transactions -> {len(accounts):,} accounts. Writing {OUTPUT} ...",
          file=sys.stderr)

    n_events = 0
    with open(OUTPUT, "w") as out:
        for acct, events in accounts.items():
            # Order this account's history chronologically (stable on ties).
            events.sort(key=lambda e: e[E_EPOCH])

            last_mark = None  # epoch of most recent laundering event seen so far
            ev_dicts = []
            for e in events:
                epoch = e[0]
                is_laund = e[10]
                if is_laund == 1:
                    elapsed = 0
                    last_mark = epoch
                elif last_mark is not None:
                    elapsed = epoch - last_mark
                else:
                    elapsed = -1

                ev_dicts.append({
                    "timestamp": epoch_to_str(epoch),
                    "feature": e[1],
                    "direction": e[2],
                    "amount": e[3],
                    "payment_currency": e[4],
                    "received_currency": e[5],
                    "counterparty_account": e[6],
                    "sender_bank_location": e[7],
                    "receiver_bank_location": e[8],
                    "payment_type": e[9],
                    "is_laundering": is_laund,
                    "laundering_type": e[11],
                    "elapsed_since_mark": elapsed,
                })

            n_events += len(ev_dicts)
            out.write(json.dumps({
                "account": acct,
                "num_events": len(ev_dicts),
                "events": ev_dicts,
            }, separators=(",", ":")))
            out.write("\n")

    print(f"Done: {len(accounts):,} account lines, {n_events:,} events written to {OUTPUT}",
          file=sys.stderr)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
