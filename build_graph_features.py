"""
Build cumulative-causal graph-topology features per LABELED EVENT (edge endpoint),
to be concatenated with PRAGMA per-event embeddings before the AML classification
head.

The transaction graph is a directed, weighted, temporal multigraph:
  node = account, edge = transaction (sender -> receiver), weighted by amount,
  timestamped. Each transaction yields two events: TRANS_SEND (on sender) and
  TRANS_RECEIVE (on receiver).

Approach (one global pass, time-ordered):
  We stream transactions in ascending timestamp order and accumulate edges into
  live adjacency structures. Because the graph only grows, the structure at the
  moment we process an event at time t is EXACTLY the subgraph of edges with
  timestamp <= t -- i.e. cumulative-causal, no future leakage. We insert the
  current edge before computing features, so an event's own (known) transaction
  is included, but nothing after it is.

Feature families (see paper limitation: PRAGMA is blind to cross-record structure):
  A -- local aggregates: degrees, distinct counterparties, amount stats, ratios,
       reciprocity, account age. Maintained incrementally, exact.
  B -- 2-hop ego / motif: mutual (2-cycle) count, directed-triangle (3-cycle)
       participation, 2-hop reach in/out, recent-window fan-in/out bursts,
       pass-through balance, gather-scatter score. Computed only at SELECTED
       events via capped neighborhood traversal.

Event selection (stratified, all rates tunable):
  positive   : is_laundering == 1                                 -> P_POS
  post_mark  : 0 < elapsed_since_mark <= WINDOW_DAYS (days)        -> P_WINDOW
  background : everything else                                    -> P_BG
  elapsed_since_mark is the causal look-back to the most recent laundering event
  in the SAME account (recomputed here, identical logic to preprocess.py).
  Sampling is deterministic (crc32 hash of account|timestamp|feature), so runs
  are reproducible without an RNG seed.

Outputs:
  graph_features.json         JSON Lines, one selected event per line.
  graph_features.sample.json  JSON array; ALL events of the accounts present in
                              preprocessed.sample.json (force-included so the
                              sample is inspectable), each tagged with its
                              stratum and whether it was sampled-in.
"""

import csv
import json
import sys
import zlib
from datetime import date

INPUT = "SAML-D.csv"
SAMPLE_ACCOUNTS_FROM = "preprocessed.sample.json"
OUT_FULL = "graph_features.json"
OUT_SAMPLE = "graph_features.sample.json"

# ----- tunable selection parameters -----
WINDOW_DAYS = 60          # post-mark window length (days)
P_POS = 1.0               # sample rate for laundering events
P_WINDOW = 1.0            # sample rate for events within WINDOW_DAYS after a mark
P_BG = 0.01               # sample rate for all other (background) events

# ----- burst horizons (INDEPENDENT of WINDOW_DAYS, which only drives event
# selection). Distinct-counterparty fan features are computed over each of these
# short windows; their short/long ratio measures burstiness. -----
BURST_WINDOWS = [1 * 86400, 7 * 86400, 30 * 86400]   # 1d, 7d, 30d (seconds)
BURST_LABELS = ["1d", "7d", "30d"]

# ----- tunable compute caps (bound cost on hub nodes) -----
EGO_CAP = 25              # neighbors expanded per node during 2-hop traversal
BURST_CAP = 200           # tail edges scanned when measuring recent-window bursts
RECIP_CAP = 1000          # cap on set-intersection iteration for reciprocity/mutual

WINDOW_SECS = WINDOW_DAYS * 86400
EPS = 1e-9

# CSV columns
C_TIME, C_DATE, C_SENDER, C_RECEIVER, C_AMOUNT = 0, 1, 2, 3, 4
C_IS_LAUND, C_LAUND_TYPE = 10, 11

_date_cache = {}


def to_epoch(date_str, time_str):
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


def sampled_in(account, ts_str, feat, rate):
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    key = f"{account}|{ts_str}|{feat}".encode()
    return (zlib.crc32(key) & 0xFFFFFFFF) / 4294967296.0 < rate


# ---------------- live graph state ----------------
# Integer node ids for compact sets/lists.
id_map = {}
id_list = []


def nid_of(acct):
    i = id_map.get(acct)
    if i is None:
        i = len(id_list)
        id_map[acct] = i
        id_list.append(acct)
        _init_node(i)
    return i


# Per-node running stats: parallel lists indexed by node id.
out_deg = []      # count of outgoing txns
in_deg = []       # count of incoming txns
tot_out = []      # summed outgoing amount
tot_in = []       # summed incoming amount
max_out = []      # max single outgoing amount
max_in = []       # max single incoming amount
first_ts = []     # first timestamp seen for this node
last_mark = []    # epoch of most recent laundering event in this account (or -1)

out_nbrs = []     # set of out-neighbor ids
in_nbrs = []      # set of in-neighbor ids
out_list = []     # time-ordered list of (ts, neighbor) outgoing
in_list = []      # time-ordered list of (ts, neighbor) incoming


def _init_node(i):
    out_deg.append(0); in_deg.append(0)
    tot_out.append(0.0); tot_in.append(0.0)
    max_out.append(0.0); max_in.append(0.0)
    first_ts.append(-1); last_mark.append(-1)
    out_nbrs.append(set()); in_nbrs.append(set())
    out_list.append([]); in_list.append([])


def insert_edge(u, v, amount, t):
    """Add transaction u -> v of `amount` at time t to the live graph."""
    if first_ts[u] < 0:
        first_ts[u] = t
    if first_ts[v] < 0:
        first_ts[v] = t
    out_deg[u] += 1
    in_deg[v] += 1
    tot_out[u] += amount
    tot_in[v] += amount
    if amount > max_out[u]:
        max_out[u] = amount
    if amount > max_in[v]:
        max_in[v] = amount
    out_nbrs[u].add(v)
    in_nbrs[v].add(u)
    out_list[u].append((t, v))
    in_list[v].append((t, u))


def _windowed_distinct(edge_list, t):
    """Distinct counterparties within each BURST_WINDOWS horizon before t.

    Single capped backward scan over the account's time-ordered edge list; a
    counterparty counts toward a window if its edge is within that window's
    horizon of t. Returns a list aligned with BURST_WINDOWS. The scan stops at
    the longest horizon or after BURST_CAP distinct counterparties.
    """
    longest = BURST_WINDOWS[-1]
    seen_per_window = [set() for _ in BURST_WINDOWS]
    total_seen = set()
    for k in range(len(edge_list) - 1, -1, -1):
        ts, other = edge_list[k]
        age = t - ts
        if age > longest:
            break
        for wi, w in enumerate(BURST_WINDOWS):
            if age <= w:
                seen_per_window[wi].add(other)
        total_seen.add(other)
        if len(total_seen) >= BURST_CAP:
            break
    return [len(s) for s in seen_per_window]


def compute_features(i, t):
    """A + B topology features for node i as of time t (edge already inserted)."""
    od, idg = out_deg[i], in_deg[i]
    to, ti = tot_out[i], tot_in[i]
    onb, inb = out_nbrs[i], in_nbrs[i]
    d_out, d_in = len(onb), len(inb)

    # ---- reciprocity / mutual (2-cycle) via capped intersection ----
    small, large = (onb, inb) if d_out <= d_in else (inb, onb)
    mutual = 0
    for c, x in enumerate(small):
        if c >= RECIP_CAP:
            break
        if x in large:
            mutual += 1
    distinct_total = d_out + d_in - mutual
    reciprocity = mutual / (distinct_total + EPS)

    # ---- 2-hop traversal (capped) : triangles + reach ----
    triangles = 0
    two_hop_out = set()
    for a_i, a in enumerate(onb):
        if a_i >= EGO_CAP:
            break
        oa = out_nbrs[a]
        for b_i, b in enumerate(oa):
            if b_i >= EGO_CAP:
                break
            two_hop_out.add(b)
            if i in out_nbrs[b]:      # i -> a -> b -> i  (directed 3-cycle)
                triangles += 1
    two_hop_in = set()
    for a_i, a in enumerate(inb):
        if a_i >= EGO_CAP:
            break
        for b_i, b in enumerate(in_nbrs[a]):
            if b_i >= EGO_CAP:
                break
            two_hop_in.add(b)

    # fan-in/out distinct counterparties over 1d/7d/30d windows; burstiness =
    # shortest / longest window (spikes when recent activity concentrates).
    fan_out = _windowed_distinct(out_list[i], t)
    fan_in = _windowed_distinct(in_list[i], t)
    fan_out_burstiness = fan_out[0] / (fan_out[-1] + EPS)
    fan_in_burstiness = fan_in[0] / (fan_in[-1] + EPS)
    balanced = min(to, ti) / (max(to, ti) + EPS)   # pass-through / layering balance

    feats = {
        # --- A: local aggregates ---
        "out_degree": od,
        "in_degree": idg,
        "distinct_receivers_out": d_out,
        "distinct_senders_in": d_in,
        "txn_count": od + idg,
        "total_amount_out": round(to, 2),
        "total_amount_in": round(ti, 2),
        "mean_amount_out": round(to / od, 2) if od else 0.0,
        "mean_amount_in": round(ti / idg, 2) if idg else 0.0,
        "max_amount_out": round(max_out[i], 2),
        "max_amount_in": round(max_in[i], 2),
        "in_out_amount_ratio": round(ti / (to + EPS), 4),
        "reciprocity": round(reciprocity, 4),
        "account_age_seconds": (t - first_ts[i]) if first_ts[i] >= 0 else 0,
        # --- B: 2-hop ego / motif ---
        "mutual_count": mutual,
        "triangle_cycle_count": triangles,
        "two_hop_out_reach": len(two_hop_out),
        "two_hop_in_reach": len(two_hop_in),
        "gather_scatter_score": min(d_in, d_out),
        "passthrough_balance": round(balanced, 4),
        "fan_out_burstiness": round(fan_out_burstiness, 4),
        "fan_in_burstiness": round(fan_in_burstiness, 4),
    }
    # windowed fan features: fan_out_1d/7d/30d, fan_in_1d/7d/30d
    for lbl, ov, iv in zip(BURST_LABELS, fan_out, fan_in):
        feats[f"fan_out_{lbl}"] = ov
        feats[f"fan_in_{lbl}"] = iv
    return feats


def stratum_and_rate(is_laund, elapsed):
    if is_laund == 1:
        return "positive", P_POS
    if 0 < elapsed <= WINDOW_SECS:
        return "post_mark", P_WINDOW
    return "background", P_BG


def emit(account, feat, counterparty, amount, t, is_laund, laund_type, elapsed, stratum, i):
    # Join key to preprocessed.json events is
    #   (account, timestamp, feature, counterparty_account, amount)
    # counterparty_account + amount disambiguate same-account/same-second events
    # (an account sending to two different receivers within one second). Both
    # fields exist verbatim in preprocessed.json, so the join is exact.
    rec = {
        "account": account,
        "timestamp": epoch_to_str(t),
        "feature": feat,
        "counterparty_account": counterparty,
        "amount": amount,
        "is_laundering": is_laund,
        "laundering_type": laund_type,
        "elapsed_since_mark": elapsed,
        "stratum": stratum,
    }
    rec.update(compute_features(i, t))
    return rec


def main():
    # sample accounts (force-included for the inspectable sample output)
    try:
        with open(SAMPLE_ACCOUNTS_FROM) as f:
            sample_set = {a["account"] for a in json.load(f)}
    except FileNotFoundError:
        sample_set = set()
    print(f"sample accounts: {len(sample_set)}", file=sys.stderr)

    # ---- load + globally sort transactions by timestamp (stable) ----
    print("loading transactions ...", file=sys.stderr)
    txns = []
    with open(INPUT, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            t = to_epoch(row[C_DATE], row[C_TIME])
            txns.append((t, row[C_SENDER], row[C_RECEIVER], float(row[C_AMOUNT]),
                         1 if row[C_IS_LAUND] == "1" else 0, sys.intern(row[C_LAUND_TYPE])))
    print(f"loaded {len(txns):,} txns; sorting ...", file=sys.stderr)
    txns.sort(key=lambda r: r[0])  # stable: ties keep CSV order (matches preprocess.py)

    n_sel = 0
    sample_rows = []

    def handle_event(account, feat, counterparty, amount, node_id, t,
                     is_laund, laund_type, elapsed, out):
        """Decide selection for one event; write it if sampled and/or if it
        belongs to a sample account. Returns 1 if written to the full output."""
        nonlocal n_sel
        stratum, rate = stratum_and_rate(is_laund, elapsed)
        selected = sampled_in(account, epoch_to_str(t), feat, rate)
        in_sample = account in sample_set
        if not (selected or in_sample):
            return 0
        rec = emit(account, feat, counterparty, amount, t,
                   is_laund, laund_type, elapsed, stratum, node_id)
        if selected:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_sel += 1
        if in_sample:
            r2 = dict(rec)
            r2["selected"] = selected
            sample_rows.append(r2)
        return 1 if selected else 0

    with open(OUT_FULL, "w") as out:
        for k, (t, su, sv, amount, is_laund, laund_type) in enumerate(txns, 1):
            u = nid_of(su)
            v = nid_of(sv)

            # causal look-back BEFORE this event updates the mark
            es = 0 if is_laund else (t - last_mark[u] if last_mark[u] >= 0 else -1)
            er = 0 if is_laund else (t - last_mark[v] if last_mark[v] >= 0 else -1)
            if is_laund:
                last_mark[u] = t
                last_mark[v] = t

            # insert edge so the current (known) transaction is part of the features
            insert_edge(u, v, amount, t)

            # SEND on su -> counterparty is the receiver sv; RECEIVE on sv ->
            # counterparty is the sender su (matches preprocessed.json).
            handle_event(su, "TRANS_SEND", sv, amount, u, t, is_laund, laund_type, es, out)
            handle_event(sv, "TRANS_RECEIVE", su, amount, v, t, is_laund, laund_type, er, out)

            if k % 1_000_000 == 0:
                print(f"  processed {k:,} txns, {n_sel:,} selected events", file=sys.stderr)

    with open(OUT_SAMPLE, "w") as f:
        json.dump(sample_rows, f, indent=2)

    print(f"Done: {n_sel:,} selected events -> {OUT_FULL}", file=sys.stderr)
    print(f"      {len(sample_rows):,} sample-account events -> {OUT_SAMPLE}", file=sys.stderr)


if __name__ == "__main__":
    main()
