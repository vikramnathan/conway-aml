# Graph Topology Features

Cumulative-causal graph features per labeled event, built to concatenate with
PRAGMA per-event embeddings before the AML classification head. These exist to
address the paper's stated limitation: PRAGMA processes each account's history
in isolation and is blind to cross-record network structure (§3.4.5), which is
exactly what AML detection depends on.

## Files

| File | Format | Contents |
|------|--------|----------|
| `build_graph_features.py` | script | Builds the features from `SAML-D.csv`. |
| `graph_features.json` | JSON Lines | One selected event per line (~486K rows). |
| `graph_features.sample.json` | JSON array | All events of the 50 sample accounts, each tagged `selected`. |

Regenerate: `python3 build_graph_features.py`  (~2 min, single pass).

## What each record is

One record = one **event** (one endpoint of one transaction). Every transaction
produces two events: `TRANS_SEND` on the sender, `TRANS_RECEIVE` on the receiver.
Features are computed **as of that event's timestamp** over the subgraph of all
edges with `timestamp <= t` — cumulative and causal, no future leakage.

```jsonc
{
  // ---- join key (unique; see below) ----
  "account": "8600542721",
  "timestamp": "2022-10-12 17:54:07",
  "feature": "TRANS_SEND",                 // TRANS_SEND | TRANS_RECEIVE
  "counterparty_account": "8804315251",
  "amount": 3355.95,
  // ---- labels / meta ----
  "is_laundering": 1,                      // 0 | 1  (training label)
  "laundering_type": "Behavioural_Change_1", // for per-typology eval slices
  "elapsed_since_mark": 0,                 // causal look-back to last laundering event in this account
  "stratum": "positive",                   // positive | post_mark | background
  // ---- A: local aggregates ----
  "out_degree": 21, "in_degree": 16,
  "distinct_receivers_out": 15, "distinct_senders_in": 14,
  "txn_count": 37,
  "total_amount_out": 54616.02, "total_amount_in": 81357.49,
  "mean_amount_out": 2600.76, "mean_amount_in": 5084.84,
  "max_amount_out": 9344.68, "max_amount_in": 18469.36,
  "in_out_amount_ratio": 1.4896, "reciprocity": 0.381,
  "account_age_seconds": 450986,
  // ---- B: 2-hop ego / motif ----
  "mutual_count": 8, "triangle_cycle_count": 0,
  "two_hop_out_reach": 1, "two_hop_in_reach": 1,
  "gather_scatter_score": 14, "passthrough_balance": 0.6713,
  "fan_out_burstiness": 0.2667, "fan_in_burstiness": 0.2143,
  "fan_out_1d": 4, "fan_in_1d": 3,
  "fan_out_7d": 15, "fan_in_7d": 14,
  "fan_out_30d": 15, "fan_in_30d": 14
}
```

## The join key

`graph_features.json` **is** the demarcation of which events have graph data —
the training/eval loop iterates this file (~486K rows), never the full 19M events.
Each row joins to its PRAGMA event embedding on the 5-tuple:

```
(account, timestamp, feature, counterparty_account, amount)
```

All five fields exist verbatim in `preprocessed.json` events (identical timestamp
formatting). The 5-tuple is verified **unique** across all records. The first
three fields alone are NOT unique — an account can send to two different
counterparties in the same 1-second timestamp — so `counterparty_account` and
`amount` are required to disambiguate.

### Join recipe

Collect the keys first, then stream `preprocessed.json` once and keep only matches
(avoids holding 19M events in memory):

```python
import json

def key(r):
    return (r["account"], r["timestamp"], r["feature"],
            r["counterparty_account"], r["amount"])

# 1. Load graph features, indexed by join key.
gf = {}
with open("graph_features.json") as f:
    for line in f:
        r = json.loads(line)
        gf[key(r)] = r

# 2. Stream PRAGMA events; keep only those we have graph data for.
GRAPH_COLS = [  # feature columns to concatenate, in fixed order
    "out_degree","in_degree","distinct_receivers_out","distinct_senders_in",
    "txn_count","total_amount_out","total_amount_in","mean_amount_out",
    "mean_amount_in","max_amount_out","max_amount_in","in_out_amount_ratio",
    "reciprocity","account_age_seconds","mutual_count","triangle_cycle_count",
    "two_hop_out_reach","two_hop_in_reach","gather_scatter_score",
    "passthrough_balance","fan_out_burstiness","fan_in_burstiness",
    "fan_out_1d","fan_in_1d","fan_out_7d","fan_in_7d","fan_out_30d","fan_in_30d",
]

examples = []
with open("preprocessed.json") as f:
    for line in f:
        acct = json.loads(line)
        for ev in acct["events"]:
            k = key(ev)                       # ev has account? add it:
            # NOTE: preprocessed.json events carry counterparty_account & amount,
            # but 'account' and 'feature' live on the event too. Build k accordingly.
            if k in gf:
                g = gf[k]
                graph_vec = [g[c] for c in GRAPH_COLS]
                pragma_emb = run_pragma(acct, ev)   # your embedding lookup
                examples.append({
                    "x": pragma_emb + graph_vec,    # concatenated feature vector
                    "y": g["is_laundering"],
                    "stratum": g["stratum"],
                })
```

> The `account` field: in `preprocessed.json` it's on the account object, not each
> event, so build the event's key as
> `(acct["account"], ev["timestamp"], ev["feature"], ev["counterparty_account"], ev["amount"])`.

## Event selection (strata)

Not every event has graph data — that would be 19M expensive motif computations.
Events are selected by stratum, all rates tunable at the top of the script:

| Stratum | Definition | Rate | Count |
|---------|-----------|------|-------|
| `positive` | `is_laundering == 1` | `P_POS = 1.0` | 19,746 |
| `post_mark` | `0 < elapsed_since_mark <= WINDOW_DAYS` (60d) | `P_WINDOW = 1.0` | 278,958 |
| `background` | everything else | `P_BG = 0.01` | 187,288 |
| **total** | | | **485,992** |

Sampling is deterministic (crc32 of `account|timestamp|feature`), so re-runs are
reproducible without an RNG seed. Use `stratum` to reweight or slice eval; note
the label is on the edge, so both endpoints of a laundering transaction are
positives (hence ~19.7K positives from ~9.9K laundering rows).

## Feature notes for the classification head

- **Scales are raw and heavy-tailed** (degrees, amounts). Standard-scale or
  `log1p` the concatenated vector before the linear head — the paper standard-
  scales PRAGMA embeddings for the same reason.
- **Compute caps** (`EGO_CAP=25`, `BURST_CAP=200`, `RECIP_CAP=1000`) bound
  traversal cost on hub accounts, so `triangle_cycle_count` / reach counts on the
  largest hubs are lower bounds. Raise them for exact counts at higher cost.
- **Burst windows** (`fan_*_1d/7d/30d`) are independent of `WINDOW_DAYS`; they are
  monotonic (`1d <= 7d <= 30d`). `fan_*_burstiness = 1d / 30d` spikes when recent
  activity concentrates (targets `Fan_Out`, `Smurfing`, `Gather-Scatter`).
- **Lifetime vs windowed:** only the fan features are windowed; cycle/reach/
  reciprocity/amount features are cumulative over all edges `<= t`.
