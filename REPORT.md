# PRAGMA for AML

We adapt the PRAGMA foundation model (an encoder-only transformer pretrained with
masked modelling on banking event sequences) to Anti-Money-Laundering detection on
the SAML-D dataset, and test whether explicit graph-topology features close the gap
the PRAGMA paper itself flags: that the model, processing each account in isolation,
is blind to the cross-record network structure AML depends on (§3.4.5).

All results are on held-out **test accounts** from a deterministic 80/20 stratified
split (stratified on guilty vs. innocent accounts so rare laundering accounts appear
in both partitions). The label is windowed: an event is positive iff its
`elapsed_since_mark` `e` satisfies `0 <= e < T`, with `T = 30 days` — i.e. the event
falls within T of the account's laundering mark.

## (a) Baseline model — PRAGMA embedding probe

**Backbone.** PRAGMA-S (~4M params on this vocab) is pretrained with the masked
language modelling (MLM) objective from the paper: events are tokenised as
(key, value, time) triples, a shared embedding table plus within-field positional
encodings feed a two-branch encoder (a per-event Event Encoder and a per-account
History Encoder, both bidirectional with continuous-time RoPE), and a masked-token
head reconstructs corrupted values. We pretrain to 17,600 steps on ~684K training
accounts. The backbone is then **frozen**.

**Downstream head.** Following the paper's embedding-probe protocol, we freeze the
backbone and train a single linear layer on top. The subtlety is leakage: PRAGMA's
History Encoder is bidirectional, so if we fed a whole account and read a mid-history
event, that event would attend to *future* events — including post-mark activity that
trivially reveals the label. We avoid this the way an MLM encoder is used for
classification: for each event we treat it as a **decision point** and feed only the
event **prefix up to "now"** (`events[0..i]`), then read the contextual `[EVT]`
representation at that final event. Attention stays bidirectional over the prefix (in
distribution with pretraining), but nothing after the decision point is visible.

**Training points.** Positives (window events) are rare, so we sample every positive
plus 1% of negatives per account, and weight the positive class by `W = 10` in the
BCE loss. A linear probe (logistic regression) is fit on the frozen embeddings.

## (b) The +graph variant

PRAGMA sees one account's history at a time; it never sees that the same counterparty
is fanning money across dozens of accounts. The +graph variant adds this missing
signal by concatenating a **28-dimensional graph-topology feature vector** to each
event's PRAGMA embedding before the same linear head. The head is otherwise identical
and is trained on the **same events, same split** — so any difference is attributable
purely to the graph features.

Features are computed **cumulatively and causally** — over the transaction subgraph of
all edges with `timestamp <= t`, so there is no future leakage — for each of the ~486K
selected events, and joined to the PRAGMA embedding on the exact 5-tuple
`(account, timestamp, feature, counterparty_account, amount)`. The 28 features, grouped:

- **Local aggregates (A):** out/in degree, distinct receivers-out / senders-in,
  txn count, total / mean / max amount out & in, in/out amount ratio, reciprocity,
  account age.
- **2-hop ego & motif (B):** mutual-neighbour count, triangle/cycle count, 2-hop
  out- and in-reach, gather-scatter score, passthrough balance.
- **Burst windows:** fan-out / fan-in counts over 1d / 7d / 30d, and fan-in/out
  burstiness (recent-vs-lifetime concentration).

These target classic laundering typologies (fan-in/out, smurfing, gather-scatter,
pass-through, cycles). Heavy-tailed columns (degrees, amounts, counts) are `log1p`-ed;
all features are standard-scaled using **train-split statistics only**.

## (c) Results — baseline vs. +graph (per-event)

All numbers on the 95,705 held-out test events. We report **F2 as the primary
metric**: in AML a missed launderer is far costlier than a false alarm, so the
operating point should prioritise recall. (The PRAGMA paper reports F0.5, which
emphasises precision; we deliberately depart from that because the operational
objective here is catching fraud.) F1 / F0.5 are shown for reference; PR-AUC and
ROC-AUC are threshold-independent.

| Metric | PRAGMA-only | PRAGMA + graph | Δ |
|---|---|---|---|
| **F2** (primary) | 0.794 | **0.818** | +3% |
| F1 | 0.621 | **0.688** | +11% |
| F0.5 | 0.530 | **0.643** | +21% |
| PR-AUC | 0.519 | **0.670** | +29% |
| ROC-AUC | 0.709 | **0.821** | +16% |

Graph topology features lift every metric — a direct confirmation of the paper's
§3.4.5 limitation: the relational signal PRAGMA cannot see in isolation is exactly
what AML detection needs. The gap is largest on the precision-sensitive metrics
(F0.5, PR-AUC): both models can be pushed to high recall, but the graph features
sharply reduce the false-positive cost of doing so.

## (d) Case clustering — method and results

Detection at the event level is only part of AML — investigators care about **cases**:
the group of accounts collaborating in one laundering episode. We reconstruct cases
from event-level flags via a transitive closure.

**Method.**

- **Guilty events.** Predicted-guilty = model score >= threshold (each model's
  F1-optimal threshold, balancing precision and recall); ground-truth-guilty =
  actual label == 1. Predicted and ground-truth groups are built independently by
  the same procedure.
- **Merge relation.** A guilty event `(n, t)` (node `n`, time `t`) links to a guilty
  event `(m, t')` when `m` is the counterparty of `n`'s guilty transaction **and**
  `|t - t'| < T`. Vertices are individual guilty *events*.
- **Groups = connected components** of that relation (union-find). The closure is
  **transitive**, so chains extend a group: `(n,t)-(m,t')-(p,t'')` puts `n` and `p`
  in one group even when `|t - t''| > T`. A group's node set is the union of its
  events' nodes; its time span is `[min t, max t]`. Because vertices are events, a
  node whose guilty activity is far apart in time (and doesn't chain) can appear in
  more than one group — but at any single instant it belongs to exactly one.
- **Scoring.** Each predicted group is matched to its best ground-truth group,
  where a candidate match requires **node-set overlap AND time-span overlap**; the
  best match maximises node-set Jaccard. The penalty is

  ```
  penalty(P) = 1 - Jaccard(nodes_P, nodes_G)   if a match exists
  penalty(P) = 1                                if no match (a miss)
  ```

  We report the mean penalty over predicted groups (**precision penalty**, the
  headline), the symmetric mean over ground-truth groups (**recall penalty**, which
  exposes missed cases), and their combination. Lower is better.

**Results.** Each model is thresholded at its **own** F1-optimal per-event cutoff
(balancing precision and recall) — these are **intentionally different** (0.785 vs
0.823). We treat the two as independent detectors, each deployed at its own best
operating point, which is how they would run in practice; we do not force a shared
threshold.

| | PRAGMA-only | PRAGMA + graph |
|---|---|---|
| flag threshold (F1-optimal, per model) | 0.785 | 0.823 |
| recall penalty (per GT group) ↓ | **0.084** | 0.168 |
| precision penalty (per predicted group) ↓ | 0.627 | **0.534** |
| combined ↓ | 0.475 | **0.407** |
| predicted groups | 58,807 | 43,122 |
| ground-truth groups | 22,918 | 22,918 |
| predicted matched | 24,015 | 22,183 |
| GT matched | 21,042 | 19,224 |

At the F1 operating point the +graph model gives markedly cleaner cases (precision
penalty 0.534 vs 0.627) while still recovering most ground-truth groups (recall
penalty 0.168; 19.2K of 22,918 matched). It reaches this with far fewer spurious
cases (43.1K vs 58.8K predicted groups), so its combined penalty is clearly lower
(0.407 vs 0.475). PRAGMA-only edges out on raw recall penalty (0.084) only because it
over-flags — at the cost of ~16K more predicted groups and worse precision.

## Takeaways

- PRAGMA embeddings transfer to AML as a genuinely useful representation
  (ROC-AUC 0.71 from a frozen linear probe, no leakage).
- **Graph-topology features are the decisive ingredient**: +3–29% across per-event
  metrics (largest on precision-sensitive ones) and, at the F1 operating point,
  cleaner case clusters (combined penalty 0.407 vs 0.475) — consistent with the
  paper's finding that network/relational signal is PRAGMA's missing piece.

## Caveats

- PRAGMA-S at 17,600 MLM steps (not fully converged; not scaled to M/L).
- Each model is clustered at its own F1-optimal threshold (0.785 vs 0.823) — by
  design, since we compare two detectors each at its own best operating point. They
  therefore sit at different precision/recall points; the threshold-independent
  per-event metrics (PR-AUC, ROC-AUC) carry the unambiguous quality comparison.
