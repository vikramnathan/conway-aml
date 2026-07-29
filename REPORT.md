# PRAGMA for AML — brief report

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

## (c) Clustering method

Detection at the event level is only part of AML — investigators care about **cases**:
the group of accounts collaborating in one laundering episode. We reconstruct cases
from event-level flags via a transitive closure.

- **Guilty events.** Predicted-guilty = model score >= threshold (each model's
  F0.5-optimal threshold); ground-truth-guilty = actual label == 1. Predicted and
  ground-truth groups are built independently by the same procedure.
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

## (d) Results

All numbers on the 95,705 held-out test events (per-event) / their induced groups.

### Per-event classification (steps 1 & 2)

| Metric | PRAGMA-only | PRAGMA + graph | Δ |
|---|---|---|---|
| F0.5 | 0.530 | **0.643** | +21% |
| F1 | 0.621 | **0.688** | +11% |
| PR-AUC | 0.519 | **0.670** | +29% |
| ROC-AUC | 0.709 | **0.821** | +16% |

Graph topology features lift every metric substantially — a direct confirmation of
the paper's §3.4.5 limitation: the relational signal PRAGMA cannot see in isolation
is exactly what AML detection needs.

### Guilty-group clustering (step 3)

Each model is thresholded at its **own** F0.5-optimal per-event cutoff — these are
**intentionally different** (0.863 vs 0.909). We treat the two as independent
detectors, each deployed at its own best operating point, which is how they would
run in practice; we do not force a shared threshold.

| | PRAGMA-only | PRAGMA + graph |
|---|---|---|
| flag threshold (F0.5-optimal, per model) | 0.863 | 0.909 |
| precision penalty (per predicted group) ↓ | 0.603 | **0.464** |
| recall penalty (per GT group) ↓ | **0.273** | 0.391 |
| combined ↓ | 0.491 | **0.431** |
| predicted groups | 44,516 | 27,936 |
| ground-truth groups | 22,918 | 22,918 |
| predicted matched | 19,719 | 16,747 |
| GT matched | 16,695 | 14,066 |

The +graph model produces markedly cleaner cases (precision penalty 0.46 vs 0.60):
it raises fewer, more precise flags (27.9K vs 44.5K predicted groups), so its
reconstructed cases align far better with the truth. PRAGMA-only over-flags —
catching slightly more ground-truth groups (better recall penalty) at the cost of
many spurious cases. On the combined penalty, +graph wins (0.431 vs 0.491).

Because each detector sits at its own threshold, it operates at a different
precision/recall point (PRAGMA-only flags ~55% more events), so the precision-vs-recall
split partly reflects those operating points, not embedding quality alone. This is
deliberate — we tune each model to its own optimum. The threshold-independent per-event
metrics above (PR-AUC, ROC-AUC) already establish the quality gap unambiguously.

### Takeaways

- PRAGMA embeddings transfer to AML as a genuinely useful representation
  (ROC-AUC 0.71 from a frozen linear probe, no leakage).
- **Graph-topology features are the decisive ingredient**, adding 16–29% across
  per-event metrics and yielding materially cleaner case clusters — consistent with
  the paper's own finding that network/relational signal is PRAGMA's missing piece.

### Caveats

- PRAGMA-S at 17,600 MLM steps (not fully converged; not scaled to M/L).
- Each model is clustered at its own F0.5-optimal threshold (0.863 vs 0.909) — by
  design, since we compare two detectors each at its own best operating point. As a
  result they sit at different precision/recall points; the threshold-independent
  per-event metrics (PR-AUC, ROC-AUC) carry the unambiguous quality comparison.
