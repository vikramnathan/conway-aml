# AML results summary

All on the held-out test accounts (deterministic 80/20 stratified split, seed 0).
PRAGMA-S backbone pretrained with MLM to step 17,600. Windowed label
`y = 1[0 <= elapsed_since_mark < T]`, T = 30 days.

## Steps 1 & 2 — per-event classification (embedding probe)

Matched comparison on the **same 95,705 test events** (the graph-selected join set);
only the feature vector differs. Frozen backbone embeddings + linear head.

| Metric | PRAGMA-only | PRAGMA + graph | Δ |
|---|---|---|---|
| F0.5 | 0.530 | **0.643** | +21% |
| F1 | 0.621 | 0.688 | +11% |
| PR-AUC | 0.519 | 0.670 | +29% |
| ROC-AUC | 0.709 | 0.821 | +16% |

Graph topology features (28 dims: degrees, motifs, fan-in/out bursts, reciprocity,
etc.) give a large, consistent lift — consistent with the paper's §3.4.5 point that
PRAGMA alone is blind to cross-record network structure, which AML depends on.

Files: `baseline_results.txt`, `graph_results.txt`, `per_event_scores.csv`.

## Step 3 — guilty-group clustering

Guilty events (predicted = score ≥ F0.5 threshold; GT = label==1) are merged into
groups: (n,t) links to (m,t') when m is n's counterparty and |t−t'| < T; groups are
the transitive closure (chains extend groups). Predicted groups matched to GT groups
by node+time-span overlap; penalty = 1 − Jaccard(nodes) if matched else 1.

| | PRAGMA-only | PRAGMA + graph |
|---|---|---|
| precision penalty (per predicted group) | 0.603 | **0.464** |
| recall penalty (per GT group) | 0.273 | 0.391 |
| combined | 0.491 | 0.431 |
| predicted groups | 44,516 | 27,936 |
| GT groups | 22,918 | 22,918 |
| predicted matched | 19,719 | 16,747 |
| GT matched | 16,695 | 14,066 |

Lower penalty = better. The +graph model has a much lower **precision penalty**
(0.46 vs 0.60): its predicted groups align far better with true groups because it
raises fewer, cleaner false-positive flags (27.9K vs 44.5K predicted groups). The
PRAGMA-only model flags more events, catching slightly more GT groups (higher recall)
but at the cost of many spurious groups (worse precision). Combined penalty favors
+graph (0.431 vs 0.491).

> Note: precision/recall penalty trade off with the flag threshold. Both models use
> their own F0.5-optimal per-event threshold here; sweeping the threshold would trace
> a penalty curve. The group counts differ because PRAGMA-only's threshold admits ~55%
> more guilty events.

## Reproduce

```
modal run train_modal.py --steps 20000 --ckpt-every 100   # MLM pretrain (already done -> pragma_S_latest.pt)
modal run matched_modal.py --ckpt pragma_S_latest.pt --head-epochs 40   # steps 1+2
modal run cluster_modal.py --model graph  --t-days 30      # step 3 (graph)
modal run cluster_modal.py --model pragma --t-days 30      # step 3 (pragma-only)
```
