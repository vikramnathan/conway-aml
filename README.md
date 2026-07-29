# PRAGMA for AML — replication guide

Implementation of the PRAGMA foundation model (arXiv:2604.08649) for banking
event sequences, plus a downstream Anti-Money-Laundering (AML) evaluation on the
[SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
transaction dataset. Three deliverables:

1. **Baseline** — PRAGMA pretrained with masked modelling (MLM); a linear probe
   on the frozen embeddings classifies each event as in-a-laundering-window.
2. **+graph** — the same probe with per-event graph-topology features concatenated.
3. **Clustering** — guilty events are merged into "cases" (connected components)
   and scored against ground-truth cases.

Findings and the results table are in [`REPORT.md`](REPORT.md).

## Prerequisites

- Python 3.12, a [Modal](https://modal.com) account (`pip install modal`).
- `modal token new` (interactive, once) to authenticate.
- `SAML-D.csv` in the repo root — download from Kaggle:
  https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
  (the dataset is not tracked in git; see `.gitignore`).
- Training/eval run on Modal GPUs (A100); no local GPU needed. `torch`/`numpy`
  are only required inside the Modal image, not locally.

## Repo layout

```
pragma/                  the model + all logic (importable package)
  config.py              PRAGMAConfig (S/M/L, paper Table 1), VocabConfig, MaskingConfig
  batch.py               PragmaBatch — the tensor contract between data and model
  embeddings.py          shared K+V table, within-field PosEmb, continuous-time RoPE, calendar MLP
  encoder.py             pre-norm bidirectional transformer block (RoPE, SDPA)
  model.py               PRAGMA backbone: Profile -> Event (independent) -> History encoders
  mlm.py                 MLM head + 3-source masking (paper §2.3.5)
  tokenizer.py           raw K/V + profiles -> token ids; JSONL/array loaders
  data.py                MLM Dataset + collate (truncation §2.4)
  aml.py                 downstream AML head (per-event + decision-point modes)
  aml_data.py            decision-point dataset (prefix up to "now", no leakage)
  split.py               deterministic 80/20 stratified split by account
  metrics.py             ROC-AUC, PR-AUC, F-beta (no sklearn)
  clustering.py          guilty-group closure + group-matching penalty

preprocess.py            SAML-D.csv        -> preprocessed.json   (per-account events)
build_profiles.py        preprocessed.json -> profiles.json       (static + lifelong)
build_graph_features.py  SAML-D.csv        -> graph_features.json (28 topology feats)

train_modal.py           MLM pretraining          -> pragma_S_latest.pt
matched_modal.py         baseline + graph probes  -> baseline_results.txt, graph_results.txt
cluster_modal.py         guilty-group clustering  -> cluster_results_{model}.txt
run_tests_modal.py       CPU smoke suite in the Modal image
tests/test_smoke.py      end-to-end unit/integration tests
```

## Step 0 — build the data artifacts (local, ~a few minutes)

```bash
python preprocess.py             # SAML-D.csv -> preprocessed.json  (JSONL, one account/line)
python build_profiles.py         # -> profiles.json                 (JSONL, one profile/line)
python build_graph_features.py   # -> graph_features.json           (JSONL, ~486K selected events)
```

`preprocessed.sample.json` / `profiles.sample.json` / `graph_features.sample.json`
are 50-account JSON arrays used by the smoke tests and the `--sample` dry-run.

## Step 1 — upload the big files to a Modal Volume (once)

Training reads data from the `pragma-data` Volume (files are multi-GB, so use
`modal volume put`, not the in-memory upload path):

```bash
modal volume create pragma-data          # no-op if it exists
modal volume put pragma-data preprocessed.json   preprocessed.json   --force
modal volume put pragma-data profiles.json       profiles.json       --force
modal volume put pragma-data graph_features.json graph_features.json --force
```

## Step 2 — smoke test (optional, ~4 min, CPU)

```bash
modal run run_tests_modal.py             # runs tests/test_smoke.py inside the image
```

## Step 3 — MLM pretraining -> backbone checkpoint

```bash
modal run train_modal.py --steps 20000 --ckpt-every 100 --size S
# writes /vol/checkpoints/pragma_S_latest.pt (overwritten every 100 steps)
```

Reproduces the deterministic 80/20 stratified split (seed 0), fits the tokenizer
on the train split only, and pretrains PRAGMA-S with the MLM objective. The
held-out test accounts are embedded in the checkpoint for leakage-free downstream
eval. (Use `--detach` to keep the run alive if your client disconnects.)

## Step 4 — baseline + graph probes (steps 1 & 2 of the study)

```bash
modal run --detach matched_modal.py --ckpt pragma_S_latest.pt --head-epochs 40
```

Joins the 486K graph-selected events to their PRAGMA event prefixes on the exact
5-tuple `(account, timestamp, feature, counterparty_account, amount)`, embeds each
once with the frozen backbone, then trains two linear heads on the **same** rows /
**same** split — PRAGMA-only and PRAGMA+graph. Writes to `/vol/results/`:
`baseline_results.txt`, `graph_results.txt`, and `per_event_scores.csv`
(per-test-event scores for both models, consumed by clustering).

Quick end-to-end dry run on the 50-account sample (random-init backbone, seconds):

```bash
modal run matched_modal.py --sample
```

## Step 5 — guilty-group clustering (step 3 of the study)

```bash
modal run cluster_modal.py --model graph  --criterion f1 --t-days 30
modal run cluster_modal.py --model pragma --criterion f1 --t-days 30
# writes /vol/results/cluster_results_{graph,pragma}.txt
```

Flags events at each model's F1-optimal threshold (computed from the per-event
scores; balances precision and recall). `--criterion f2|f0.5` switches the objective
(f2 = recall-leaning for a fraud-catching operating point); `--threshold` overrides
it outright. `--t-days` is the merge window T.

## Step 6 — pull results locally

```bash
for f in baseline_results.txt graph_results.txt \
         cluster_results_graph.txt cluster_results_pragma.txt per_event_scores.csv; do
  modal volume get pragma-data results/$f $f --force
done
```

## Key knobs

| Flag | Where | Meaning |
|---|---|---|
| `--size S\|M\|L` | train_modal | PRAGMA family size (paper Table 1) |
| `--steps`, `--ckpt-every` | train_modal | MLM steps; checkpoint cadence |
| `--t-days` | matched, cluster | window T for the label `0<=e<T` and the merge relation |
| `--w` | matched | positive-class loss weight (default 10) |
| `--head-epochs` | matched | linear-probe epochs on cached embeddings |
| `--model graph\|pragma` | cluster | which score column to cluster |
| `--criterion f1\|f2\|f0.5` | cluster | F-beta objective for the flag threshold (default f1) |
| `--threshold` | cluster | override the flag threshold outright |

Determinism: the account split (seed 0), tokenizer fit, and graph event selection
(crc32-based) are all reproducible, so re-runs land on the same train/test rows.
