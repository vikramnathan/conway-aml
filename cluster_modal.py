"""Step 3: guilty-group clustering + group-matching evaluation on the test set.

Reads per_event_scores.csv (from matched_modal.py). Each row is a test event with
node (account), timestamp, counterparty, label, and model scores. Then:

  1. Guilty events: predicted = score >= threshold; ground-truth = label == 1.
  2. Build guilty GROUPS as the transitive closure of guilty events under
     "counterparty guilty within T": (n,t) links to (m,t') when m is n's
     counterparty and |t-t'| < T. Groups are connected components.
  3. Score predicted groups vs GT groups: penalty = 1 - Jaccard(nodes) if a group
     matches (node + time-span overlap) else 1. Report mean precision/recall.

Everything needed is in the CSV (counterparty included), so no corpus load.

    modal run cluster_modal.py --model graph --t-days 30
"""

from __future__ import annotations

import modal

app = modal.App("pragma-cluster")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy<2.0", "torch==2.5.1")  # torch pulled in transitively by pragma/__init__
    .add_local_python_source("pragma")
)

vol = modal.Volume.from_name("pragma-data", create_if_missing=True)
VOL_PATH = "/vol"


@app.function(image=image, volumes={VOL_PATH: vol}, memory=16 * 1024, timeout=60 * 60)
def cluster_eval(
    scores_csv: str = "results/per_event_scores.csv",
    model: str = "graph",            # "graph" | "pragma"
    threshold: float | None = None,   # None -> {criterion}-optimal threshold from results file
    criterion: str = "f2",           # which F-beta threshold to operate at: f2 | f1 | f0.5
    t_days: float = 30.0,             # T for the merge relation (match the label window)
):
    import csv

    import numpy as np

    from pragma.clustering import build_groups, match_penalty, guilty_events_from_rows
    from pragma.metrics import best_fbeta

    T = t_days * 24 * 3600.0
    score_col = f"score_{model}"

    rows = []
    with open(f"{VOL_PATH}/{scores_csv}") as f:
        rows = list(csv.DictReader(f))
    print(f"[scores] {len(rows)} test rows; column={score_col}")

    # Compute the operating threshold directly from these scores (self-contained;
    # independent of any stale results file). Default criterion = F2 (recall-leaning:
    # for fraud we prefer catching launderers over avoiding false alarms).
    if threshold is None:
        beta = {"f2": 2.0, "f1": 1.0, "f0.5": 0.5}[criterion]
        s = np.array([float(r[score_col]) for r in rows])
        y = np.array([int(r["label"]) for r in rows])
        _, threshold = best_fbeta(s, y, beta)
        print(f"[threshold] {criterion}-optimal threshold = {threshold:.4f}")
    else:
        print(f"[threshold] provided = {threshold:.4f}")

    pred_rows = [r for r in rows if float(r[score_col]) >= threshold]
    gt_rows = [r for r in rows if int(r["label"]) == 1]
    print(f"[guilty] predicted_events={len(pred_rows)} gt_events={len(gt_rows)}")

    pred_ev = guilty_events_from_rows(pred_rows)
    gt_ev = guilty_events_from_rows(gt_rows)

    pred_groups = build_groups(pred_ev, T)
    gt_groups = build_groups(gt_ev, T)
    res = match_penalty(pred_groups, gt_groups)
    print(f"[groups] pred={res['n_pred_groups']} gt={res['n_gt_groups']} "
          f"pred_matched={res['n_pred_matched']} gt_matched={res['n_gt_matched']}")
    print(f"[penalty] precision={res['precision_penalty']:.4f} "
          f"recall={res['recall_penalty']:.4f} combined={res['combined_penalty']:.4f}")

    out = f"{VOL_PATH}/results/cluster_results_{model}.txt"
    with open(out, "w") as fs:
        fs.write(f"Guilty-group clustering eval (model={model}, T={t_days}d, "
                 f"threshold={threshold:.4f})\n\n")
        for k, v in res.items():
            fs.write(f"{k}: {v}\n")
    vol.commit()
    print(f"[done] wrote {out}")
    return res


@app.local_entrypoint()
def main(model: str = "graph", criterion: str = "f2", t_days: float = 30.0, threshold: float = -1.0):
    thr = None if threshold < 0 else threshold
    res = cluster_eval.remote(model=model, criterion=criterion, t_days=t_days, threshold=thr)
    print("[result]", res)
