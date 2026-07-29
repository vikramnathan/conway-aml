"""Matched AML pipeline (steps 1 & 2): PRAGMA-only vs PRAGMA+graph, same events.

Protocol (embedding probe, §3.1.1):
  1. Take the 486K graph-selected events (graph_features.json). Each is a decision
     point: feed the account's event prefix up to that event, read the frozen
     backbone's contextual [EVT] representation at that event (no future leakage).
  2. Cache, per selected event: PRAGMA embedding (d), graph feature vector (28),
     windowed label y = 1[0 <= elapsed_since_mark < T], stratum, and join key.
  3. Train two linear heads on the SAME cached rows / SAME split:
        - PRAGMA-only      -> baseline_results.txt
        - PRAGMA + graph   -> graph_results.txt
     Difference is ONLY the graph features => clean ablation.

Split reuses the deterministic account-level 80/20 stratified split, so no
test account leaks into training for either head.

    modal run matched_modal.py --ckpt pragma_S_latest.pt --epochs 30
"""

from __future__ import annotations

import modal

app = modal.App("pragma-matched")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy<2.0")
    .add_local_python_source("pragma")
    # sample files for a fast end-to-end validation run (--records-name *.sample.json)
    .add_local_file("preprocessed.sample.json", f"{'/vol_samples'}/preprocessed.sample.json")
    .add_local_file("profiles.sample.json", "/vol_samples/profiles.sample.json")
    .add_local_file("graph_features.sample.json", "/vol_samples/graph_features.sample.json")
)

vol = modal.Volume.from_name("pragma-data", create_if_missing=True)
VOL_PATH = "/vol"

GRAPH_COLS = [
    "out_degree", "in_degree", "distinct_receivers_out", "distinct_senders_in",
    "txn_count", "total_amount_out", "total_amount_in", "mean_amount_out",
    "mean_amount_in", "max_amount_out", "max_amount_in", "in_out_amount_ratio",
    "reciprocity", "account_age_seconds", "mutual_count", "triangle_cycle_count",
    "two_hop_out_reach", "two_hop_in_reach", "gather_scatter_score",
    "passthrough_balance", "fan_out_burstiness", "fan_in_burstiness",
    "fan_out_1d", "fan_in_1d", "fan_out_7d", "fan_in_7d", "fan_out_30d", "fan_in_30d",
]
# graph columns that are heavy-tailed counts/amounts -> log1p before scaling
LOG1P_COLS = {
    "out_degree", "in_degree", "distinct_receivers_out", "distinct_senders_in",
    "txn_count", "total_amount_out", "total_amount_in", "mean_amount_out",
    "mean_amount_in", "max_amount_out", "max_amount_in", "account_age_seconds",
    "mutual_count", "triangle_cycle_count", "two_hop_out_reach", "two_hop_in_reach",
    "gather_scatter_score", "fan_out_1d", "fan_in_1d", "fan_out_7d", "fan_in_7d",
    "fan_out_30d", "fan_in_30d",
}


@app.function(image=image, gpu="A100-40GB", volumes={VOL_PATH: vol},
              memory=128 * 1024, timeout=12 * 60 * 60)
def run(
    data_dir: str = VOL_PATH,
    records_name: str = "preprocessed.json",
    profiles_name: str = "profiles.json",
    graph_name: str = "graph_features.json",
    ckpt: str = "pragma_S_latest.pt",
    size: str = "S",
    t_days: float = 30.0,
    seed: int = 0,
    embed_batch: int = 64,
    head_epochs: int = 30,
    head_lr: float = 1e-2,
    w: float = 10.0,
    use_profile: bool = True,
):
    import json
    import os
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from pragma.config import PRAGMAConfig
    from pragma.tokenizer import fit_tokenizer, load_records, load_profiles
    from pragma.split import stratified_split, split_summary, is_guilty
    from pragma.model import PRAGMA
    from pragma.aml_data import AMLDecisionDataset, collate_decision
    from pragma.metrics import all_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    T = t_days * 24 * 3600.0

    # ---- load graph features, index by 5-tuple join key ----
    def gkey(r):
        return (r["account"], r["timestamp"], r["feature"],
                r["counterparty_account"], float(r["amount"]))

    print("[graph] loading graph features")
    gf = {}
    from pragma.tokenizer import load_records as _load_any  # handles JSONL + array
    for r in _load_any(f"{data_dir}/{graph_name}"):
        gf[gkey(r)] = r
    print(f"[graph] {len(gf)} selected events")

    # ---- reproduce split + tokenizer ----
    records = load_records(f"{data_dir}/{records_name}")
    train_records, test_records = stratified_split(records, train_frac=0.8, seed=seed)
    print(f"[split] {split_summary(train_records, test_records)}")
    test_accts = {r["account"] for r in test_records}

    profiles = load_profiles(f"{data_dir}/{profiles_name}") if use_profile else {}
    fit_recs = train_records[:200000]
    fit_profs = [profiles[r["account"]] for r in fit_recs if r["account"] in profiles]
    tok = fit_tokenizer(fit_recs, profiles=fit_profs)
    vocab = tok.vocab_config()

    # ---- load frozen backbone ----
    cfg = PRAGMAConfig.from_name(size, use_profile=use_profile)
    backbone = PRAGMA(vocab, cfg)
    ckpt_path = f"{VOL_PATH}/checkpoints/{ckpt}"
    state = {"step": "random-init"}
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        bb = {k[len("backbone."):]: v for k, v in state["model"].items() if k.startswith("backbone.")}
        backbone.load_state_dict(bb, strict=False)
        print(f"[ckpt] loaded backbone {ckpt} step {state.get('step','?')}")
    else:
        print(f"[ckpt] {ckpt_path} not found -> random-init backbone (plumbing test only)")
    backbone.to(device).eval()

    # ---- build decision points for every graph-selected event ----
    # index records by account for prefix construction
    def ekey(acct, ev):
        return (acct, ev["timestamp"], ev["feature"], ev["counterparty_account"], float(ev["amount"]))

    # points: (record_index, cut_index, label); parallel meta lists for graph+key+split
    all_recs = train_records + test_records
    is_test = [False] * len(train_records) + [True] * len(test_records)
    points, graph_rows, is_test_pt = [], [], []
    matched = 0
    for ridx, rec in enumerate(all_recs):
        acct = rec["account"]
        for i, ev in enumerate(rec["events"]):
            k = ekey(acct, ev)
            g = gf.get(k)
            if g is None:
                continue
            matched += 1
            e = float(ev.get("elapsed_since_mark", -1))
            label = 1 if (0.0 <= e < T) else 0
            points.append((ridx, i, label))
            graph_rows.append(g)
            is_test_pt.append(is_test[ridx])
    print(f"[join] matched {matched} of {len(gf)} graph events to prefixes")

    ds = AMLDecisionDataset(all_recs, tok, points, profiles=profiles)
    dl = DataLoader(ds, batch_size=embed_batch, shuffle=False, collate_fn=collate_decision,
                    num_workers=8, persistent_workers=True)

    # ---- embed all decision points with the frozen backbone ----
    print("[embed] computing PRAGMA embeddings for selected events")
    embs = []
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            out = backbone(batch)
            idx = batch.decision_index
            nb = idx.shape[0]
            rep = out.z_h_evt[torch.arange(nb, device=device), idx]
            embs.append(rep.cpu().numpy())
            done += nb
            if done % (embed_batch * 200) < embed_batch:
                print(f"[embed] {done}/{len(points)} ({(time.time()-t0)/max(done,1)*1000:.1f} ms/ev)")
    E = np.concatenate(embs).astype(np.float32)  # (N, d)
    d = E.shape[1]
    print(f"[embed] done {E.shape}")

    # ---- graph feature matrix (log1p heavy-tailed cols) ----
    G = np.zeros((len(graph_rows), len(GRAPH_COLS)), dtype=np.float32)
    for j, col in enumerate(GRAPH_COLS):
        vals = np.array([float(gr.get(col, 0.0) or 0.0) for gr in graph_rows], dtype=np.float64)
        if col in LOG1P_COLS:
            vals = np.log1p(np.clip(vals, 0, None))
        G[:, j] = vals
    y = np.array([lab for _, _, lab in points], dtype=np.float32)
    test_mask = np.array(is_test_pt, dtype=bool)
    print(f"[data] N={len(y)} pos={int(y.sum())} test={int(test_mask.sum())}")

    # ---- standard-scale features using TRAIN stats only ----
    def fit_scale(X, tr):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-6
        return (X - mu) / sd
    tr = ~test_mask
    Es = fit_scale(E, tr)
    Gs = fit_scale(G, tr)

    Xp = Es                                   # PRAGMA-only
    Xg = np.concatenate([Es, Gs], axis=1)     # PRAGMA + graph

    # ---- train a linear head on cached features ----
    def train_head(X, tag):
        Xt = torch.tensor(X, device=device)
        yt = torch.tensor(y, device=device)
        trm = torch.tensor(tr, device=device)
        tem = torch.tensor(test_mask, device=device)
        clf = torch.nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=head_lr, weight_decay=1e-4)
        pos_w = torch.tensor([w], device=device)
        Xtr, ytr = Xt[trm], yt[trm]
        for ep in range(head_epochs):
            clf.train()
            perm = torch.randperm(Xtr.shape[0], device=device)
            bs = 8192
            for b in range(0, Xtr.shape[0], bs):
                sl = perm[b:b + bs]
                logit = clf(Xtr[sl]).squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, ytr[sl], pos_weight=pos_w)
                opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            scores = torch.sigmoid(clf(Xt[tem]).squeeze(-1)).cpu().numpy()
        labels = y[test_mask]
        m = all_metrics(scores, labels)
        print(f"[{tag}] F2={m['f2']:.4f} F1={m['f1']:.4f} F0.5={m['f0.5']:.4f} "
              f"PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f}")
        return m, scores, labels

    m_p, sc_p, _ = train_head(Xp, "pragma-only")
    m_g, sc_g, _ = train_head(Xg, "pragma+graph")

    # ---- write results + per-event scores (test set) ----
    import csv
    res_dir = f"{VOL_PATH}/results"
    os.makedirs(res_dir, exist_ok=True)

    def write_summary(path, title, m):
        with open(path, "w") as fs:
            fs.write(title + "\n")
            fs.write(f"backbone_ckpt={ckpt} step={state.get('step','?')} T_days={t_days} "
                     f"W={w} head_epochs={head_epochs} seed={seed}\n")
            fs.write(f"selected_events={len(y)} pos={int(y.sum())} "
                     f"test_events={int(test_mask.sum())} test_pos={int(y[test_mask].sum())}\n\n")
            for k in ["f2", "f1", "f0.5", "pr_auc", "roc_auc",
                      "f2_threshold", "f1_threshold", "f0.5_threshold",
                      "n", "n_pos", "pos_rate"]:
                fs.write(f"{k}: {m[k]}\n")

    write_summary(f"{res_dir}/baseline_results.txt",
                  "PRAGMA-only AML (embedding probe on graph-selected events)", m_p)
    write_summary(f"{res_dir}/graph_results.txt",
                  "PRAGMA + graph-topology features AML (embedding probe)", m_g)

    # per-event scores for step-3 clustering (test set): key + both models' scores
    test_idx = np.where(test_mask)[0]
    with open(f"{res_dir}/per_event_scores.csv", "w", newline="") as fcsv:
        wc = csv.writer(fcsv)
        wc.writerow(["account", "timestamp", "feature", "counterparty_account", "amount",
                     "label", "elapsed_since_mark", "stratum", "score_pragma", "score_graph"])
        for j, gi in enumerate(test_idx):
            gr = graph_rows[gi]
            wc.writerow([gr["account"], gr["timestamp"], gr["feature"],
                         gr["counterparty_account"], gr["amount"],
                         int(y[gi]), gr.get("elapsed_since_mark", -1), gr.get("stratum"),
                         float(sc_p[j]), float(sc_g[j])])
    vol.commit()
    print(f"[done] wrote baseline_results.txt, graph_results.txt, per_event_scores.csv "
          f"({len(test_idx)} test rows)")

    def pyfloat(m):
        return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                for k, v in m.items()}
    return {"pragma_only": pyfloat(m_p), "pragma_graph": pyfloat(m_g)}


@app.local_entrypoint()
def main(ckpt: str = "pragma_S_latest.pt", head_epochs: int = 30, t_days: float = 30.0,
         w: float = 10.0, sample: bool = False):
    if sample:
        # fast end-to-end plumbing validation on the 50-account sample files
        res = run.remote(
            data_dir="/vol_samples",
            records_name="preprocessed.sample.json",
            profiles_name="profiles.sample.json",
            graph_name="graph_features.sample.json",
            ckpt="__none__", head_epochs=head_epochs, t_days=t_days, w=w,
        )
    else:
        res = run.remote(ckpt=ckpt, head_epochs=head_epochs, t_days=t_days, w=w)
    print("[result]", res)
