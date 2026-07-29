"""Modal app: downstream AML fine-tuning + eval on the pretrained PRAGMA backbone.

    modal run aml_modal.py                       # embedding-probe on latest S ckpt
    modal run aml_modal.py --mode full --epochs 3
    modal run aml_modal.py --ckpt pragma_S_latest.pt --w 10 --t-days 30

Reproduces the SAME deterministic 80/20 stratified split and train-only tokenizer
used in pretraining (seed-derived, so no dependency on the pretraining job and no
leakage), loads the pretrained backbone, attaches the per-event AML head, trains,
then evaluates on the held-out test accounts. The lag-penalty loss weights the
laundering window (0<=e<T) by W; F0.5 is the headline metric (matching the paper).

Modes:
  probe : freeze the backbone, train only the linear classifier (§3.1.1).
  full  : fine-tune the whole model.
"""

from __future__ import annotations

import modal

app = modal.App("pragma-aml")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy<2.0")
    .add_local_python_source("pragma")
)

vol = modal.Volume.from_name("pragma-data", create_if_missing=True)
VOL_PATH = "/vol"


@app.function(image=image, gpu="A100-40GB", volumes={VOL_PATH: vol},
              memory=96 * 1024, timeout=12 * 60 * 60)
def finetune(
    records_name: str = "preprocessed.json",
    profiles_name: str = "profiles.json",
    ckpt: str = "pragma_S_latest.pt",
    mode: str = "probe",          # "probe" (freeze backbone) or "full"
    size: str = "S",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    fit_limit: int = 200000,
    seed: int = 0,
    t_days: float = 30.0,
    w: float = 10.0,
    neg_frac: float = 0.01,
    use_profile: bool = True,
):
    import os
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from pragma.config import PRAGMAConfig
    from pragma.tokenizer import fit_tokenizer, load_records, load_profiles
    from pragma.split import stratified_split, split_summary
    from pragma.model import PRAGMA
    from pragma.aml import PRAGMAForAML
    from pragma.aml_data import build_decision_points, AMLDecisionDataset, collate_decision
    from pragma.metrics import all_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    T = t_days * 24 * 3600.0

    # ---- reproduce split + tokenizer exactly as in pretraining ----
    records = load_records(f"{VOL_PATH}/{records_name}")
    train_records, test_records = stratified_split(records, train_frac=0.8, seed=seed)
    print(f"[split] {split_summary(train_records, test_records)}")

    profiles = load_profiles(f"{VOL_PATH}/{profiles_name}") if use_profile else {}

    fit_recs = train_records[:fit_limit]
    fit_profs = [profiles[r["account"]] for r in fit_recs if r["account"] in profiles]
    tok = fit_tokenizer(fit_recs, profiles=fit_profs)
    vocab = tok.vocab_config()
    print(f"[vocab] vocab_size={vocab.vocab_size}")

    # ---- build model, load pretrained backbone ----
    cfg = PRAGMAConfig.from_name(size, use_profile=use_profile)
    backbone = PRAGMA(vocab, cfg)
    state = torch.load(f"{VOL_PATH}/checkpoints/{ckpt}", map_location="cpu", weights_only=False)
    sd = state["model"]
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(bb, strict=False)
    print(f"[ckpt] loaded backbone from {ckpt} (step {state.get('step','?')}); "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    model = PRAGMAForAML(backbone, cfg, T=T, W=w).to(device)
    if mode == "probe":
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = list(model.classifier.parameters())
        print("[mode] probe: backbone frozen, training classifier only")
    else:
        params = list(model.parameters())
        print("[mode] full fine-tune")
    opt = torch.optim.Adam(params, lr=lr)

    # Decision points: all positives (0<=e<T), 1% of negatives per account.
    tr_pts = build_decision_points(train_records, T_seconds=T, neg_frac=neg_frac, seed=seed)
    te_pts = build_decision_points(test_records, T_seconds=T, neg_frac=neg_frac, seed=seed)
    n_pos_tr = sum(l for _, _, l in tr_pts)
    n_pos_te = sum(l for _, _, l in te_pts)
    print(f"[points] train={len(tr_pts)} (pos={n_pos_tr}) test={len(te_pts)} (pos={n_pos_te})")

    tr_ds = AMLDecisionDataset(train_records, tok, tr_pts, profiles=profiles)
    te_ds = AMLDecisionDataset(test_records, tok, te_pts, profiles=profiles)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_decision,
                       drop_last=True, num_workers=8, persistent_workers=True)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_decision,
                       num_workers=4)

    def evaluate(return_scores=False):
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for batch in te_dl:
                batch = batch.to(device)
                out = model(batch)
                scores.append(out["probs"].cpu().numpy())
                labels.append(out["labels"].cpu().numpy())
        model.train()
        s = np.concatenate(scores) if scores else np.array([])
        y = np.concatenate(labels) if labels else np.array([])
        m = all_metrics(s, y)
        return (m, s, y) if return_scores else m

    # ---- train ----
    model.train()
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        for batch in tr_dl:
            batch = batch.to(device)
            out = model(batch)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"[ep {ep} step {step}] loss={out['loss'].item():.4f} "
                      f"n_pos={out['n_pos']} ({(time.time()-t0)/step*1000:.0f} ms/step)")
        m = evaluate()
        print(f"[eval ep {ep}] F0.5={m['f0.5']:.4f} F1={m['f1']:.4f} "
              f"PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
              f"(n={m['n']} pos={m['n_pos']} rate={m['pos_rate']:.4g})")

    final, scores, labels = evaluate(return_scores=True)
    out_ckpt = f"{VOL_PATH}/checkpoints/aml_{size}_{mode}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "vocab": vocab,
                "metrics": final, "T": T, "W": w}, out_ckpt)

    # ---- persist per-event results (eval te_dl is shuffle=False -> aligns with te_pts) ----
    # Each test decision point -> (account, decision_timestamp, label, score).
    import csv
    res_dir = f"{VOL_PATH}/results"
    os.makedirs(res_dir, exist_ok=True)
    per_event_path = f"{res_dir}/baseline_per_event_{mode}.csv"
    n = min(len(scores), len(te_pts))
    with open(per_event_path, "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["account", "decision_timestamp", "cut_index", "label", "score"])
        for i in range(n):
            ridx, cut, lab = te_pts[i]
            rec = test_records[ridx]
            ts = rec["events"][cut]["timestamp"]
            wcsv.writerow([rec["account"], ts, cut, int(labels[i]), float(scores[i])])

    # ---- headline metrics summary ----
    summary_path = f"{res_dir}/baseline_results.txt"
    with open(summary_path, "w") as fs:
        fs.write("PRAGMA AML baseline (embedding probe, decision-point, no graph features)\n")
        fs.write(f"backbone_ckpt={ckpt} step={state.get('step','?')} mode={mode} "
                 f"epochs={epochs} T_days={t_days} W={w} neg_frac={neg_frac}\n")
        fs.write(f"train_points={len(tr_pts)} (pos={n_pos_tr}) "
                 f"test_points={len(te_pts)} (pos={n_pos_te})\n\n")
        for k in ["f0.5", "f1", "pr_auc", "roc_auc", "f0.5_threshold", "f1_threshold",
                  "n", "n_pos", "pos_rate"]:
            fs.write(f"{k}: {final[k]}\n")
    vol.commit()
    print(f"[done] saved {out_ckpt}")
    print(f"[results] {summary_path} + {per_event_path} ({n} rows)")
    print(f"[final] {final}")
    return final


@app.local_entrypoint()
def main(
    ckpt: str = "pragma_S_latest.pt",
    mode: str = "probe",
    epochs: int = 2,
    w: float = 10.0,
    t_days: float = 30.0,
    neg_frac: float = 0.01,
):
    res = finetune.remote(ckpt=ckpt, mode=mode, epochs=epochs, w=w,
                          t_days=t_days, neg_frac=neg_frac)
    print("[result]", res)
