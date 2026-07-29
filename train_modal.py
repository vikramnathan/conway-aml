"""Modal app: MLM pre-training of PRAGMA on the preprocessed event corpus.

Usage:
    modal run train_modal.py            # smoke run on the sample file
    modal run train_modal.py --records-path /data/preprocessed.json --size S --steps 5000

The preprocessed JSON is uploaded to a Modal Volume once (see `upload_data`),
then training streams it from the Volume. Checkpoints are written back to the
Volume under /vol/checkpoints.

This trains the architecture in pragma/ with the masking objective from §2.3.5.
"""

from __future__ import annotations

import modal

app = modal.App("pragma-pretrain")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy<2.0")
    .add_local_python_source("pragma")
)

vol = modal.Volume.from_name("pragma-data", create_if_missing=True)
VOL_PATH = "/vol"


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=60 * 60)
def upload_data(local_bytes: bytes, name: str = "preprocessed.json"):
    """Persist an uploaded JSON blob into the Volume."""
    import os
    os.makedirs(VOL_PATH, exist_ok=True)
    dest = f"{VOL_PATH}/{name}"
    with open(dest, "wb") as f:
        f.write(local_bytes)
    vol.commit()
    return dest


@app.function(image=image, gpu="A100-40GB", volumes={VOL_PATH: vol},
              memory=96 * 1024, timeout=12 * 60 * 60)
def train(
    records_name: str = "preprocessed.json",
    profiles_name: str | None = "profiles.json",
    size: str = "S",
    steps: int = 2000,
    batch_size: int = 16,
    lr: float = 3e-4,
    fit_limit: int = 20000,
    seed: int = 0,
    use_profile: bool = True,
    ckpt_every: int = 100,
):
    import json
    import time
    import torch
    from torch.utils.data import DataLoader

    from pragma.config import PRAGMAConfig, MaskingConfig
    from pragma.tokenizer import fit_tokenizer, load_records, load_profiles
    from pragma.data import PragmaDataset, collate
    from pragma.mlm import PRAGMAForMLM, apply_masking
    from pragma.split import stratified_split, split_summary

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = f"{VOL_PATH}/{records_name}"
    print(f"[data] loading {path}")
    records = load_records(path)

    # Stratified 80/20 train/test split by account (guilty vs innocent strata).
    train_records, test_records = stratified_split(records, train_frac=0.8, seed=seed)
    print(f"[split] {split_summary(train_records, test_records)}")
    test_accounts = [r["account"] for r in test_records]

    profiles = {}
    if use_profile and profiles_name:
        profiles = load_profiles(f"{VOL_PATH}/{profiles_name}")
        print(f"[data] {len(profiles)} profiles loaded")

    # Fit tokenizer on TRAIN ONLY (no test leakage into vocab/bucket edges).
    fit_recs = train_records[:fit_limit]
    fit_profs = [profiles[r["account"]] for r in fit_recs if r["account"] in profiles]
    print(f"[data] {len(train_records)} train records; fitting tokenizer on <= {fit_limit}")
    tok = fit_tokenizer(fit_recs, profiles=fit_profs)
    vocab = tok.vocab_config()
    print(f"[vocab] n_keys={vocab.n_keys} n_values={vocab.n_values} vocab_size={vocab.vocab_size}")

    cfg = PRAGMAConfig.from_name(size, use_profile=use_profile)
    mcfg = MaskingConfig()
    model = PRAGMAForMLM(vocab, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] PRAGMA-{size} params={n_params/1e6:.1f}M device={device}")

    ds = PragmaDataset(train_records, tok, profiles=profiles)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                    drop_last=True, num_workers=8, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    import os
    os.makedirs(f"{VOL_PATH}/checkpoints", exist_ok=True)
    latest = f"{VOL_PATH}/checkpoints/pragma_{size}_latest.pt"

    def save(step):
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
             "cfg": cfg, "vocab": vocab,
             "test_accounts": test_accounts, "split_seed": seed},
            latest,
        )
        vol.commit()

    gen = torch.Generator(device=device).manual_seed(seed)
    model.train()
    step = 0
    t0 = time.time()
    while step < steps:
        for batch in dl:
            batch = batch.to(device)
            batch = apply_masking(batch, vocab, mcfg, generator=gen)
            out = model(batch)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                acc = out.get("acc")
                accs = f" acc={acc:.3f}" if acc is not None else ""
                print(f"[step {step:5d}] loss={loss.item():.4f}{accs} "
                      f"masked={out['n_masked']} ({(time.time()-t0)/step*1000:.0f} ms/step)")
            # checkpoint every 100 steps, overwriting the single 'latest' file
            if step % ckpt_every == 0:
                save(step)
                print(f"[ckpt] step {step} -> {latest}")
            if step >= steps:
                break

    save(step)
    print(f"[done] saved {latest} (held-out test accounts: {len(test_accounts)})")
    return {"final_loss": loss.item(), "steps": step, "params_M": n_params / 1e6,
            "n_train": len(train_records), "n_test": len(test_records)}


@app.local_entrypoint()
def main(
    records_name: str = "preprocessed.json",
    profiles_name: str = "profiles.json",
    size: str = "S",
    steps: int = 20000,
    batch_size: int = 32,
    fit_limit: int = 200000,
    ckpt_every: int = 100,
):
    # Data is already on the Volume (via `modal volume put`); no upload here.
    result = train.remote(
        records_name=records_name, profiles_name=profiles_name, size=size, steps=steps,
        batch_size=batch_size, fit_limit=fit_limit, ckpt_every=ckpt_every,
    )
    print("[result]", result)
