"""End-to-end smoke test for the PRAGMA architecture + MLM objective.

Runs on CPU. Verifies:
  1. tokenizer fits on the real sample and produces the expected id layout;
  2. collate builds correctly-shaped, mask-consistent PragmaBatch tensors;
  3. masking populates labels only at real value-token positions;
  4. a forward pass produces finite loss and correct logit shape;
  5. a few optimisation steps on a tiny model reduce the loss (arch can learn).
"""

import json

import torch

from pragma.config import PRAGMAConfig, MaskingConfig, NUM_SPECIAL
from pragma.tokenizer import fit_tokenizer, load_records
from pragma.data import PragmaDataset, collate
from pragma.mlm import PRAGMAForMLM, apply_masking, IGNORE_INDEX
from pragma.aml import PRAGMAForAML, aml_targets


def _load():
    records = load_records("preprocessed.sample.json")
    profiles = {p["account"]: p for p in json.load(open("profiles.sample.json"))}
    return records, profiles


def test_sample_end_to_end():
    records, profiles = _load()
    assert len(records) > 0
    tok = fit_tokenizer(records, profiles=list(profiles.values()))
    vocab = tok.vocab_config()

    # id layout sanity
    assert vocab.key_id_lo == NUM_SPECIAL
    assert vocab.value_id_lo == NUM_SPECIAL + vocab.n_keys
    assert vocab.vocab_size == NUM_SPECIAL + vocab.n_keys + vocab.n_values

    ds = PragmaDataset(records, tok, profiles=profiles)
    batch = collate([ds[i] for i in range(min(8, len(ds)))])

    # profile branch populated
    assert batch.profile_key_ids is not None
    assert batch.profile_token_mask.any()

    # shape + mask consistency
    B, E, Te = batch.event_key_ids.shape
    assert batch.event_token_mask.shape == (B, E, Te)
    # every real token has a key id in the key range and value id in value range
    real = batch.event_token_mask
    assert (batch.event_key_ids[real] >= vocab.key_id_lo).all()
    assert (batch.event_key_ids[real] < vocab.key_id_hi).all()
    assert (batch.event_val_ids[real] >= vocab.value_id_lo).all()
    assert (batch.event_val_ids[real] < vocab.value_id_hi).all()
    # padded positions are PAD_ID (0)
    assert (batch.event_key_ids[~real] == 0).all()

    # masking
    mcfg = MaskingConfig()
    gen = torch.Generator().manual_seed(0)
    masked = apply_masking(batch, vocab, mcfg, generator=gen)
    labels = masked.mlm_labels
    supervised = labels != IGNORE_INDEX
    assert supervised.sum() > 0
    # supervised positions must be real tokens
    assert real[supervised].all()

    # forward (MLM, with profile branch on)
    cfg = PRAGMAConfig.from_name("S", use_profile=True)
    model = PRAGMAForMLM(vocab, cfg)
    out = model(masked)
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape[1] == vocab.vocab_size
    print(f"forward ok: loss={out['loss'].item():.3f} masked={out['n_masked']}")


def test_aml_targets():
    T, W = 100.0, 10.0
    e = torch.tensor([-1.0, 0.0, 50.0, 99.0, 100.0, 150.0])
    y, w, in_loss = aml_targets(e, T, W)
    #                e<0    e=0    mid    <T     =T (reset) >T
    assert torch.allclose(y,       torch.tensor([0., 1., 1., 1., 0., 0.]))
    assert torch.allclose(w,       torch.tensor([1., 10., 10., 10., 0., 0.]))
    assert torch.equal(in_loss, torch.tensor([True, True, True, True, False, False]))
    print("aml targets ok")


def test_split():
    from pragma.split import stratified_split, is_guilty, split_summary
    records, _ = _load()
    train, test = stratified_split(records, train_frac=0.8, seed=0)
    print("split:", split_summary(train, test))
    # no account leaks across partitions
    tr_ac = {r["account"] for r in train}
    te_ac = {r["account"] for r in test}
    assert tr_ac.isdisjoint(te_ac)
    assert len(train) + len(test) == len(records)
    # deterministic
    train2, _ = stratified_split(records, train_frac=0.8, seed=0)
    assert [r["account"] for r in train] == [r["account"] for r in train2]
    # guilty accounts present in both strata splits when there are enough
    g_total = sum(is_guilty(r) for r in records)
    if g_total >= 5:
        assert any(is_guilty(r) for r in test), "guilty stratum should reach test set"
    print("split ok")


def test_aml_head_forward():
    records, profiles = _load()
    tok = fit_tokenizer(records, profiles=list(profiles.values()))
    vocab = tok.vocab_config()
    ds = PragmaDataset(records, tok, profiles=profiles)
    batch = collate([ds[i] for i in range(min(8, len(ds)))])

    cfg = PRAGMAConfig.from_name("S", use_profile=True)
    from pragma.model import PRAGMA
    backbone = PRAGMA(vocab, cfg)
    model = PRAGMAForAML(backbone, cfg, T=30 * 24 * 3600.0, W=10.0)
    out = model(batch)
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == batch.event_mask.shape
    out["loss"].backward()
    print(f"aml forward ok: loss={out['loss'].item():.4f} "
          f"n_pos={out['n_pos']} n_in_loss={out['n_in_loss']} n_valid={out['n_valid']}")


def test_metrics():
    import numpy as np
    from pragma.metrics import roc_auc, pr_auc, best_fbeta, all_metrics
    # perfect separation -> AUC 1.0
    s = np.array([0.9, 0.8, 0.2, 0.1]); y = np.array([1, 1, 0, 0])
    assert abs(roc_auc(s, y) - 1.0) < 1e-9
    assert abs(pr_auc(s, y) - 1.0) < 1e-9
    f1, _ = best_fbeta(s, y, 1.0)
    assert abs(f1 - 1.0) < 1e-9
    # reversed -> AUC 0.0
    assert abs(roc_auc(s, 1 - y) - 0.0) < 1e-9
    # ties handled: all equal scores -> AUC 0.5
    assert abs(roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), y) - 0.5) < 1e-9
    m = all_metrics(s, y)
    assert set(m) >= {"roc_auc", "pr_auc", "f1", "f0.5"}
    print("metrics ok")


def test_aml_decision_points_and_train():
    """Decision-point construction + prefix-based AML head trains, no leakage."""
    from pragma.model import PRAGMA
    from pragma.aml_data import build_decision_points, AMLDecisionDataset, collate_decision
    records, profiles = _load()
    tok = fit_tokenizer(records, profiles=list(profiles.values()))
    vocab = tok.vocab_config()
    T = 30 * 24 * 3600.0

    pts = build_decision_points(records, T_seconds=T, neg_frac=1.0, seed=0)  # keep all for test
    n_pos = sum(l for _, _, l in pts)
    assert n_pos > 0, "expected positive decision points"
    # label correctness: positive iff 0<=e<T at the cut event
    for ridx, cut, label in pts[:200]:
        e = float(records[ridx]["events"][cut].get("elapsed_since_mark", -1))
        assert label == (1 if 0.0 <= e < T else 0)

    ds = AMLDecisionDataset(records, tok, pts, profiles=profiles)
    # build a batch mixing positives and negatives
    pos_i = [k for k, p in enumerate(pts) if p[2] == 1][:6]
    neg_i = [k for k, p in enumerate(pts) if p[2] == 0][:6]
    batch = collate_decision([ds[k] for k in (pos_i + neg_i)])
    assert batch.decision_index is not None
    assert batch.aml_label.shape[0] == len(pos_i) + len(neg_i)

    cfg = PRAGMAConfig.from_name("S", use_profile=True)
    model = PRAGMAForAML(PRAGMA(vocab, cfg), cfg, T=T, W=10.0)
    out0 = model(batch)
    assert out0["logits"].shape == batch.aml_label.shape  # one logit per record
    assert out0["n_pos"] == len(pos_i)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for i in range(30):
        out = model(batch)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        if i == 0:
            first = out["loss"].item()
        last = out["loss"].item()
    print(f"aml decision trains: first={first:.4f} last={last:.4f} "
          f"n_pts={len(pts)} n_pos={n_pos}")
    assert last < first


def test_learns_on_batch():
    """Overfit a single batch: loss should drop clearly."""
    records, profiles = _load()
    tok = fit_tokenizer(records, profiles=list(profiles.values()))
    vocab = tok.vocab_config()
    ds = PragmaDataset(records, tok, profiles=profiles)
    batch = collate([ds[i] for i in range(min(8, len(ds)))])

    cfg = PRAGMAConfig.from_name("S", use_profile=True)
    model = PRAGMAForMLM(vocab, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    first = last = None
    for i in range(40):
        m = apply_masking(batch, vocab, MaskingConfig(), generator=gen)
        out = model(m)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if i == 0:
            first = out["loss"].item()
        last = out["loss"].item()
    print(f"learns: first={first:.3f} last={last:.3f}")
    assert last < first, "loss did not decrease"


if __name__ == "__main__":
    test_sample_end_to_end()
    test_aml_targets()
    test_aml_head_forward()
    test_split()
    test_metrics()
    test_aml_decision_points_and_train()
    test_learns_on_batch()
    print("ALL SMOKE TESTS PASSED")
