"""Configuration for PRAGMA (Revolut foundation model) architecture.

Two configs:
  - VocabConfig: the shared key/value token vocabulary and reserved special ids.
  - PRAGMAConfig: model dimensions for the S / M / L family (paper Table 1).

The vocabulary is a *single shared table* E that maps both semantic-type
tokens (keys, ~60 in the paper) and value tokens (~28k in the paper) to
d-dimensional vectors, plus a handful of reserved special tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- Reserved special token ids (must occupy the first slots of the table) ---
PAD_ID = 0   # padding for ragged token / event dimensions (padding_idx, no grad)
MASK_ID = 1  # replaces a value token that the model must reconstruct (in loss)
UNK_ID = 2   # corruption token; excluded from the MLM loss (input dropout, §2.3.5)
USR_ID = 3   # learnable [USR] token prepended to the profile-state sequence
EVT_ID = 4   # learnable [EVT] token prepended to every event sequence
NUM_SPECIAL = 5


@dataclass
class VocabConfig:
    """Sizes of the shared embedding table.

    key ids occupy      [NUM_SPECIAL,            NUM_SPECIAL + n_keys)
    value ids occupy    [NUM_SPECIAL + n_keys,   NUM_SPECIAL + n_keys + n_values)

    The event builder is responsible for assigning ids in this layout; the model
    only needs `vocab_size` and the special ids above. `key_id_lo/hi` and
    `value_id_lo/hi` are provided so the loss can optionally be restricted to the
    value sub-vocabulary if desired.
    """

    n_keys: int = 64          # semantic types (paper: ~60)
    n_values: int = 28_000    # numerical buckets + categorical + BPE subwords (paper: ~28k)
    max_field_pos: int = 32   # max #value-tokens within a single field (for within-field PosEmb)

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL + self.n_keys + self.n_values

    @property
    def key_id_lo(self) -> int:
        return NUM_SPECIAL

    @property
    def key_id_hi(self) -> int:
        return NUM_SPECIAL + self.n_keys

    @property
    def value_id_lo(self) -> int:
        return NUM_SPECIAL + self.n_keys

    @property
    def value_id_hi(self) -> int:
        return NUM_SPECIAL + self.n_keys + self.n_values


@dataclass
class PRAGMAConfig:
    """Model dimensions. Defaults are PRAGMA-S (10M). See `from_name`."""

    d_model: int = 192
    d_ffn: int = 768
    n_heads: int = 3
    profile_depth: int = 1
    event_depth: int = 5
    history_depth: int = 2

    dropout: float = 0.1
    label_smoothing: float = 0.1
    rope_theta: float = 10_000.0

    use_profile: bool = True  # if False, run event-only (paper's event-only ablation)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        hd = self.d_model // self.n_heads
        assert hd % 2 == 0, "head_dim must be even for RoPE"
        return hd

    @classmethod
    def from_name(cls, name: str, **overrides) -> "PRAGMAConfig":
        name = name.strip().upper().replace("PRAGMA-", "")
        table = {
            # d_model, d_ffn, heads, profile, event, history   (paper Table 1)
            "S": dict(d_model=192, d_ffn=768, n_heads=3, profile_depth=1, event_depth=5, history_depth=2),
            "M": dict(d_model=512, d_ffn=2048, n_heads=8, profile_depth=3, event_depth=16, history_depth=6),
            "L": dict(d_model=1024, d_ffn=4096, n_heads=16, profile_depth=9, event_depth=45, history_depth=18),
        }
        if name not in table:
            raise ValueError(f"unknown PRAGMA size {name!r}; choose from {list(table)}")
        cfg = {**table[name], **overrides}
        return cls(**cfg)


@dataclass
class MaskingConfig:
    """Masking strategy (§2.3.5). Probabilities are per the paper."""

    token_prob: float = 0.15   # individual value-token masking
    event_prob: float = 0.10   # whole-event masking
    key_prob: float = 0.10     # mask all values of a randomly-selected key
    unk_fraction: float = 0.10  # of selected positions, fraction sent to [UNK] (no loss)
