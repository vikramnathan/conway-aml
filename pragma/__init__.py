"""PRAGMA: encoder-only foundation model for banking event sequences.

Implements the architecture and MLM pretraining objective from
"PRAGMA: Revolut Foundation Model" (arXiv:2604.08649).
"""

from .config import PRAGMAConfig, VocabConfig, MaskingConfig
from .batch import PragmaBatch, soft_log_seconds
from .model import PRAGMA, PragmaOutput
from .mlm import PRAGMAForMLM, apply_masking, IGNORE_INDEX
from .aml import PRAGMAForAML, aml_targets
from .split import stratified_split, is_guilty, split_summary
from .metrics import all_metrics, roc_auc, pr_auc, best_fbeta
from .clustering import build_groups, match_penalty, jaccard, guilty_events_from_rows, Group

__all__ = [
    "PRAGMAConfig", "VocabConfig", "MaskingConfig",
    "PragmaBatch", "soft_log_seconds",
    "PRAGMA", "PragmaOutput",
    "PRAGMAForMLM", "apply_masking", "IGNORE_INDEX",
    "PRAGMAForAML", "aml_targets",
    "stratified_split", "is_guilty", "split_summary",
    "all_metrics", "roc_auc", "pr_auc", "best_fbeta",
    "build_groups", "match_penalty", "jaccard", "guilty_events_from_rows", "Group",
]
