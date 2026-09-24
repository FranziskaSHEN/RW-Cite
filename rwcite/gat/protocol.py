"""EWM graph-attention data-contract constants and artifact paths.

The constants identify the frozen comparison manifest and the score-and-rank
fusion artifacts used to reproduce the reported EWM result.
"""

from __future__ import annotations

from pathlib import Path

from rwcite.gat import NOFUTURE_GEXF_SHA256, SPLIT_ID

DOMAIN = "ewm"
WD = "embodied_world_model_retrieval"

FULL_O2_GEXF = f"{WD}/description/test_graph_rr.o2.gexf"
FULL_O2_SHA256 = (
    "38424660264ab21006da2f76d370b7bb3950a7cce28017d012f364c801d0ce33"
)
NOFUTURE_GEXF = f"{WD}/description/test_graph_rr.o2.nofuture.gexf"
SPLIT_DIR = f"{WD}/data/splits"
TRAIN_JSONL = f"{WD}/data/reference_recommend/train.jsonl"
TEST_JSONL = f"{WD}/data/reference_recommend/test.jsonl"
BASELINE_FULL381_MERGED = (
    f"{WD}/ranker/eval/ewm_v5_scibert_c2s_nofuture_test381_merged.json"
)
# Frozen EWM score-and-rank fusion result (full-gold Hits@10).
L0_DEFAULT_HITS_AT_10 = 4.000
L0_DEFAULT_MERGED = f"{WD}/ranker/eval/ewm_gat_l0_default_test381_merged.json"
L0_PRIOR_ZBLEND_HITS_AT_10 = 3.924

GAT_MVP_EVAL_TAG = "ewm_gat_mvp_test381"
GAT_MVP_MERGED = f"{WD}/ranker/eval/{GAT_MVP_EVAL_TAG}_merged.json"
GAT_MVP_CKPT_DIR = f"{WD}/ranker/gat_mvp"

# Deprecated experimental path names retained only for artifact compatibility.
GAT_FULLU_CKPT_DIR = f"{WD}/ranker/gat_fullU"  # Invalid for reported evaluation.
GAT_FULLU_EVAL_TAG = "ewm_gat_fullU_test381"
GAT_U_CKPT_DIR = f"{WD}/ranker/gat_U"  # fair u_only — closed fail
GAT_U_EVAL_TAG = "ewm_gat_U_test381"

POOL_MODE_STRUCT = "struct"
POOL_MODE_FULL_U = "full_u"  # deprecated / unfair (inject)
POOL_MODE_U_ONLY = "u_only"  # deprecated research (fair Top-K U)
POOL_MODE_U_FULL = "u_full"  # deprecated

# MVP / 3.0 shortlist width.
STRUCT_WINDOW = 400
GAT_U_DEFAULT_POOL_CAP = 3072  # historical K*; unused in production
GAT_CHUNK = 512

N_TRAIN_SOURCES = 3423
N_TEST_SOURCES = 381
N_FRONTIER_SOURCES = 196


def root_path(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    import os

    env = os.environ.get("RWCITE_ROOT")
    if env:
        return Path(env)
    # rwcite/gat/protocol.py → repo root
    return Path(__file__).resolve().parents[2]
