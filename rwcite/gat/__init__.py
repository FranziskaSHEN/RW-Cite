"""Graph-attention ranking over the split-masked domain graph.

The public evaluation contract admits queries on the complete extraction graph,
trains and scores on the split-masked graph, and evaluates the structural
Top-400 window. Random edge splits are not used for reported metrics.
"""

__all__ = [
    "BASELINE_FULL381_HITS_AT_10",
    "NOFUTURE_GEXF_SHA256",
    "SPLIT_ID",
]

# Frozen EWM split-masked comparison target.
BASELINE_FULL381_HITS_AT_10 = 2.735
NOFUTURE_GEXF_SHA256 = (
    "ba5daf92f2a41c98e76f05c0a659b31fe9115618cac4636d3d2f016cb169c335"
)
SPLIT_ID = "test_admit_v1"
