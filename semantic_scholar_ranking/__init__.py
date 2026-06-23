"""ranking — MRR-calibrated rank fusion for CFE Hackathon.

Quick start
-----------
from ranking import load_dataset, tune_rrf, evaluate, S2Client

dataset, _ = load_dataset("../Data")
cfg        = tune_rrf(dataset.dev, sources=["agent_a", "agent_b"])
results    = evaluate(dataset.test, cfg.rank_fn())
print(results)
"""

from .schema      import Query, Dataset
from .metrics     import paper_mrr, corpus_mrr, recall_at_k, ndcg_at_k, evaluate
from .fusion      import (weighted_rrf, borda_count, combsum,
                           feature_bias_term,
                           make_rrf_rank_fn, make_single_source_rank_fn)
from .optimize    import tune_rrf, RRFConfig
from .s2_client   import S2Client, GraphSignalExtractor, GraphContext
from .s2_agent    import S2DirectionAgent
from .data_loader import load_dataset

__all__ = [
    "Query", "Dataset",
    "paper_mrr", "corpus_mrr", "recall_at_k", "ndcg_at_k", "evaluate",
    "weighted_rrf", "borda_count", "combsum",
    "feature_bias_term", "make_rrf_rank_fn", "make_single_source_rank_fn",
    "tune_rrf", "RRFConfig",
    "S2Client", "GraphSignalExtractor", "GraphContext",
    "S2DirectionAgent",
    "load_dataset",
]
