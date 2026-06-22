"""Rank-fusion + MRR harness for ResearchHorizon (Agent 3).

Public API:
    from rank_fusion import (
        Query, Dataset,
        evaluate, corpus_mrr, paper_mrr,
        weighted_rrf, make_rrf_rank_fn, make_single_source_rank_fn,
        tune_rrf, RRFConfig,
        MetaRanker, fit_select_meta_ranker,
        make_synthetic_dataset, load_real_dataset,
    )
"""

from .schema import Query, Dataset
from .metrics import evaluate, corpus_mrr, paper_mrr, recall_at_k, ndcg_at_k
from .fusion import (
    weighted_rrf, borda_count, combsum,
    make_rrf_rank_fn, make_single_source_rank_fn,
)
from .optimize import tune_rrf, RRFConfig
from .meta_ranker import MetaRanker, fit_select_meta_ranker
from .learned_rankers import (
    PairwiseRankNet, ListNetRanker, SoftMRRRanker,
    fit_select_ranknet, fit_select_listnet, fit_select_soft_mrr,
)
from .diagnostics import (
    oracle_mrr, candidate_recall, candidate_recall_table, bootstrap_mrr_ci,
)
from .explain import (
    DirectionCard, build_direction_card, explain_ranking, render_card_markdown,
    rrf_attribution, linear_shap, ablation_probe, reason_logically,
    faithfulness_check, ground_evidence, Calibrator,
)
from .synthetic import (
    make_synthetic_dataset, make_candidate_generation_dataset,
    DEFAULT_SKILLS, DEFAULT_GENERATORS,
)
from .realdata import load_real_dataset

__all__ = [
    "load_real_dataset",
    "PairwiseRankNet", "ListNetRanker", "SoftMRRRanker",
    "fit_select_ranknet", "fit_select_listnet", "fit_select_soft_mrr",
    "oracle_mrr", "candidate_recall", "candidate_recall_table", "bootstrap_mrr_ci",
    "make_candidate_generation_dataset", "DEFAULT_GENERATORS",
    "DirectionCard", "build_direction_card", "explain_ranking", "render_card_markdown",
    "rrf_attribution", "linear_shap", "ablation_probe", "reason_logically",
    "faithfulness_check", "ground_evidence", "Calibrator",
    "Query", "Dataset",
    "evaluate", "corpus_mrr", "paper_mrr", "recall_at_k", "ndcg_at_k",
    "weighted_rrf", "borda_count", "combsum",
    "make_rrf_rank_fn", "make_single_source_rank_fn",
    "tune_rrf", "RRFConfig",
    "MetaRanker", "fit_select_meta_ranker",
    "make_synthetic_dataset", "DEFAULT_SKILLS",
]
