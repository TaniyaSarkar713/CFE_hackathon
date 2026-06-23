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
    topk_bottomk_separation,
)
from .significance import (
    per_paper_mrr, paired_bootstrap_delta, paired_ttest, welch_ttest,
    bonferroni, compare_to_baseline,
)
from .matching import (
    match_relevance, similarity_matrix, calibrate_tau,
)
from .agent import (
    CandidateHarvestAgent, AgentConfig, AgentState, Action, Candidate,
    evaluate_pool, candidate_texts_from_paper, keywords_of,
)
from .integrate import (
    pool_to_query, graph_signal_fn, attach_features, add_horizon_source,
    horizon_prior, beta_from_horizon_weights, HORIZON_KEYS, GRAPH_SIGNAL_MAP,
    s2_agent_source, s2_agent_features,
)
from .explain import (
    DirectionCard, build_direction_card, explain_ranking, render_card_markdown,
    rrf_attribution, linear_shap, ablation_probe, reason_logically,
    faithfulness_check, ground_evidence, Calibrator,
)
from .synthetic import (
    make_synthetic_dataset, make_candidate_generation_dataset,
    make_redundancy_dataset, make_consensus_dataset,
    DEFAULT_SKILLS, DEFAULT_GENERATORS,
)
from .consensus import (
    ConsensusConfig, consensus_score, make_consensus_rank_fn, corpus_mrr_at_k,
    tune_consensus, add_consensus_features, cluster_sources,
    build_consensus_query, CONSENSUS_KEYS,
)
from .select import (
    compare_strategies, select_best_strategy, strategy_rank_fns,
)
from .entities import (
    keyphrases, entity_rerank, entity_rerank_texts, rerank_s2_source,
)
from .diversity import (
    mmr_rerank, make_mmr_rank_fn, feature_embedding_sim, cosine_sim_callable,
)
from .pairwise import swiss_tournament, comparator_from_scores
from .dedup import dedup_candidates, embed_minilm
from .realdata import load_real_dataset

__all__ = [
    "load_real_dataset",
    "PairwiseRankNet", "ListNetRanker", "SoftMRRRanker",
    "fit_select_ranknet", "fit_select_listnet", "fit_select_soft_mrr",
    "oracle_mrr", "candidate_recall", "candidate_recall_table", "bootstrap_mrr_ci",
    "topk_bottomk_separation",
    "per_paper_mrr", "paired_bootstrap_delta", "paired_ttest", "welch_ttest",
    "bonferroni", "compare_to_baseline",
    "match_relevance", "similarity_matrix", "calibrate_tau",
    "CandidateHarvestAgent", "AgentConfig", "AgentState", "Action", "Candidate",
    "evaluate_pool", "candidate_texts_from_paper", "keywords_of",
    "pool_to_query", "graph_signal_fn", "attach_features", "add_horizon_source",
    "horizon_prior", "beta_from_horizon_weights", "HORIZON_KEYS", "GRAPH_SIGNAL_MAP",
    "s2_agent_source", "s2_agent_features",
    "make_candidate_generation_dataset", "make_redundancy_dataset",
    "make_consensus_dataset", "DEFAULT_GENERATORS",
    "ConsensusConfig", "consensus_score", "make_consensus_rank_fn", "corpus_mrr_at_k",
    "tune_consensus", "add_consensus_features", "cluster_sources",
    "build_consensus_query", "CONSENSUS_KEYS",
    "compare_strategies", "select_best_strategy", "strategy_rank_fns",
    "keyphrases", "entity_rerank", "entity_rerank_texts", "rerank_s2_source",
    "mmr_rerank", "make_mmr_rank_fn", "feature_embedding_sim", "cosine_sim_callable",
    "swiss_tournament", "comparator_from_scores",
    "dedup_candidates", "embed_minilm",
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
