"""Unit + behavioural tests for the rank-fusion harness.

Run with:  python -m pytest tests/ -q   (or  python tests/test_rank_fusion.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_fusion import (  # noqa: E402
    Query, make_synthetic_dataset, make_candidate_generation_dataset,
    weighted_rrf, paper_mrr, corpus_mrr, make_rrf_rank_fn,
    make_single_source_rank_fn, tune_rrf,
    oracle_mrr, candidate_recall, candidate_recall_table, bootstrap_mrr_ci,
    fit_select_ranknet, fit_select_soft_mrr,
)
from rank_fusion.metrics import (  # noqa: E402
    reciprocal_rank_first, recall_at_k, ndcg_at_k,
)
from rank_fusion.fusion import feature_bias_term  # noqa: E402


# --- metrics correctness ---------------------------------------------------- #
def test_reciprocal_rank_first():
    assert reciprocal_rank_first(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank_first(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank_first(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank_first(["a", "b", "c"], {"z"}) == 0.0


def test_paper_mrr_per_direction():
    # two true directions: g0 matched at rank 2, g1 matched at rank 4
    ranked = ["x", "a", "y", "b"]
    rel = {"g0": {"a"}, "g1": {"b"}}
    assert abs(paper_mrr(ranked, rel) - (0.5 + 0.25) / 2) < 1e-9


def test_recall_and_ndcg():
    ranked = ["a", "x", "b", "y"]
    rel = {"g0": {"a"}, "g1": {"b"}, "g2": {"z"}}
    assert recall_at_k(ranked, rel, 1) == 1 / 3      # only a in top-1
    assert recall_at_k(ranked, rel, 3) == 2 / 3      # a and b in top-3
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0  # perfect order


# --- RRF mechanics ---------------------------------------------------------- #
def test_weighted_rrf_scores():
    rankings = {"m1": ["a", "b"], "m2": ["b", "a"]}
    ordered, scores = weighted_rrf(rankings, k=60.0)
    # a: 1/61 + 1/62 ; b: 1/62 + 1/61  -> tie, deterministic id order
    assert abs(scores["a"] - scores["b"]) < 1e-12
    assert ordered == ["a", "b"]


def test_weighted_rrf_weighting_changes_order():
    # m1 prefers a, m2 prefers b; up-weighting m2 should surface b first
    rankings = {"m1": ["a", "b"], "m2": ["b", "a"]}
    ordered, _ = weighted_rrf(rankings, weights={"m1": 0.1, "m2": 5.0}, k=1.0)
    assert ordered[0] == "b"


def test_rrf_spans_full_pool():
    rankings = {"m1": ["a"]}
    ordered, _ = weighted_rrf(rankings, candidate_ids=["a", "b", "c"])
    assert set(ordered) == {"a", "b", "c"} and ordered[0] == "a"


# --- behavioural: fusion and tuning actually help MRR ----------------------- #
def test_fusion_beats_best_single_and_tuning_helps():
    ds = make_synthetic_dataset(n_papers=200, seed=3)
    sources = ds.all_sources()

    single = {s: corpus_mrr(ds.dev, make_single_source_rank_fn(s)) for s in sources}
    best_single = max(single.values())

    uni = corpus_mrr(ds.dev, make_rrf_rank_fn({s: 1.0 for s in sources}, 60.0))
    cfg = tune_rrf(ds.dev, sources, n_random=200, coord_rounds=2, seed=3)

    # uniform fusion should beat the best individual ranker ...
    assert uni > best_single + 1e-4, (uni, best_single)
    # ... and MRR-tuned fusion should not be worse than uniform on dev.
    assert cfg.dev_mrr >= uni - 1e-9, (cfg.dev_mrr, uni)


def test_tuned_rrf_generalises_to_test():
    ds = make_synthetic_dataset(n_papers=300, seed=11)
    sources = ds.all_sources()
    cfg = tune_rrf(ds.dev, sources, n_random=300, coord_rounds=3, seed=11)
    uni_fn = make_rrf_rank_fn({s: 1.0 for s in sources}, 60.0)
    test_uni = corpus_mrr(ds.test, uni_fn)
    test_tuned = corpus_mrr(ds.test, cfg.rank_fn())
    # tuned should transfer to held-out test (allow tiny slack for variance)
    assert test_tuned >= test_uni - 0.01, (test_tuned, test_uni)


# --- Level 2: per-source k and feature-bias term ---------------------------- #
def test_per_source_k_and_feature_bias():
    rankings = {"m1": ["a", "b"], "m2": ["b", "a"]}
    # per-source k as a dict must be accepted and used
    _, sc = weighted_rrf(rankings, k={"m1": 1.0, "m2": 100.0})
    assert sc["a"] > 0 and sc["b"] > 0
    # an additive feature bias can flip the order
    q = Query("p", ["a", "b"], {"m1": ["a", "b"]},
              {"g": {"b"}}, features={"a": {"x": 0.0}, "b": {"x": 1.0}})
    bias = feature_bias_term(q, {"x": 10.0})
    ordered, _ = weighted_rrf(q.rankings, candidate_ids=q.candidate_ids,
                              feature_bias=bias)
    assert ordered[0] == "b"


# --- Level 1: candidate recall / oracle ------------------------------------- #
def test_candidate_recall_union_beats_singletons():
    cg = make_candidate_generation_dataset(n_papers=150, seed=13)
    gens = list(cg.queries[0].candidate_sources.keys())
    tbl = candidate_recall_table(cg.dev, gens)
    best_single = max(tbl[g] for g in gens)
    assert tbl["union"] >= best_single, (tbl["union"], best_single)
    # oracle MRR equals union pool recall (every match could sit at rank 1)
    assert abs(oracle_mrr(cg.dev) - tbl["union"]) < 1e-9


def test_oracle_is_upper_bound_on_actual():
    cg = make_candidate_generation_dataset(n_papers=150, seed=5)
    actual = corpus_mrr(cg.dev, make_single_source_rank_fn("ranker"))
    assert oracle_mrr(cg.dev) >= actual - 1e-9


# --- Level 3 / 4: learned rankers are competitive --------------------------- #
def test_ranknet_beats_best_single():
    ds = make_synthetic_dataset(n_papers=200, seed=7)
    src = ds.all_sources()
    best_single = max(corpus_mrr(ds.dev, make_single_source_rank_fn(s)) for s in src)
    rn, rn_dev, _ = fit_select_ranknet(ds.train, ds.dev, src,
                                       lr_grid=(0.05, 0.1), epochs_grid=(200,), seed=7)
    assert rn_dev >= best_single - 1e-3, (rn_dev, best_single)


def test_listnet_beats_best_single():
    from rank_fusion import fit_select_listnet
    ds = make_synthetic_dataset(n_papers=200, seed=7)
    src = ds.all_sources()
    best_single = max(corpus_mrr(ds.dev, make_single_source_rank_fn(s)) for s in src)
    ln, ln_dev, _ = fit_select_listnet(ds.train, ds.dev, src,
                                       lr_grid=(0.05, 0.1), epochs_grid=(300,), seed=7)
    # listwise loss should at least match the best single ranker on dev MRR
    assert ln_dev >= best_single - 1e-3, (ln_dev, best_single)
    ranked = ln.rank_fn()(ds.test[0])
    assert set(ranked) == set(ds.test[0].candidate_ids)  # valid permutation


def test_soft_mrr_runs_and_ranks():
    ds = make_synthetic_dataset(n_papers=120, seed=7)
    src = ds.all_sources()
    sm, sm_dev, _ = fit_select_soft_mrr(ds.train, ds.dev, src,
                                        tau_grid=(0.5, 1.0), seed=7)
    ranked = sm.rank_fn()(ds.test[0])
    assert set(ranked) == set(ds.test[0].candidate_ids)  # a valid permutation
    assert sm_dev > 0.0


# --- Level 5: cost-aware tuning drops an expensive useless source ----------- #
def test_cost_aware_drops_expensive_source():
    ds = make_synthetic_dataset(n_papers=200, seed=7)
    src = ds.all_sources()
    # make citation (already weakest) very expensive -> governor should zero it
    costs = {s: (5.0 if s == "citation" else 0.1) for s in src}
    cfg = tune_rrf(ds.dev, src, n_random=300, coord_rounds=3,
                   costs=costs, lambda_cost=0.05, seed=7)
    assert cfg.weights["citation"] == 0.0, cfg.weights


# --- bootstrap CI brackets the point estimate ------------------------------- #
def test_bootstrap_ci_brackets_point():
    ds = make_synthetic_dataset(n_papers=200, seed=7)
    fn = make_rrf_rank_fn({s: 1.0 for s in ds.all_sources()}, 60.0)
    pt, lo, hi = bootstrap_mrr_ci(ds.test, fn, n_boot=400, seed=1)
    assert lo <= pt <= hi


# --- Explainability layer (Direction Cards, plan §6) ------------------------ #
def test_rrf_attribution_sums_to_score():
    from rank_fusion.explain import rrf_attribution
    from rank_fusion import weighted_rrf
    q = make_synthetic_dataset(n_papers=5, seed=1).queries[0]
    cid = q.candidate_ids[0]
    _, scores = weighted_rrf(q.rankings, {s: 1.0 for s in q.sources}, 60.0,
                             q.candidate_ids)
    attr = rrf_attribution(q, cid, {s: 1.0 for s in q.sources}, 60.0)
    assert abs(sum(attr.values()) - scores[cid]) < 1e-9  # exact decomposition


def test_ablation_probe_signs():
    from rank_fusion.explain import ablation_probe
    q = make_synthetic_dataset(n_papers=5, seed=2).queries[0]
    cid = q.candidate_ids[0]
    deltas = ablation_probe(q, cid, {s: 1.0 for s in q.sources}, 60.0)
    assert set(deltas.keys()) == set(q.sources)
    assert all(isinstance(v, int) for v in deltas.values())


def test_logical_reasoner_verdicts():
    from rank_fusion import reason_logically
    # strong consensus + multi-source -> recommend
    v, _, _ = reason_logically({"agent_agreement": True, "multi_source": True})
    assert v == "recommend"
    # thin evidence defeats everything -> abstain
    v, _, _ = reason_logically({"agent_agreement": True, "multi_source": True,
                                "thin": True})
    assert v == "abstain"
    # disagreement downgrades consensus -> caveat
    v, _, _ = reason_logically({"agent_agreement": True, "multi_source": True,
                                "disagreement": True})
    assert v == "recommend_with_caveat"


def test_linear_shap_sums_to_logit():
    import numpy as np
    from rank_fusion import fit_select_meta_ranker, linear_shap
    ds = make_synthetic_dataset(n_papers=120, seed=3)
    src = ds.all_sources()
    mr, _, _ = fit_select_meta_ranker(ds.train, ds.dev, src, seed=3)
    q = ds.test[0]
    cid = q.candidate_ids[0]
    shap = linear_shap(mr, q, cid)
    # SHAP values + intercept must reconstruct the model logit (exactness)
    from rank_fusion.meta_ranker import _query_features
    cand_ids, X = _query_features(q, src, mr.k)
    i = cand_ids.index(cid)
    z = (X[i, mr.keep_cols] - mr.scaler.mean_) / mr.scaler.scale_
    logit = float(mr.model.intercept_[0] + mr.model.coef_[0] @ z)
    assert abs(sum(shap.values()) + mr.model.intercept_[0] - logit) < 1e-6


def test_faithfulness_and_evidence():
    from rank_fusion import faithfulness_check, ground_evidence
    assert faithfulness_check(["gpt"], ["gpt", "claude"])["faithful"] is True
    assert faithfulness_check(["recency"], ["gpt", "claude"])["faithful"] is False
    gaps = [{"description": "single-dataset evaluation limits generality",
             "status": "explicit"}]
    addressed, spans = ground_evidence(
        "Evaluate on multiple datasets to test generality across domains.",
        gaps, tau=0.05)
    assert addressed and "explicit" in addressed[0]


def test_build_direction_card_end_to_end():
    from rank_fusion import build_direction_card
    ds = make_synthetic_dataset(n_papers=20, attach_features=True, seed=7)
    q = ds.queries[0]
    w = {s: 1.0 for s in q.sources}
    card = build_direction_card(q, q.candidate_ids[0], rank=1, weights=w, k=60.0)
    assert card.verdict in {"recommend", "recommend_with_caveat", "abstain"}
    assert card.faithfulness["faithful"] in (True, False)
    assert abs(sum(card.attribution.values()) - card.score) < 1e-3


# --- Diversity (MMR), Swiss tournament, dedup ------------------------------- #
def test_mmr_improves_redundancy_mrr():
    from rank_fusion import (make_redundancy_dataset, make_rrf_rank_fn,
                             make_mmr_rank_fn, feature_embedding_sim, corpus_mrr)
    ds = make_redundancy_dataset(n_papers=150, noise_sd=0.2, seed=21)
    src = ds.all_sources()
    base = make_rrf_rank_fn({s: 1.0 for s in src}, 60.0)
    mmr = make_mmr_rank_fn(base, feature_embedding_sim, lambda_=0.8)
    rel_only = corpus_mrr(ds.queries, base)
    with_mmr = corpus_mrr(ds.queries, mmr)
    # diversifying near-duplicate clusters surfaces the buried true direction
    assert with_mmr > rel_only, (with_mmr, rel_only)


def test_mmr_lambda_one_is_relevance_only():
    from rank_fusion import mmr_rerank
    ids = ["a", "b", "c"]
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    sim = lambda x, y: 1.0 if x == y else 0.9   # everything near-duplicate
    assert mmr_rerank(ids, scores, sim, lambda_=1.0) == ["a", "b", "c"]


def test_swiss_tournament_recovers_order():
    from rank_fusion import swiss_tournament, comparator_from_scores
    items = [f"d{i}" for i in range(16)]
    true = {it: 16 - i for i, it in enumerate(items)}     # d0 best
    cmp = comparator_from_scores(true, noise=0.3, seed=1)
    ranked, pts = swiss_tournament(items, cmp, n_rounds=6, seed=1)
    # the true best should land in the top quartile of the aggregated ranking
    assert "d0" in ranked[:4]
    assert pts[ranked[0]] >= pts[ranked[-1]]


def test_dedup_removes_duplicate():
    from rank_fusion import dedup_candidates
    import numpy as np
    ids = ["a", "b", "c"]
    texts = ["x", "y", "z"]
    vecs = {"x": [1.0, 0.0], "y": [0.99, 0.01], "z": [0.0, 1.0]}  # a≈b, c distinct
    kept, rep, backend = dedup_candidates(
        ids, texts, threshold=0.8, embed_fn=lambda ts: np.array([vecs[t] for t in ts]))
    assert kept == ["a", "c"] and rep == {"b": "a"}


# --- Significance, separation diagnostic, match function -------------------- #
def test_paired_bootstrap_detects_real_gap():
    from rank_fusion import (paired_bootstrap_delta, make_rrf_rank_fn,
                             make_single_source_rank_fn)
    ds = make_synthetic_dataset(n_papers=300, seed=7)
    src = ds.all_sources()
    fused = make_rrf_rank_fn({s: 1.0 for s in src}, 60.0)
    weak = make_single_source_rank_fn("citation")   # clearly worse
    res = paired_bootstrap_delta(ds.queries, fused, weak, n_boot=1000, seed=0)
    assert res["delta"] > 0 and res["significant"] and res["p_value"] < 0.05


def test_paired_bootstrap_no_gap_for_identical():
    from rank_fusion import paired_bootstrap_delta, make_rrf_rank_fn
    ds = make_synthetic_dataset(n_papers=150, seed=7)
    fn = make_rrf_rank_fn({s: 1.0 for s in ds.all_sources()}, 60.0)
    res = paired_bootstrap_delta(ds.queries, fn, fn, n_boot=500, seed=0)
    assert abs(res["delta"]) < 1e-9 and not res["significant"]


def test_bonferroni_correction():
    from rank_fusion import bonferroni
    out = bonferroni([0.01, 0.04], alpha=0.05)   # *2 -> 0.02 sig, 0.08 not
    assert out[0]["significant"] and not out[1]["significant"]


def test_separation_diagnostic():
    from rank_fusion import topk_bottomk_separation, make_rrf_rank_fn
    ds = make_synthetic_dataset(n_papers=200, seed=7)
    fn = make_rrf_rank_fn({s: 1.0 for s in ds.all_sources()}, 60.0)
    sep = topk_bottomk_separation(ds.queries, fn, k=5)
    assert sep["gap"] > 0 and sep["top_k_relevance"] > sep["bottom_k_relevance"]


def test_match_relevance_and_tau_calibration():
    from rank_fusion import match_relevance, calibrate_tau
    cand_ids = ["d1", "d2", "d3"]
    cand_texts = ["Evaluate on multiple datasets to test generalization",
                  "Develop a user-friendly decision support tool",
                  "Quantify long-term outcomes with longitudinal cohorts"]
    gt = ["Further evaluation across multiple datasets is needed for generalization"]
    rel = match_relevance(cand_ids, cand_texts, gt, tau=0.05)
    assert "g0" in rel and "d1" in rel["g0"]      # d1 matches the GT
    best = calibrate_tau([("multi dataset evaluation", "evaluate on many datasets"),
                          ("user tool", "quantify health outcomes")], [1, 0])
    assert 0.0 <= best["tau"] <= 0.5 and best["f1"] >= 0.0


# --- Autonomous candidate-harvesting agent ---------------------------------- #
class _FakeS2:
    """Canned S2 client: maps (kind, key) -> list of paper dicts."""
    def __init__(self, table):
        self.table = table
    def get_paper(self, cid):
        return {}
    def get_citations(self, cid, limit=50):
        return self.table.get(("cite", cid), [])
    def get_references(self, cid, limit=50):
        return self.table.get(("ref", cid), [])
    def search(self, query, limit=5):
        for (kind, key), v in self.table.items():
            if kind == "search" and key in query:
                return v
        return []


def _paper(pid, fd_title, fd_rationale):
    return {"paperId": pid, "title": fd_title, "fieldsOfStudy": ["X"],
            "future_directions": [{"title": fd_title, "rationale": fd_rationale}]}


def test_agent_stops_when_all_gaps_covered():
    from rank_fusion import CandidateHarvestAgent, AgentConfig
    gaps = ["Establish referral pathways for perpetrators of violence identified by the tool",
            "Develop tailored adaptations for youths with limited language proficiency"]
    table = {
        ("cite", "SEED"): [_paper("A",
            "Establish referral pathways for perpetrators of violence identified by screening",
            "create clear referral pathways for perpetrators of violence identified")],
        ("search", gaps[1]): [_paper("B",
            "Develop tailored adaptations for youths with limited language proficiency",
            "tailored language adaptations for youths with limited proficiency")],
    }
    agent = CandidateHarvestAgent(_FakeS2(table), AgentConfig(match_tau=0.15))
    res = agent.run("SEED", gaps)
    assert res["coverage"]["n_covered"] == 2
    assert res["stop_reason"] == "all_gaps_covered"
    assert len(res["pool"]) >= 2


def test_agent_stops_on_no_improvement():
    from rank_fusion import CandidateHarvestAgent, AgentConfig
    gaps = ["Establish referral pathways for perpetrators of violence",
            "An entirely unrelated gap about quantum error correction codes"]
    table = {  # only gap 0 is ever coverable; gap 1 never matches anything
        ("cite", "SEED"): [_paper("A",
            "Establish referral pathways for perpetrators of violence",
            "clear referral pathways for perpetrators of violence")],
        ("cite", "A"): [_paper("A2", "More on referral pathways for perpetrators",
                                "referral pathways for perpetrators of violence again")],
    }
    agent = CandidateHarvestAgent(_FakeS2(table), AgentConfig(match_tau=0.15, patience=2))
    res = agent.run("SEED", gaps)
    assert res["coverage"]["n_covered"] == 1            # only gap 0
    assert res["stop_reason"].startswith("no_improvement")
    assert res["s2_calls"] < 30                          # stopped early, didn't burn budget


def test_agent_respects_budget():
    from rank_fusion import CandidateHarvestAgent, AgentConfig
    gaps = ["gap that is never covered by anything here"]
    table = {("cite", "SEED"): [_paper("A", "irrelevant topic one", "nothing matching"),
                                _paper("B", "irrelevant topic two", "still nothing")]}
    # patience high so it would keep going; budget must be the thing that stops it
    agent = CandidateHarvestAgent(_FakeS2(table),
                                  AgentConfig(max_s2_calls=4, patience=99, max_rounds=99))
    res = agent.run("SEED", gaps)
    assert res["s2_calls"] <= 4
    assert res["stop_reason"] in ("budget_exhausted", "frontier_exhausted")


def test_agent_evaluate_pool_recall():
    from rank_fusion import Candidate, evaluate_pool
    pool = [Candidate("c0", "Evaluate the tool across multiple datasets for generalization",
                      [], "seed", None)]
    gt = ["Further evaluation across multiple datasets is needed to test generalization"]
    ev = evaluate_pool(pool, gt, tau=0.08)
    assert ev["recall"] == 1.0 and ev["n_covered_gt"] == 1


# --- Integration: agent output + S2 graph signals -> rank fusion ------------ #
def test_pool_to_query_from_agent_output():
    from rank_fusion import Candidate, pool_to_query
    cands = [Candidate("c0", "Evaluate across multiple datasets for generalization",
                       [], "search:gap0", "A"),
             Candidate("c1", "Develop a decision support tool for policymakers",
                       [], "expand:X", "B")]
    q = pool_to_query("P1", cands, paper_text="datasets generalization evaluation",
                      gt_texts=["Evaluate on more datasets to test generalization"],
                      tau=0.05)
    assert q.candidate_ids == ["c0", "c1"]
    assert set(q.rankings["retrieval"]) == {"c0", "c1"}
    assert q.rankings["retrieval"][0] == "c0"          # most similar to the paper
    assert "agent_search" in q.candidate_sources and "agent_expand" in q.candidate_sources
    assert q.has_ground_truth()                        # c0 matches the GT


def test_add_horizon_source_from_features():
    from rank_fusion import Query, add_horizon_source
    q = Query("p", ["a", "b"], {}, {}, features={
        "a": {"recency": 0.9, "novelty": 0.8, "popularity": 0.1, "relevance": 0.7},
        "b": {"recency": 0.1, "novelty": 0.1, "popularity": 0.1, "relevance": 0.1}})
    add_horizon_source(q)
    assert q.rankings["horizon"] == ["a", "b"]         # a has higher horizon_prior


def test_graph_signal_fn_maps_names():
    from rank_fusion import graph_signal_fn

    class FakeExt:
        def signals_for_text(self, text, ctx, top_k=5):
            return {"graph_relevance": 0.7, "graph_novelty": 0.4,
                    "graph_recency": 0.6, "citation_velocity": 0.3}

    sig = graph_signal_fn(FakeExt(), ctx=None)("a direction")
    assert sig["relevance"] == 0.7 and sig["popularity"] == 0.3
    assert sig["recency"] == 0.6 and sig["novelty"] == 0.4


def test_meta_ranker_consumes_query_features():
    from rank_fusion import make_synthetic_dataset, linear_shap
    from rank_fusion.meta_ranker import MetaRanker
    ds = make_synthetic_dataset(n_papers=150, attach_features=True, seed=7)
    src = ds.all_sources()
    mr = MetaRanker(src, extra_feature_keys=["sim_to_paper", "novelty"]).fit(ds.train)
    assert "sim_to_paper" in mr.feature_names and "novelty" in mr.feature_names
    ranked = mr.rank_fn()(ds.test[0])
    assert set(ranked) == set(ds.test[0].candidate_ids)   # valid permutation
    shap = linear_shap(mr, ds.test[0], ds.test[0].candidate_ids[0])
    assert any(k in ("sim_to_paper", "novelty") for k in shap)


# --- Consensus-aware fusion (GPT + Claude + S2/KG) -------------------------- #
def test_consensus_agreement_and_interaction_terms():
    from rank_fusion import Query, ConsensusConfig, consensus_score
    # 'a' is ranked #1 by both LLMs and present in s2; 'b' is not both-#1
    q = Query("p", ["a", "b", "c"],
              {"llm1": ["a", "b", "c"], "llm2": ["a", "c", "b"], "s2": ["a", "b", "c"]}, {})
    w = {"llm1": 1.0, "llm2": 1.0, "s2": 1.0}
    base = consensus_score(q, ConsensusConfig(weights=w, k=60))
    boosted = consensus_score(q, ConsensusConfig(weights=w, k=60, eta1=5.0))
    assert abs((boosted["a"] - base["a"]) - 5.0) < 1e-9   # a got the both-#1 boost
    assert abs(boosted["b"] - base["b"]) < 1e-9           # b did not
    # rho interaction: a is both-top-t AND in s2 -> extra support
    inter = consensus_score(q, ConsensusConfig(weights=w, k=60, rho=10.0, t=3))
    assert inter["a"] > base["a"]


def test_tune_consensus_beats_uniform_on_dev():
    from rank_fusion import (make_consensus_dataset, tune_consensus,
                             make_rrf_rank_fn, corpus_mrr_at_k)
    ds = make_consensus_dataset(n_papers=150, seed=23)
    src = ds.all_sources()
    cfg = tune_consensus(ds.dev, n_random=150, coord_rounds=2, mrr_k=3, seed=23)
    uni = corpus_mrr_at_k(ds.dev, make_rrf_rank_fn({s: 1.0 for s in src}, 60.0), 3)
    assert cfg.dev_mrr >= uni - 1e-9


def test_cluster_sources_merges_paraphrases():
    from rank_fusion import cluster_sources
    cl, rk, txt = cluster_sources({
        "llm1": ["extend the method to multimodal data"],
        "llm2": ["extend method to multimodal data now"],
        "s2": ["a totally different research topic"]}, threshold=0.3)
    assert len(cl) == 2                              # the two paraphrases merged
    merged = set(rk["llm1"]) & set(rk["llm2"])       # same cluster ranked by both LLMs
    assert len(merged) == 1


def test_s2_agent_source_follows_agent_rank():
    from rank_fusion import s2_agent_source
    # Agent C output: (cand_ids, cand_texts, ranked_ids, features)
    out = (["p_C1", "p_C2", "p_C3"], ["beta", "alpha", "gamma"],
           ["p_C1", "p_C3", "p_C2"], {})
    assert s2_agent_source(out) == ["beta", "gamma", "alpha"]   # ranked order


def test_s2_source_fuses_into_consensus_query():
    from rank_fusion import build_consensus_query
    q = build_consensus_query("p", {
        "llm1": ["evaluate on multiple datasets for generalization"],
        "llm2": ["test across many datasets"],
        "s2": ["A study on dataset evaluation across clinical domains"]},
        gt_texts=["evaluate on more datasets to test generalization"],
        tau=0.05, threshold=0.5)
    assert "s2" in q.rankings and q.rankings["s2"]    # the S2 source is fused in
    assert q.candidate_ids


def test_build_consensus_query_end_to_end():
    from rank_fusion import build_consensus_query, CONSENSUS_KEYS
    q = build_consensus_query("p", {
        "llm1": ["evaluate on multiple datasets for generalization"],
        "llm2": ["test the method across many datasets"],
        "s2": ["dataset evaluation across domains"]},
        gt_texts=["evaluate on more datasets to test generalization"],
        tau=0.05, threshold=0.2)
    assert q.candidate_ids and q.has_ground_truth()
    feats = next(iter(q.features.values()))
    assert all(key in feats for key in CONSENSUS_KEYS)


# --- Best-of selection (LLM1 / LLM2 / S2 / fused, by MRR) -------------------- #
def test_compare_and_select_strategies():
    from rank_fusion import (make_consensus_dataset, compare_strategies,
                             select_best_strategy, make_rrf_rank_fn)
    ds = make_consensus_dataset(n_papers=150, seed=23)
    src = ds.all_sources()                          # llm1, llm2, s2
    fused = {"fused:RRF": make_rrf_rank_fn({s: 1.0 for s in src}, 60.0)}
    rows = compare_strategies(ds.dev, src, fused_rank_fns=fused)
    assert {r["name"] for r in rows} == {
        "single:llm1", "single:llm2", "single:s2", "fused:RRF"}
    assert rows == sorted(rows, key=lambda r: -r["mrr"])     # sorted desc
    sel = select_best_strategy(ds.dev, src, fused_rank_fns=fused)
    assert sel["best_dev_mrr"] == rows[0]["mrr"]             # winner tops the table
    best_single = max(r["mrr"] for r in rows if r["name"].startswith("single:"))
    assert sel["best_dev_mrr"] >= best_single - 1e-12        # never worse than best single


def test_select_top_sources_fair_scoring():
    from rank_fusion import Query, select_top_sources
    # A: relevant at rank 1 (MRR 1.0); B: rank 2 (0.5); C: no match anywhere (0)
    qA = [Query("p1", ["a", "b"], {}, {"g": {"a"}}),
          Query("p2", ["a", "b"], {}, {"g": {"a"}})]
    qB = [Query("p1", ["b", "a"], {}, {"g": {"a"}}),
          Query("p2", ["b", "a"], {}, {"g": {"a"}})]
    qC = [Query("p1", ["b", "c"], {}, {}),          # empty relevance -> miss -> 0
          Query("p2", ["b", "c"], {}, {})]
    res = select_top_sources({"A": qA, "B": qB, "C": qC}, k=2)
    assert res["top_k"] == ["A", "B"]
    mrr = {r["source"]: r["mrr"] for r in res["ranked"]}
    assert abs(mrr["A"] - 1.0) < 1e-9 and abs(mrr["B"] - 0.5) < 1e-9 and mrr["C"] == 0.0
    # fair denominator: C's misses are scored (0), not skipped
    assert all(r["n_papers"] == 2 for r in res["ranked"])


def test_select_cv_prefers_simpler_on_near_tie():
    from rank_fusion import Query, select_best_strategy_cv, make_rrf_rank_fn
    # llm1 always ranks the true 'a' first (MRR 1.0); a "fusion" that's ~equal
    # should NOT be chosen over the simpler single source on a near-tie.
    qs = [Query(f"p{i}", ["a", "b"], {"llm1": ["a", "b"], "llm2": ["a", "b"]},
                {"g": {"a"}}) for i in range(20)]
    builders = {"uniform RRF": lambda tr: make_rrf_rank_fn({"llm1": 1.0, "llm2": 1.0}, 60.0)}
    res = select_best_strategy_cv(qs, ["llm1", "llm2"], fused_builders=builders,
                                  n_folds=4, margin=0.02)
    assert res["best_name"].startswith("single:")      # simpler wins the near-tie


def test_select_falls_back_when_fusion_dilutes():
    from rank_fusion import Query, select_best_strategy
    # llm1 is perfect (true 'a' at #1); llm2/s2 bury 'a' -> uniform RRF demotes it
    qs = [Query(f"p{i}", ["a", "b", "c", "d"],
                {"llm1": ["a", "b", "c", "d"], "llm2": ["b", "c", "d", "a"],
                 "s2": ["c", "d", "b", "a"]}, {"g0": {"a"}}) for i in range(15)]
    sel = select_best_strategy(qs, ["llm1", "llm2", "s2"])
    assert sel["best_name"] == "single:llm1"                 # falls back to best single


# --- Entity-grounded S2 re-ranking ------------------------------------------ #
def test_entity_rerank_surfaces_gap_aligned():
    from rank_fusion import entity_rerank_texts
    gap = ("referral pathways for perpetrators; tailored adaptations for youths "
           "with intellectual disabilities and limited language proficiency")
    cands = ["Battery cathode materials optimization for energy storage",
             "Adapt the tool for youths with intellectual disabilities and language barriers",
             "Citation network recency trend analysis"]
    ids = ["c0", "c1", "c2"]
    ranked, info = entity_rerank_texts(ids, cands, gap)
    assert ranked[0] == "c1"                       # gap-aligned candidate first
    assert info["c1"]["gap_hits"] >= 2
    assert info["c0"]["gap_hits"] == 0             # off-topic -> no gap entities


def test_entity_rerank_core_on_sets():
    from rank_fusion import entity_rerank
    cand_ents = {"a": {"intellectual disabilities", "language"},
                 "b": {"battery", "cathode"}}
    ranked, info = entity_rerank(["a", "b"], cand_ents,
                                 gap_entities={"intellectual disabilities", "language", "referral"})
    assert ranked == ["a", "b"]
    assert info["a"]["gap_hits"] == 2 and info["b"]["gap_hits"] == 0


# --- White-space gap-bridging generator ------------------------------------- #
def test_whitespace_generates_adjacent_bridges():
    from rank_fusion import whitespace_candidates, whitespace_source
    seed = {"sexit", "school"}
    neighbors = [
        "sexit and intellectual disabilities in youth care",
        "sexit screening for youth with intellectual disabilities",
        "unrelated quantum computing error correction codes",   # non-adjacent
    ]
    cands = whitespace_candidates(seed, neighbors, min_support=2,
                                  max_df_ratio=1.0, top_k=10)
    ents = {c.entity for c in cands}
    assert any("disabilities" in e for e in ents)       # adjacent+recurring bridge
    assert not any("quantum" in e for e in ents)         # non-adjacent -> excluded
    assert all(c.connects for c in cands)                # every bridge ties to the seed
    assert whitespace_source(cands)[0].startswith("Investigate")


def test_whitespace_prefers_sparse_over_ubiquitous():
    from rank_fusion import whitespace_candidates
    # 'rarebridge' co-occurs with seed in 2 of 10 docs (sparse, tight);
    # 'common' co-occurs with seed but appears almost everywhere (generic)
    seed = {"seed"}
    docs = ["seed rarebridge"] * 2 + ["seed common filler%d" % i for i in range(8)]
    cands = whitespace_candidates(seed, docs, min_support=2, max_df_ratio=1.0, top_k=10)
    by = {c.entity: c.score for c in cands}
    assert by.get("rarebridge", 0) > by.get("common", 0)   # sparse beats ubiquitous


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
