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
