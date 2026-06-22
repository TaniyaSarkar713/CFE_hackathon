"""End-to-end rank-fusion experiment on synthetic data.

Implements the staged optimisation strategy:

  Level 1  candidate-pool recall (the ceiling)      -> oracle MRR / Recall table
  Level 2  weighted RRF (per-source w, k)           -> uniform vs MRR-tuned
  Level 3  learned meta-rankers                      -> logistic / pairwise RankNet
  Level 4  differentiable soft-MRR surrogate         -> SoftMRRRanker
  Level 5  cost-aware fusion (MRR - lambda*cost)     -> compute-risk governor

Everything is selected on dev and reported on test (temporal split, no leakage),
with bootstrap 95% CIs so a small dev set isn't over-read.

    python run_synthetic.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")  # silence benign sklearn/numpy BLAS warnings

from rank_fusion import (
    make_synthetic_dataset, make_candidate_generation_dataset,
    DEFAULT_SKILLS, evaluate, corpus_mrr,
    make_rrf_rank_fn, make_single_source_rank_fn,
    tune_rrf, fit_select_meta_ranker,
    fit_select_ranknet, fit_select_listnet, fit_select_soft_mrr,
    oracle_mrr, candidate_recall_table, bootstrap_mrr_ci,
)


def line(name, dev, test, ci=None, extra=""):
    cis = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
    return f"{name:<32} {dev:>7.4f} {test:>8.4f}  {cis:<16} {extra}"


def level1_candidate_recall(seed=13):
    print("#" * 72)
    print("# LEVEL 1 — candidate-pool recall is the ceiling (tune this FIRST)")
    print("#" * 72)
    cg = make_candidate_generation_dataset(seed=seed)
    gens = list(cg.queries[0].candidate_sources.keys())
    tbl = candidate_recall_table(cg.dev, gens)
    print(f"{len(cg.queries)} papers; per-generator candidate recall (dev):")
    for s in gens:
        print(f"   {s:<22} Recall = {tbl[s]:.3f}")
    print(f"   {'UNION pool':<22} Recall = {tbl['union']:.3f}   "
          f"(oracle MRR ceiling = {oracle_mrr(cg.dev):.3f})")
    actual = corpus_mrr(cg.dev, make_single_source_rank_fn("ranker"))
    print(f"   actual single-ranker MRR over the union pool = {actual:.3f}")
    print("   -> union recall >> any single source; no ranker can exceed the "
          "oracle ceiling.\n")


def main(seed: int = 7):
    level1_candidate_recall()

    ds = make_synthetic_dataset(seed=seed)
    sources = ds.all_sources()
    print("#" * 72)
    print("# LEVELS 2-5 — ranking & fusion on the candidate pool")
    print("#" * 72)
    print(f"{len(ds.queries)} papers "
          f"({len(ds.train)} train / {len(ds.dev)} dev / {len(ds.test)} test); "
          f"skills={DEFAULT_SKILLS}\n")

    rows = []  # (name, dev, test, ci_on_test, extra)

    # --- per-source single rankers --------------------------------------- #
    for s in sources:
        fn = make_single_source_rank_fn(s)
        rows.append((f"single ranker: {s}", corpus_mrr(ds.dev, fn),
                     corpus_mrr(ds.test, fn), None, ""))

    # --- Level 2: uniform vs tuned RRF ----------------------------------- #
    uni_fn = make_rrf_rank_fn({s: 1.0 for s in sources}, 60.0)
    rows.append(("uniform RRF (k=60)", corpus_mrr(ds.dev, uni_fn),
                 corpus_mrr(ds.test, uni_fn), None, ""))

    cfg = tune_rrf(ds.dev, sources, n_random=500, coord_rounds=4, seed=seed)
    tuned_fn = cfg.rank_fn()
    rows.append(("tuned RRF (argmax dev MRR)", cfg.dev_mrr,
                 corpus_mrr(ds.test, tuned_fn),
                 bootstrap_mrr_ci(ds.test, tuned_fn, n_boot=500, seed=1)[1:],
                 f"k={cfg.k:g}"))

    # --- Level 5: cost-aware tuned RRF ----------------------------------- #
    costs = {"gpt": 3.0, "claude": 3.0, "retrieval": 0.2, "citation": 0.2}
    cfg_c = tune_rrf(ds.dev, sources, n_random=500, coord_rounds=4,
                     costs=costs, lambda_cost=0.02, seed=seed)
    dev_pure = corpus_mrr(ds.dev, cfg_c.rank_fn())
    active = [s for s in sources if cfg_c.weights[s] > 0]
    rows.append(("cost-aware RRF (MRR-lambda*cost)", dev_pure,
                 corpus_mrr(ds.test, cfg_c.rank_fn()), None,
                 f"active={','.join(active)}"))

    # --- Level 3: learned meta-rankers ----------------------------------- #
    mr, mr_dev, _ = fit_select_meta_ranker(ds.train, ds.dev, sources, seed=seed)
    rows.append(("meta-ranker: logistic", mr_dev,
                 corpus_mrr(ds.test, mr.rank_fn()), None, ""))

    rn, rn_dev, rn_info = fit_select_ranknet(ds.train, ds.dev, sources, seed=seed)
    rows.append(("meta-ranker: pairwise RankNet", rn_dev,
                 corpus_mrr(ds.test, rn.rank_fn()),
                 bootstrap_mrr_ci(ds.test, rn.rank_fn(), n_boot=500, seed=1)[1:],
                 str(rn_info)))

    ln, ln_dev, ln_info = fit_select_listnet(ds.train, ds.dev, sources, seed=seed)
    rows.append(("meta-ranker: listwise ListNet", ln_dev,
                 corpus_mrr(ds.test, ln.rank_fn()), None, str(ln_info)))

    # --- Level 4: differentiable soft-MRR surrogate ---------------------- #
    sm, sm_dev, sm_info = fit_select_soft_mrr(ds.train, ds.dev, sources, seed=seed)
    rows.append(("soft-MRR surrogate (Level 4)", sm_dev,
                 corpus_mrr(ds.test, sm.rank_fn()), None, str(sm_info)))

    # ---- ablation table -------------------------------------------------- #
    print(f"{'System':<32} {'DevMRR':>7} {'TestMRR':>8}  {'95% CI (test)':<16} notes")
    print("-" * 88)
    for r in rows:
        print(line(*r))
    print("-" * 88)

    print("\nFull metric bundle for tuned RRF on TEST:")
    print("   " + "  ".join(f"{k}={v:.4f}" for k, v in evaluate(ds.test, tuned_fn).items()))


if __name__ == "__main__":
    main()
