"""Demo: consensus-aware rank fusion (GPT + Claude + S2/KG).

Encodes the intuition "if both LLMs independently rank the same direction near
the top that's strong evidence; S2/KG supports it" as MRR@3-tuned fusion with
explicit LLM-agreement and agreement×S2-support terms — and lets the dev labels
decide how much each matters.

Stages compared (all tuned on dev MRR@3, reported on test):
  per-source -> uniform RRF -> tuned RRF -> consensus-aware RRF -> meta-ranker.

    python run_consensus.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    make_consensus_dataset, corpus_mrr_at_k, make_rrf_rank_fn,
    make_single_source_rank_fn, tune_consensus, add_consensus_features,
    fit_select_meta_ranker, CONSENSUS_KEYS,
)

K = 3


def main(seed: int = 23):
    ds = make_consensus_dataset(seed=seed)
    src = ds.all_sources()                       # ["llm1", "llm2", "s2"]
    print(f"{len(ds.queries)} papers (llm1, llm2, s2); reporting MRR@{K} on test.\n")

    rows = []
    for s in src:
        rows.append((f"single: {s}", corpus_mrr_at_k(ds.test, make_single_source_rank_fn(s), K)))
    rows.append(("uniform RRF", corpus_mrr_at_k(
        ds.test, make_rrf_rank_fn({s: 1.0 for s in src}, 60.0), K)))

    # plain weighted-RRF vs consensus-aware, same MRR@3 objective
    plain = tune_consensus(ds.dev, mrr_k=K, use_consensus_terms=False, seed=seed)
    cons = tune_consensus(ds.dev, mrr_k=K, use_consensus_terms=True, seed=seed)
    rows.append(("tuned RRF (w, k)", corpus_mrr_at_k(ds.test, plain.rank_fn(), K)))
    rows.append(("consensus-aware RRF (+ η, ρ)", corpus_mrr_at_k(ds.test, cons.rank_fn(), K)))

    # Stage 2: learned meta-ranker over rank features + consensus features
    for split in (ds.train, ds.dev, ds.test):
        for q in split:
            add_consensus_features(q, ("llm1", "llm2"), "s2")
    mr, _, info = fit_select_meta_ranker(ds.train, ds.dev, src,
                                         extra_feature_keys=CONSENSUS_KEYS, seed=seed)
    rows.append(("meta-ranker (+ consensus feats)", corpus_mrr_at_k(ds.test, mr.rank_fn(), K)))

    print(f"{'system':<34}{'test MRR@'+str(K):>12}")
    print("-" * 46)
    for name, v in rows:
        print(f"{name:<34}{v:>12.4f}")
    print("-" * 46)
    print(f"\nconsensus-aware learned weights: "
          f"w={ {s: round(cons.weights[s], 2) for s in src} }, k={cons.k:g}")
    print(f"  agreement boosts:  η1(both #1)={cons.eta1}  η3(both top-{cons.t})={cons.eta3}")
    print(f"  S2-supports-consensus interaction:  ρ={cons.rho}")
    print("\nReading it: the tuner LETS THE LABELS DECIDE. A large η means LLM "
          "agreement is\npredictive beyond what RRF already captures; a large ρ "
          "means S2 helps mainly when\nit backs the LLM consensus. Additive RRF "
          "already captures much of the agreement,\nso the clearest gains usually "
          "come from the meta-ranker's nonlinear use of the\nagreement×S2 feature.")


if __name__ == "__main__":
    main()
