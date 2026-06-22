"""Demo: diversity-aware MMR, Swiss-tournament aggregation, and candidate dedup.

Three additions to the rank-fusion engine, each shown working:

  1. MMR re-ranking (§9.3) — on a synthetic "redundancy" corpus where near-
     duplicate candidates crowd the top and bury a distinct true direction,
     MMR lifts per-direction MRR by surfacing the buried gap.
  2. Swiss tournament — aggregate sparse *pairwise* comparisons into a global
     ranking at ~N·n/2 comparisons instead of n²/2 (for a pairwise Agent-2).
  3. Candidate dedup (Si et al.) — all-MiniLM-L6-v2 cosine >= 0.8 (TF-IDF
     fallback), collapsing near-duplicate directions to one representative.

    python run_diversity.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    make_redundancy_dataset, make_rrf_rank_fn, corpus_mrr, evaluate,
    make_mmr_rank_fn, feature_embedding_sim,
    swiss_tournament, comparator_from_scores,
    dedup_candidates,
)


def demo_mmr():
    print("=" * 72)
    print("1) DIVERSITY-AWARE MMR RE-RANKING (§9.3)")
    print("=" * 72)
    ds = make_redundancy_dataset(n_papers=250, noise_sd=0.2, seed=21)
    src = ds.all_sources()
    base = make_rrf_rank_fn({s: 1.0 for s in src}, 60.0)
    print("Redundancy corpus: each paper has near-duplicate candidate clusters; "
          "two\ndistinct clusters hold the true directions, one far more salient "
          "than the other.\n")
    print(f"{'system':<34}{'MRR':>8}{'Recall@3':>10}{'nDCG@5':>9}")
    rel = evaluate(ds.test, base, ks=(3, 5))
    print(f"{'relevance-only RRF':<34}{rel['MRR']:>8.4f}{rel['Recall@3']:>10.4f}"
          f"{rel['nDCG@5']:>9.4f}")
    for lam in (0.9, 0.8, 0.7):
        mmr = make_mmr_rank_fn(base, feature_embedding_sim, lambda_=lam)
        m = evaluate(ds.test, mmr, ks=(3, 5))
        print(f"{f'  + MMR (lambda={lam})':<34}{m['MRR']:>8.4f}"
              f"{m['Recall@3']:>10.4f}{m['nDCG@5']:>9.4f}")
    print("\n  -> MMR demotes near-duplicates so the top-K covers distinct gaps, "
          "raising\n     per-direction MRR (it recovers the buried second true "
          "direction).\n")


def demo_swiss():
    print("=" * 72)
    print("2) SWISS-TOURNAMENT AGGREGATION OF PAIRWISE COMPARISONS")
    print("=" * 72)
    n = 20
    items = [f"cand{i:02d}" for i in range(n)]
    true = {it: n - i for i, it in enumerate(items)}      # cand00 is the best
    print(f"{n} candidates, a noisy pairwise judge (e.g. GPT 'is A a better next "
          f"step than B?').")
    for rounds in (3, 5, 8):
        cmp = comparator_from_scores(true, noise=0.4, seed=1)
        ranked, pts = swiss_tournament(items, cmp, n_rounds=rounds, seed=1)
        # how many of the true top-5 are in the recovered top-5
        hit = len(set(ranked[:5]) & set(items[:5]))
        comps = rounds * n // 2
        print(f"  N={rounds} rounds (~{comps:>3} comparisons): top-5 recovers "
              f"{hit}/5 of the true best  | top-3 = {ranked[:3]}")
    print(f"  (full round-robin would be {n*(n-1)//2} comparisons; Swiss spends "
          f"them on near-ties.)\n")


def demo_dedup():
    print("=" * 72)
    print("3) CANDIDATE DEDUP — Si et al. recipe (all-MiniLM-L6-v2, cosine >= 0.8)")
    print("=" * 72)
    # a candidate pool with two near-duplicate pairs (paraphrases / overlaps)
    pool = {
        "g1": "Scale up language access and cultural safety interventions in GBV services",
        "g2": "Scale up language access and cultural safety interventions across GBV service settings",
        "r1": "Build multinational standardized datasets with intersectional identity modules",
        "c1": "Quantify health and non-health outcomes of unhindered GBV service access",
        "c2": "Quantify the health and non-health outcomes of unhindered access to GBV services",
    }
    ids, texts = list(pool.keys()), list(pool.values())
    kept, rep, backend = dedup_candidates(ids, texts, threshold=0.6)  # 0.6 for TF-IDF
    print(f"backend = {backend}  (install sentence-transformers to use MiniLM @ 0.8)\n")
    print(f"pool ({len(ids)}): {ids}")
    print(f"kept ({len(kept)}): {kept}")
    print(f"collapsed duplicates -> representative: {rep}")
    print("\n  -> near-duplicate directions collapse to one representative, "
          "keeping the\n     candidate pool tight before the rankers see it.\n")


if __name__ == "__main__":
    demo_mmr()
    demo_swiss()
    demo_dedup()
