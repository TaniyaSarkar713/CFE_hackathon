"""Demo: statistical rigor for the ablation (Si et al. 2024 eval methodology).

Bootstrap CIs tell you how stable one system is; this script tells you whether
the *gaps between systems* are real:

  1. Per-system MRR + 95% CI (test).
  2. Each system vs the best single ranker — paired-bootstrap delta, CI, and a
     Bonferroni-corrected p-value (family-wise over all comparisons).
  3. Ranker validation: top-K vs bottom-K relevance separation (Si Table 1).

    python run_significance.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    make_synthetic_dataset, corpus_mrr, bootstrap_mrr_ci,
    make_rrf_rank_fn, make_single_source_rank_fn,
    tune_rrf, fit_select_ranknet, fit_select_listnet,
    compare_to_baseline, topk_bottomk_separation,
)


def main(seed: int = 7):
    ds = make_synthetic_dataset(seed=seed)
    src = ds.all_sources()

    # build the systems we want to compare
    single = {s: make_single_source_rank_fn(s) for s in src}
    best_single = max(src, key=lambda s: corpus_mrr(ds.dev, single[s]))
    cfg = tune_rrf(ds.dev, src, n_random=400, coord_rounds=3, seed=seed)
    rn, _, _ = fit_select_ranknet(ds.train, ds.dev, src, seed=seed)
    ln, _, _ = fit_select_listnet(ds.train, ds.dev, src, seed=seed)

    systems = {
        f"single:{best_single}": single[best_single],
        "uniform RRF": make_rrf_rank_fn({s: 1.0 for s in src}, 60.0),
        "tuned RRF": cfg.rank_fn(),
        "RankNet": rn.rank_fn(),
        "ListNet": ln.rank_fn(),
    }

    print("=" * 70)
    print("1) Per-system MRR with 95% bootstrap CI (TEST)")
    print("=" * 70)
    for name, fn in systems.items():
        pt, lo, hi = bootstrap_mrr_ci(ds.test, fn, n_boot=2000, seed=1)
        print(f"   {name:<18} MRR = {pt:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 70)
    print(f"2) Each system vs baseline '{f'single:{best_single}'}' "
          f"(paired bootstrap, Bonferroni)")
    print("=" * 70)
    rows = compare_to_baseline(ds.test, systems, baseline=f"single:{best_single}",
                               n_boot=2000, seed=1)
    print(f"   {'system':<14}{'ΔMRR':>8}{'95% CI':>20}{'p':>9}{'p_corr':>9}  sig")
    for r in rows:
        ci = f"[{r['lo']:+.3f},{r['hi']:+.3f}]"
        sig = "**" if r["significant_corrected"] else ""
        print(f"   {r['system']:<14}{r['delta']:>+8.4f}{ci:>20}"
              f"{r['p_value']:>9.4f}{r['p_corrected']:>9.4f}  {sig}")
    print("   (** = significant after Bonferroni correction; CI excluding 0 ⇔ real gap)")

    print("\n" + "=" * 70)
    print("3) Ranker validation — top-K vs bottom-K relevance separation (TEST)")
    print("=" * 70)
    for name, fn in systems.items():
        sep = topk_bottomk_separation(ds.test, fn, k=5)
        print(f"   {name:<18} top-5={sep['top_k_relevance']:.3f}  "
              f"bottom-5={sep['bottom_k_relevance']:.3f}  gap={sep['gap']:+.3f}")
    print("   (a large positive gap = the ranker genuinely separates good from bad)")


if __name__ == "__main__":
    main()
