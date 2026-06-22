"""Run the MRR harness on the real `Data/` folder.

Proves the metric works on real future-direction text: loads each paper's
generated `future_directions`, matches them against the masked ground-truth
future-work text via TF-IDF cosine, and reports MRR / Recall@K for the model's
given ordering (the single "summary" ranker currently available in Data/).

    python run_realdata.py [--tau 0.12]
"""

from __future__ import annotations

import argparse
import os

from rank_fusion import load_real_dataset, evaluate, corpus_mrr
from rank_fusion.fusion import make_single_source_rank_fn

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.12,
                    help="TF-IDF cosine match threshold")
    ap.add_argument("--data", default=DATA_DIR)
    args = ap.parse_args()

    ds, diag = load_real_dataset(args.data, tau=args.tau)
    n_with_gt = sum(1 for q in ds.queries if q.has_ground_truth())
    print(f"Loaded {len(ds.queries)} papers with summary+GT; "
          f"{n_with_gt} have >=1 matched ground-truth direction at tau={args.tau}\n")

    print(f"{'paper':<12} {'#cand':>5} {'#gt':>4} {'#matched':>9} {'maxsim':>7}")
    for pid, d in sorted(diag.items()):
        print(f"{pid:<12} {d['n_candidates']:>5} {d['n_gt_sentences']:>4} "
              f"{d['n_matched_gt']:>9} {d['max_sim']:>7.3f}")

    fn = make_single_source_rank_fn("summary")
    print(f"\nMRR of the model's given ordering: {corpus_mrr(ds.queries, fn):.4f}")
    bundle = evaluate(ds.queries, fn)
    print("Full metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in bundle.items()))

    # tau sweep (plan §8.1: report match-function sensitivity)
    print("\ntau sensitivity (MRR vs threshold):")
    for tau in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        ds_t, _ = load_real_dataset(args.data, tau=tau)
        print(f"   tau={tau:.2f}  MRR={corpus_mrr(ds_t.queries, fn):.4f}  "
              f"(papers w/ match: {sum(1 for q in ds_t.queries if q.has_ground_truth())})")


if __name__ == "__main__":
    main()
