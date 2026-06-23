"""run.py — end-to-end rank fusion pipeline for CFE Hackathon.

What this does
--------------
1. Load all papers from Data/Paper_summaries into Query objects
2. Optionally fetch S2 graph signals (graph_relevance, graph_novelty,
   graph_recency, citation_velocity) for each candidate direction
3. Evaluate baselines:  Agent A alone, Agent B alone, uniform RRF
4. Tune RRF weights + beta (S2 signal weights) on dev set toward MRR
5. Evaluate tuned RRF on test set
6. Print ablation table + save results.json

Usage
-----
  # offline — no S2 calls, fast
  python run.py

  # with S2 graph signals
  python run.py --api-key YOUR_KEY

  # single paper
  python run.py --paper-id 11812686

  # tune weights and evaluate
  python run.py --api-key YOUR_KEY --tune

  # save output
  python run.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# make `ranking/` importable when running as a script from its parent folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking.data_loader  import load_dataset, DEFAULT_DATA_DIR
from ranking.fusion       import (make_rrf_rank_fn, make_single_source_rank_fn,
                                   feature_bias_term, weighted_rrf)
from ranking.metrics      import evaluate, corpus_mrr
from ranking.optimize     import tune_rrf
from ranking.s2_client    import S2Client

S2_SIGNALS = ["graph_relevance", "graph_novelty", "graph_recency", "citation_velocity"]


# ===========================================================================
# Pretty printing
# ===========================================================================

def print_table(rows, headers):
    widths = [max(len(h), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


def print_ranked(paper_id, query, ranked_ids, scores=None, top_k=5):
    print(f"\n{'='*70}")
    print(f"Paper: {paper_id}")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Src':<8} {'Score':>7}  {'Rel':>4} {'Nov':>4} {'Vel':>4}  Title")
    print(f"{'-'*70}")
    for i, cid in enumerate(ranked_ids[:top_k], 1):
        feats = query.features.get(cid, {})
        sc    = f"{scores[cid]:.4f}" if scores and cid in scores else "  —  "
        # reconstruct title from candidate id suffix
        src   = "A" if "_A" in cid else "B"
        print(
            f"{i:<4} {src:<8} {sc:>7}"
            f"  {feats.get('graph_relevance',0):>4.2f}"
            f" {feats.get('graph_novelty',0):>4.2f}"
            f" {feats.get('citation_velocity',0):>4.2f}"
            f"  {cid}"
        )
    print(f"{'='*70}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="CFE Hackathon — Rank Fusion Pipeline")
    p.add_argument("--data-dir",  default=DEFAULT_DATA_DIR,
                   help=f"Path to Data/ folder (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--paper-id",  default=None,
                   help="Run on a single paper only")
    p.add_argument("--api-key",   default=None,
                   help="Semantic Scholar API key (omit for offline mode)")
    p.add_argument("--tau",       type=float, default=0.15,
                   help="TF-IDF cosine threshold for ground-truth matching (default 0.15)")
    p.add_argument("--tune",      action="store_true",
                   help="Tune RRF weights toward dev MRR")
    p.add_argument("--k",         type=float, default=60.0,
                   help="RRF smoothing constant (default 60)")
    p.add_argument("--output",    default=None,
                   help="Save full results to this JSON file")
    p.add_argument("--top-k",     type=int, default=5,
                   help="Number of directions to display per paper (default 5)")
    args = p.parse_args()

    # --- load data ----------------------------------------------------------
    s2 = S2Client(api_key=args.api_key) if args.api_key else None

    logger.info("Loading dataset from %s ...", args.data_dir)
    dataset, diag = load_dataset(
        data_dir  = args.data_dir,
        tau       = args.tau,
        s2_client = s2,
        paper_ids = [args.paper_id] if args.paper_id else None,
    )

    if not dataset.queries:
        print("No papers loaded. Check --data-dir path.")
        return

    has_s2  = any(q.features for q in dataset.queries)
    has_gt  = any(q.has_ground_truth() for q in dataset.queries)
    sources = ["agent_a", "agent_b"]

    print(f"\nCFE Hackathon — Rank Fusion")
    print(f"Papers loaded : {len(dataset.queries)}")
    print(f"Ground truth  : {'YES (TF-IDF proxy)' if has_gt else 'NO'}")
    print(f"S2 signals    : {'YES' if has_s2 else 'NO (offline mode)'}")

    # --- per-paper ranked output -------------------------------------------
    uniform_rrf = make_rrf_rank_fn(weights={s: 1.0 for s in sources}, k=args.k)
    all_output  = []

    for q in dataset.queries:
        ranked, scores = weighted_rrf(
            q.rankings,
            weights       = {s: 1.0 for s in sources},
            k             = args.k,
            candidate_ids = q.candidate_ids,
            feature_bias  = (feature_bias_term(q,
                              {s: 0.05 for s in S2_SIGNALS})
                             if has_s2 else None),
        )
        print_ranked(q.query_id, q, ranked, scores, top_k=args.top_k)
        all_output.append({
            "paper_id":     q.query_id,
            "pool_size":    len(q.candidate_ids),
            "gt_matched":   len(q.relevance),
            "ranked":       [{"rank": i+1, "id": cid,
                              "rrf_score": round(scores.get(cid, 0), 6),
                              "s2_signals": q.features.get(cid, {})}
                             for i, cid in enumerate(ranked[:10])],
        })

    # --- MRR evaluation (only if ground truth available) -------------------
    if has_gt:
        print("\n\nMRR Evaluation")
        dev_q  = dataset.dev
        test_q = dataset.test

        # baselines
        a_fn   = make_single_source_rank_fn("agent_a")
        b_fn   = make_single_source_rank_fn("agent_b")
        uni_fn = make_rrf_rank_fn({s: 1.0 for s in sources}, args.k)

        rows = [
            ("Agent A only",   f"{corpus_mrr(test_q, a_fn):.4f}"),
            ("Agent B only",   f"{corpus_mrr(test_q, b_fn):.4f}"),
            ("Uniform RRF",    f"{corpus_mrr(test_q, uni_fn):.4f}"),
        ]

        if has_s2:
            s2_fn = make_rrf_rank_fn(
                {s: 1.0 for s in sources}, args.k,
                beta={s: 0.05 for s in S2_SIGNALS}
            )
            rows.append(("RRF + S2 signals", f"{corpus_mrr(test_q, s2_fn):.4f}"))

        # tuned (if requested)
        best_cfg = None
        if args.tune and dev_q:
            logger.info("Tuning RRF weights on dev set (%d papers) ...", len(dev_q))
            best_cfg = tune_rrf(
                dev_q,
                sources      = sources,
                beta_signals = S2_SIGNALS if has_s2 else [],
                verbose      = True,
            )
            tuned_mrr = corpus_mrr(test_q, best_cfg.rank_fn())
            rows.append((f"Tuned RRF (dev MRR={best_cfg.dev_mrr:.4f})",
                         f"{tuned_mrr:.4f}"))
            print(f"\nBest weights : {best_cfg.weights}")
            print(f"Best beta    : {best_cfg.beta}")
            print(f"Best k       : {best_cfg.k}")

        print_table(rows, ["System", "Test MRR"])

        # full metric bundle for best system
        best_fn = best_cfg.rank_fn() if best_cfg else uni_fn
        bundle  = evaluate(test_q, best_fn)
        print("\nFull metrics (best system):")
        for m, v in bundle.items():
            print(f"  {m:<15} {v:.4f}")

    # --- save ---------------------------------------------------------------
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_output, f, indent=2)
        print(f"\nSaved -> {args.output}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
