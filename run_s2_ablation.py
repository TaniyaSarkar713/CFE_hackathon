"""Ablation: title/topology ranking vs entity-gap re-ranking of S2 candidates.

For every paper with ground truth, we take the same S2-neighbour candidate pool
(other local papers used as the citation neighbourhood, since live S2 is
rate-limited) and compare two ORDERINGS of it, scored by the same match function
against the masked future-work directions:

  A) title / topology  — neighbours ordered by similarity to the seed (what
                          Agent C's graph signals approximate)
  B) entity-gap rerank — neighbours re-ordered by overlap with the seed's
                          open-gap entities  (rank_fusion.entity_rerank)

Only the ranking differs; the candidate pool and the relevance labels are
identical, so any MRR gap is purely from the entity re-ranking. We also report
the white-space generator as a separate candidate source.

    python run_s2_ablation.py [--k 20] [--tau 0.10]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    match_relevance, paper_mrr, entity_rerank_texts, whitespace_candidates,
    whitespace_source, keyphrases,
)

DATA = os.path.join(os.path.dirname(__file__), "Data")
_SENT = re.compile(r"(?<=[.!?])\s+")


def load_corpus():
    papers = {}
    for f in glob.glob(os.path.join(DATA, "Paper_summaries", "*_future_directions.json")):
        pid = os.path.basename(f).replace("_future_directions.json", "")
        s = json.load(open(f))
        s["_simtext"] = s.get("topic_and_scope", "") + " " + " ".join(s.get("main_themes", []))
        papers[pid] = s
    return papers


def ground_truth(pid):
    p = os.path.join(DATA, "Ground Truth Future Directions", f"{pid}.txt")
    if not os.path.exists(p):
        return []
    return [x.strip() for x in _SENT.split(open(p).read()) if len(x.strip()) > 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="neighbourhood size")
    ap.add_argument("--tau", type=float, default=0.06, help="match threshold")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    papers = load_corpus()
    ids = list(papers)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([papers[i]["_simtext"] for i in ids])
    pos = {i: k for k, i in enumerate(ids)}

    rows_a, rows_b, rows_w = [], [], []
    n_eval = 0
    for seed in ids:
        gt = ground_truth(seed)
        if not gt:
            continue
        # neighbourhood = top-k most similar OTHER papers (the S2-neighbour proxy)
        sims = cosine_similarity(mat[pos[seed]], mat)[0]
        order = [ids[j] for j in sims.argsort()[::-1] if ids[j] != seed][:args.k]
        # candidate pool = the neighbours' future-direction statements, in
        # neighbour-similarity order (the topology baseline ordering)
        cand_ids, cand_texts = [], []
        for n in order:
            for d in papers[n].get("future_directions", []):
                cand_ids.append(f"c{len(cand_ids)}")
                cand_texts.append(f"{d.get('title','')}. {d.get('rationale','')}")
        if not cand_ids:
            continue

        # relevance (identical for both rankings): which candidates match a GT dir
        relevance = match_relevance(cand_ids, cand_texts, gt, tau=args.tau)
        if not relevance:                       # no matchable candidate -> recall ceiling 0
            continue
        n_eval += 1

        gap_text = " ".join(g["description"] for g in papers[seed].get("key_gaps", []))
        covered = " ".join(papers[seed].get("main_findings", []))

        rank_a = cand_ids                       # A: similarity/topology order
        rank_b, _ = entity_rerank_texts(cand_ids, cand_texts, gap_text, covered)  # B: entity-gap
        rows_a.append(paper_mrr(rank_a, relevance))
        rows_b.append(paper_mrr(rank_b, relevance))

        # white-space generator as a separate candidate source
        seed_ents = keyphrases(papers[seed]["_simtext"])
        gap_ents = keyphrases(gap_text)
        neigh_docs = [papers[n]["_simtext"] for n in order]
        ws = whitespace_candidates(seed_ents, neigh_docs, gap_entities=gap_ents,
                                   min_support=2, top_k=args.k)
        ws_ids = [f"w{j}" for j in range(len(ws))]
        ws_texts = whitespace_source(ws)
        if ws_ids:
            ws_rel = match_relevance(ws_ids, ws_texts, gt, tau=args.tau)
            rows_w.append(paper_mrr(ws_ids, ws_rel) if ws_rel else 0.0)
        else:
            rows_w.append(0.0)

    def mean(x):
        return statistics.mean(x) if x else 0.0

    print(f"S2-candidate ranking ablation  (papers evaluated: {n_eval}, "
          f"k={args.k}, tau={args.tau})")
    print("  pool = neighbours' future-direction statements; same pool + match, "
          "only the ranking differs\n")
    print(f"  {'strategy':<36}{'mean MRR':>10}")
    print("  " + "-" * 46)
    print(f"  {'A) neighbour-similarity (topology)':<36}{mean(rows_a):>10.4f}")
    print(f"  {'B) entity-gap re-rank (NEW)':<36}{mean(rows_b):>10.4f}")
    print(f"  {'   white-space generator (NEW source)':<36}{mean(rows_w):>10.4f}")
    if rows_a:
        import numpy as np
        lift = mean(rows_b) - mean(rows_a)
        wins = sum(1 for a, b in zip(rows_a, rows_b) if b > a)
        loss = sum(1 for a, b in zip(rows_a, rows_b) if b < a)
        ties = n_eval - wins - loss
        diffs = np.array(rows_b) - np.array(rows_a)
        rng = np.random.default_rng(0)
        boot = [float(diffs[rng.integers(0, len(diffs), len(diffs))].mean())
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant (CI spans 0)"
        print(f"\n  entity-rerank vs topology: {lift:+.4f} mean  "
              f"(better {wins} / worse {loss} / tie {ties}  of {n_eval})")
        print(f"  95% bootstrap CI of the lift: [{lo:+.4f}, {hi:+.4f}]  -> {sig}")


if __name__ == "__main__":
    main()
