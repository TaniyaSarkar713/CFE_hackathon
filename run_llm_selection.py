"""LLM-selection agent: rank the N LLM agents by MRR, keep the top 2.

Each `Data/Paper_summaries/<llm>_fd/` folder holds one LLM agent's ranked
`future_directions` per paper. This agent:
  1. matches every LLM's ranked directions against the masked ground-truth
     future directions (structured GT),
  2. computes each LLM's corpus MRR on the SAME set of papers (fair),
  3. selects the top-2 LLMs — the strongest two rankers to feed the fusion.

    python run_llm_selection.py [--tau 0.10] [--limit N] [--mrr-k K]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import Query, match_relevance, select_top_sources

DATA = os.path.join(os.path.dirname(__file__), "Data")
SUMM = os.path.join(DATA, "Paper_summaries")
GT_DIR = os.path.join(DATA, "Ground Truth Entities", "future_sec_NER_ground_truth")


def load_gt():
    """pid -> [ground-truth future-direction statements] (the masked answer)."""
    gt = {}
    for f in glob.glob(os.path.join(GT_DIR, "*_FD_filtered_entities.json")):
        pid = os.path.basename(f).split("_")[0]
        try:
            dirs = json.load(open(f)).get("future_directions", [])
        except Exception:
            continue
        texts = [d.get("direction", "") for d in dirs if d.get("direction")]
        if texts:
            gt[pid] = texts
    return gt


def llm_folders():
    """The <llm>_fd directories (skips the stray 'gpt5nano_fd 2' copy)."""
    out = []
    for d in sorted(glob.glob(os.path.join(SUMM, "*_fd"))):
        if os.path.isdir(d) and " " not in os.path.basename(d):
            out.append(d)
    return out


def pids_in(folder):
    return {os.path.basename(f).replace("_future_directions.json", "")
            for f in glob.glob(os.path.join(folder, "*_future_directions.json"))}


def build_queries(folder, common, gt, tau):
    """One Query per paper: candidate_ids in the LLM's ranked (id) order;
    relevance = which of its directions match the GT (TF-IDF >= tau)."""
    queries = []
    for pid in common:
        fp = os.path.join(folder, f"{pid}_future_directions.json")
        if not os.path.exists(fp):
            continue
        try:
            dirs = json.load(open(fp)).get("future_directions", [])
        except Exception:
            continue
        if not dirs:
            continue
        cand_ids = [f"{pid}_d{i}" for i in range(len(dirs))]      # id order = the ranking
        cand_texts = [f"{d.get('title','')}. {d.get('rationale','')}" for d in dirs]
        rel = match_relevance(cand_ids, cand_texts, gt[pid], tau=tau)
        queries.append(Query(pid, cand_ids, {}, rel))
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.10, help="TF-IDF match threshold")
    ap.add_argument("--limit", type=int, default=0, help="cap #papers (0 = all common)")
    ap.add_argument("--mrr-k", type=int, default=0, help="MRR@K (0 = unbounded MRR)")
    args = ap.parse_args()
    mrr_k = args.mrr_k or None

    gt = load_gt()
    folders = llm_folders()
    names = [os.path.basename(f) for f in folders]
    pidsets = {n: pids_in(f) for n, f in zip(names, folders)}
    common = sorted(set.intersection(*pidsets.values()) & set(gt))
    if args.limit:
        common = common[:args.limit]
    print(f"{len(folders)} LLM agents: {names}")
    print(f"{len(common)} papers common to all agents AND with ground truth "
          f"(tau={args.tau}, MRR{'@'+str(mrr_k) if mrr_k else ''})\n")

    src_to_q = {}
    for n, f in zip(names, folders):
        t = time.time()
        src_to_q[n] = build_queries(f, common, gt, args.tau)
        print(f"  scored {n:<14} ({time.time()-t:.0f}s)")

    res = select_top_sources(src_to_q, k=2, mrr_k=mrr_k)
    print("\n" + "=" * 46)
    print(f"  {'LLM agent':<16}{'MRR':>8}{'#papers':>9}")
    print("  " + "-" * 42)
    for r in res["ranked"]:
        mark = "  <- TOP 2" if r["source"] in res["top_k"] else ""
        print(f"  {r['source']:<16}{r['mrr']:>8.4f}{r['n_papers']:>9}{mark}")
    print("=" * 46)
    print(f"\nSelected top-2 LLM rankers for fusion: {res['top_k']}")


if __name__ == "__main__":
    main()
