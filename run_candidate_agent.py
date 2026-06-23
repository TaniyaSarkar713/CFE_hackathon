"""Demo: the autonomous candidate-harvesting agent.

Runs the agent on a seed paper (offline, over the local Data/ corpus via
LocalCorpusS2Client — swap in EnrichedS2Client for the live Semantic Scholar
API). The agent:
  - reads neighbours' keywords + future directions through the S2 client,
  - adds non-duplicate candidates to the recommender pool,
  - stops on its own when coverage of the paper's open gaps stops improving.

Then it evaluates the harvested pool against the MASKED ground-truth future-work
(recall / oracle-MRR) and compares the agentic loop to a fixed one-shot baseline.

    python run_candidate_agent.py [PMCID]
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import CandidateHarvestAgent, AgentConfig, evaluate_pool
from s2_tools import LocalCorpusS2Client

DATA = os.path.join(os.path.dirname(__file__), "Data")
_SENT = re.compile(r"(?<=[.!?])\s+")


def load_seed(pid):
    s = json.load(open(os.path.join(DATA, "Paper_summaries", f"{pid}_future_directions.json")))
    gaps = [g["description"] for g in s.get("key_gaps", [])]
    gt_path = os.path.join(DATA, "Ground Truth Future Directions", f"{pid}.txt")
    gt = []
    if os.path.exists(gt_path):
        txt = open(gt_path, encoding="utf-8").read()
        gt = [x.strip() for x in _SENT.split(txt) if len(x.strip()) > 20]
    return s, gaps, gt


# TF-IDF (the offline matcher) needs a lower threshold than SPECTER2 would.
LOCAL_TAU = 0.06


def run_one(pid, verbose=True):
    seed, gaps, gt = load_seed(pid)
    client = LocalCorpusS2Client(DATA, exclude=pid)   # exclude the seed itself
    agent = CandidateHarvestAgent(
        client, AgentConfig(max_rounds=10, patience=2, match_tau=LOCAL_TAU))
    result = agent.run(pid, gaps)

    if verbose:
        print(f"SEED {pid}: {seed['topic_and_scope'][:90]}...")
        print(f"open gaps ({len(gaps)}):")
        for g in gaps:
            print(f"   - {g[:88]}")
        print("\nAgent trace (tool-loop):")
        print(f"   {'rnd':>3}  {'action':<22}{'nbrs':>5}{'added':>6}  {'cover':>6}{'gain':>6}{'calls':>6}")
        for t in result["trace"]:
            cov = f"{t['covered']}/{t['total']}"
            print(f"   {t['round']:>3}  {t['action']:<22}{t['neighbours']:>5}"
                  f"{t['added']:>6}  {cov:>6}{t['gain']:>+6}{t['calls']:>6}")
        print(f"   -> STOP: {result['stop_reason']}  "
              f"(pool={len(result['pool'])}, S2 calls={result['s2_calls']})\n")

        print("Sample harvested candidates (with provenance + keywords):")
        for c in result["pool"][:5]:
            kw = ", ".join(c.keywords[:3])
            print(f"   • [{c.provenance}] {c.text[:80]}")
            if kw:
                print(f"       keywords: {kw}")
    return result, gt


def main():
    pid = sys.argv[1] if len(sys.argv) > 1 else "11816106"
    _, gaps, gt = load_seed(pid)
    result, gt = run_one(pid)

    print("\n" + "=" * 68)
    print("EVAL — harvested pool vs MASKED ground-truth future-work")
    print("=" * 68)
    if not gt:
        print("(no ground-truth future-work file for this paper)")
        return
    agentic = evaluate_pool(result["pool"], gt, tau=LOCAL_TAU)
    # fixed baseline: a single expansion round (no adaptive loop, no stop logic)
    base = CandidateHarvestAgent(
        LocalCorpusS2Client(DATA, exclude=pid),
        AgentConfig(max_rounds=1, patience=99, match_tau=LOCAL_TAU)).run(pid, gaps)
    base_eval = evaluate_pool(base["pool"], gt, tau=LOCAL_TAU)

    print(f"   {'system':<26}{'#cand':>7}{'GT recall':>11}{'oracle-MRR':>12}")
    print(f"   {'fixed one-shot expand':<26}{base_eval['n_candidates']:>7}"
          f"{base_eval['recall']:>11.3f}{base_eval['oracle_mrr']:>12.3f}")
    print(f"   {'autonomous agent':<26}{agentic['n_candidates']:>7}"
          f"{agentic['recall']:>11.3f}{agentic['oracle_mrr']:>12.3f}")
    print(f"\n   covered {agentic.get('n_covered_gt', 0)}/{agentic['n_gt']} "
          f"masked future-work directions; stop reason: {result['stop_reason']}.")


if __name__ == "__main__":
    main()
