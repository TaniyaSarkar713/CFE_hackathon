"""Single-paper test: wire Naz's Agent C (S2 knowledge graph) into the fusion.

Three sources for one paper:
  llm1  = the LLM summary's future_directions      (agent_a)
  llm2  = the LLM summary's key_gaps + main_themes  (agent_b)
  s2    = Naz's S2DirectionAgent, run LIVE on the citation graph  (agent_c)

They're clustered across sources (`build_consensus_query`), fused with RRF, and
scored against the masked ground-truth future directions.

    python run_s2_fusion.py [PMCID]        # default 11816194

Needs network + an S2 API key for the live Agent C; if S2 is rate-limited the
test still runs on llm1+llm2 and reports that Agent C returned nothing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="   [s2] %(message)s")

from rank_fusion import (
    build_consensus_query, cluster_sources, s2_agent_source,
    make_rrf_rank_fn, paper_mrr,
)

DATA = os.path.join(os.path.dirname(__file__), "Data")
S2_KEY = os.environ.get("S2_API_KEY", "s2k-VqZPfeS1kZk3Mqd5ac7i1PKN0rv15d8wrKm7LNUR")


def llm_sources(pid):
    s = json.load(open(os.path.join(DATA, "Paper_summaries", f"{pid}_future_directions.json")))
    llm1 = [f"{d['title']}. {d.get('rationale','')}" for d in s.get("future_directions", [])]
    llm2 = [g["description"] for g in s.get("key_gaps", [])] + s.get("main_themes", [])
    title = open(os.path.join(DATA, "papers", f"{pid}.txt")).readline().strip()
    return llm1, llm2, title


def ground_truth(pid):
    f = os.path.join(DATA, "Ground Truth Entities", "future_sec_NER_ground_truth",
                     f"{pid}_FD_filtered_entities.json")
    if os.path.exists(f):
        return [d["direction"] for d in json.load(open(f)).get("future_directions", [])]
    return []


def run_agent_c(pid, title):
    """Run Naz's Agent C live; return its ranked text list (or [] if S2 fails)."""
    try:
        from semantic_scholar_ranking.s2_client import S2Client, GraphSignalExtractor
        from semantic_scholar_ranking.s2_agent import S2DirectionAgent
        client = S2Client(api_key=S2_KEY, rps=0.5)
        ext = GraphSignalExtractor(client)
        real_id = client.resolve_corpus_id(title) or pid
        ctx = ext.build_context(real_id)
        agent = S2DirectionAgent(ext, max_hops=1, max_per_hop=8, min_candidates=4)
        out = agent.run(real_id, ctx)
        return s2_agent_source(out)
    except Exception as e:
        print(f"   [s2] Agent C failed: {type(e).__name__}: {e}")
        return []


def main():
    pid = sys.argv[1] if len(sys.argv) > 1 else "11816194"
    llm1, llm2, title = llm_sources(pid)
    gt = ground_truth(pid)
    print(f"SEED {pid}: {title[:90]}")
    print(f"sources: llm1(future_dirs)={len(llm1)}  llm2(gaps+themes)={len(llm2)}  "
          f"| ground-truth directions={len(gt)}")

    print("\n--- running Agent C live on the S2 citation graph ---")
    s2 = run_agent_c(pid, title)
    print(f"Agent C (s2) returned {len(s2)} ranked candidates"
          + (f"; e.g. '{s2[0][:60]}'" if s2 else " (S2 rate-limited/empty)"))

    sources = {"llm1": llm1, "llm2": llm2}
    if s2:
        sources["s2"] = s2

    # cluster across sources + build the fusion Query (relevance vs masked GT)
    _cids, _rk, ctext = cluster_sources(sources, threshold=0.5)
    q = build_consensus_query(pid, sources, gt_texts=gt, threshold=0.5, tau=0.10)
    print(f"\nClustered {sum(len(v) for v in sources.values())} candidates -> "
          f"{len(q.candidate_ids)} cross-source clusters; "
          f"matched {len(q.relevance)}/{len(gt)} GT directions")

    fused = make_rrf_rank_fn({s: 1.0 for s in q.rankings}, 60.0)(q)
    rel = q.relevant_ids
    print("\nFused top-5 (RRF over llm1 + llm2 + s2):")
    for rank, cid in enumerate(fused[:5], 1):
        srcs = [s for s in q.rankings if cid in q.rankings[s]]
        hit = " ✓GT" if cid in rel else ""
        print(f"  {rank}. [{','.join(srcs)}]{hit}  {ctext.get(cid,'')[:66]}")

    print(f"\nPaper MRR (fused vs masked future-work) = "
          f"{paper_mrr(fused, q.relevance):.3f}")


if __name__ == "__main__":
    main()
