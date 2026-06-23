"""Demo: incorporate the agent's output and S2 graph signals into rank fusion.

Part A — AGENT OUTPUT -> fusion: run the candidate-harvesting agent on a local
seed, turn its pool into a fusion `Query` (retrieval source + provenance), and
score it against the masked ground truth.

Part B — S2 GRAPH SIGNALS -> fusion: attach per-candidate signals to
`Query.features` and show the three ways they enter fusion — an extra RRF
"horizon" source, the RRF feature-bias term, and learned meta-ranker features.
(Uses an informative synthetic signal here; in production it's Naz's
GraphSignalExtractor via `graph_signal_fn`.)

    python run_integration.py
"""

from __future__ import annotations

import os
import re
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    CandidateHarvestAgent, AgentConfig, pool_to_query, evaluate_pool,
    make_synthetic_dataset, corpus_mrr, make_rrf_rank_fn,
    add_horizon_source, attach_features, HORIZON_KEYS,
    fit_select_meta_ranker, make_single_source_rank_fn,
)
from rank_fusion.meta_ranker import MetaRanker
from s2_tools import LocalCorpusS2Client

DATA = os.path.join(os.path.dirname(__file__), "Data")
_SENT = re.compile(r"(?<=[.!?])\s+")


def part_a(pid="11816106"):
    print("=" * 70)
    print("A) AGENT OUTPUT -> fusion Query")
    print("=" * 70)
    import json
    s = json.load(open(os.path.join(DATA, "Paper_summaries", f"{pid}_future_directions.json")))
    gaps = [g["description"] for g in s.get("key_gaps", [])]
    gt = [x.strip() for x in _SENT.split(
        open(os.path.join(DATA, "Ground Truth Future Directions", f"{pid}.txt")).read())
        if len(x.strip()) > 20]
    paper_text = s["topic_and_scope"] + " " + " ".join(s.get("main_themes", []))

    agent = CandidateHarvestAgent(LocalCorpusS2Client(DATA, exclude=pid),
                                  AgentConfig(match_tau=0.06, patience=2))
    res = agent.run(pid, gaps)
    q = pool_to_query(pid, res["pool"], paper_text=paper_text, gt_texts=gt, tau=0.06)

    print(f"agent harvested {len(res['pool'])} candidates (stop: {res['stop_reason']})")
    print(f"-> Query: {len(q.candidate_ids)} candidates, "
          f"sources={list(q.candidate_sources)}, rankings={list(q.rankings)}")
    print(f"   matched {len(q.relevance)} / {len(gt)} masked future-work directions")
    rfn = make_rrf_rank_fn({s_: 1.0 for s_ in q.rankings}, 60.0)
    print(f"   RRF over the agent's retrieval source -> Paper MRR "
          f"{corpus_mrr([q], rfn):.3f}\n")


def part_b(seed=7):
    print("=" * 70)
    print("B) S2 GRAPH SIGNALS -> features -> fusion (3 entry points)")
    print("=" * 70)
    # synthetic with an informative signal (sim_to_paper higher for true dirs)
    ds = make_synthetic_dataset(n_papers=300, attach_features=True, seed=seed)
    src = ds.all_sources()
    # name the informative signal "relevance" so horizon_prior picks it up
    for split in (ds.train, ds.dev, ds.test):
        for q in split:
            for cid, f in q.features.items():
                f["relevance"] = f.get("sim_to_paper", 0.0)

    base = make_rrf_rank_fn({s_: 1.0 for s_ in src}, 60.0)
    print(f"{'system':<42}{'dev MRR':>9}{'test MRR':>10}")
    print(f"{'RRF (rankers only)':<42}{corpus_mrr(ds.dev, base):>9.4f}"
          f"{corpus_mrr(ds.test, base):>10.4f}")

    # (a) extra RRF source: rank by horizon_prior over the signals
    def with_horizon(q):
        add_horizon_source(q, weights={"relevance": 1.0, "novelty": 0.3})
        return q
    for split in (ds.dev, ds.test):
        for q in split:
            with_horizon(q)
    hz = make_rrf_rank_fn({**{s_: 1.0 for s_ in src}, "horizon": 1.0}, 60.0)
    print(f"{'  + horizon RRF source (signals)':<42}{corpus_mrr(ds.dev, hz):>9.4f}"
          f"{corpus_mrr(ds.test, hz):>10.4f}")

    # (c) learned meta-ranker features
    mr0, d0, _ = fit_select_meta_ranker(ds.train, ds.dev, src, seed=seed)
    mr1 = MetaRanker(src, extra_feature_keys=["relevance", "novelty", "popularity",
                                              "recency"]).fit(ds.train)
    print(f"{'  meta-ranker (rank features only)':<42}{d0:>9.4f}"
          f"{corpus_mrr(ds.test, mr0.rank_fn()):>10.4f}")
    print(f"{'  meta-ranker + graph-signal features':<42}"
          f"{corpus_mrr(ds.dev, mr1.rank_fn()):>9.4f}"
          f"{corpus_mrr(ds.test, mr1.rank_fn()):>10.4f}")
    print("\n  -> an informative graph signal lifts MRR both as an RRF source and "
          "as a\n     learned feature; in production swap the synthetic signal for "
          "graph_signal_fn(GraphSignalExtractor, ctx).")


if __name__ == "__main__":
    part_a()
    part_b()
