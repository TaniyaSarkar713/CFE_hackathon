"""Complete ResearchHorizon pipeline (end to end).

  STAGE 1  LLM agents        — rank the N LLM agents by MRR, keep the top 2
  STAGE 2  Semantic Scholar  — per paper, build an S2 knowledge-graph source
                               (white-space directions from the neighbourhood)
  STAGE 3  Rank fusion       — cluster the 3 sources (top-2 LLMs + S2), tune
                               weighted RRF on dev MRR, wrap in the best-of selector
  STAGE 4  Evaluation        — ablation table on a held-out test split: each
                               source alone vs uniform/tuned RRF vs the selected
                               strategy, scored against the masked ground truth

Reproducible/offline by default (S2 source = white-space over the local
neighbourhood; live S2/Agent C is rate-limited). Metric = first-relevant MRR.

    python run_pipeline.py [--papers 150] [--tau 0.10] [--neighbours 30]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    Query, match_relevance, select_top_sources, build_consensus_query,
    tune_consensus, corpus_mrr_at_k, make_rrf_rank_fn, make_single_source_rank_fn,
    compare_strategies, select_best_strategy_cv,
    whitespace_candidates, whitespace_source, keyphrases,
    mlrc_benchmark, format_benchmark,
)

DATA = os.path.join(os.path.dirname(__file__), "Data")
SUMM = os.path.join(DATA, "Paper_summaries")
GT_DIR = os.path.join(DATA, "Ground Truth Entities", "future_sec_NER_ground_truth")
PD = True                        # per-direction MRR (credit multiple true dirs)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_gt():
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
    return [d for d in sorted(glob.glob(os.path.join(SUMM, "*_fd")))
            if os.path.isdir(d) and " " not in os.path.basename(d)]


def pids_in(folder):
    return {os.path.basename(f).replace("_future_directions.json", "")
            for f in glob.glob(os.path.join(folder, "*_future_directions.json"))}


def read_summary(folder, pid):
    fp = os.path.join(folder, f"{pid}_future_directions.json")
    if not os.path.exists(fp):
        return None
    try:
        return json.load(open(fp))
    except Exception:
        return None


def directions_of(summary):
    return [f"{d.get('title','')}. {d.get('rationale','')}"
            for d in summary.get("future_directions", [])]


# --------------------------------------------------------------------------- #
# Stage 2 helper: S2 white-space source from the (offline) neighbourhood
# --------------------------------------------------------------------------- #
class Neighbourhood:
    """TF-IDF index over papers' topic+themes -> top-k similar papers per seed."""
    def __init__(self, folder, pids):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids = sorted(pids)
        self.blob = {}
        for p in self.ids:
            s = read_summary(folder, p) or {}
            self.blob[p] = s.get("topic_and_scope", "") + " " + " ".join(s.get("main_themes", []))
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform([self.blob[i] for i in self.ids])
        self.pos = {i: j for j, i in enumerate(self.ids)}

    def docs(self, pid, k):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.mat[self.pos[pid]], self.mat)[0]
        order = [self.ids[j] for j in sims.argsort()[::-1] if self.ids[j] != pid][:k]
        return [self.blob[n] for n in order]


def s2_whitespace_source(pid, summary, neigh, k):
    seed_ents = keyphrases(summary.get("topic_and_scope", "") + " "
                           + " ".join(summary.get("main_themes", [])))
    gap_ents = keyphrases(" ".join(g["description"] for g in summary.get("key_gaps", [])))
    ws = whitespace_candidates(seed_ents, neigh.docs(pid, k), gap_entities=gap_ents,
                               min_support=2, top_k=15)
    return whitespace_source(ws)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=150)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--neighbours", type=int, default=30)
    ap.add_argument("--mrr-k", type=int, default=3, help="tune/eval on MRR@K (top-K dirs)")
    args = ap.parse_args()
    K = args.mrr_k

    gt = load_gt()
    folders = llm_folders()
    names = [os.path.basename(f) for f in folders]
    fold = dict(zip(names, folders))

    # ---- STAGE 1: rank LLM agents, keep top 2 ---------------------------- #
    print("=" * 64)
    print("STAGE 1 — LLM AGENTS: rank by MRR, keep top 2")
    print("=" * 64)
    common = sorted(set.intersection(*[pids_in(f) for f in folders]) & set(gt))
    src_q = {}
    for n in names:
        qs = []
        for pid in common:
            s = read_summary(fold[n], pid)
            if not s or not s.get("future_directions"):
                continue
            texts = directions_of(s)
            cids = [f"{pid}_d{i}" for i in range(len(texts))]
            rel = match_relevance(cids, texts, gt[pid], tau=args.tau)
            qs.append(Query(pid, cids, {}, rel))
        src_q[n] = qs
    sel = select_top_sources(src_q, k=2, mrr_k=K, per_direction=PD)
    for r in sel["ranked"]:
        mark = "  <- TOP 2" if r["source"] in sel["top_k"] else ""
        print(f"   {r['source']:<14} MRR={r['mrr']:.4f}{mark}")
    top2 = sel["top_k"]
    print(f"\n   selected LLM rankers: llm1={top2[0]}, llm2={top2[1]}")

    # ---- STAGE 2: build 3-source Queries (top-2 LLMs + S2 white-space) --- #
    print("\n" + "=" * 64)
    print("STAGE 2 — SEMANTIC SCHOLAR: white-space S2 source per paper")
    print("=" * 64)
    papers = common[:args.papers]
    neigh = Neighbourhood(fold[top2[0]], set(common))
    queries = []
    for pid in papers:
        s1 = read_summary(fold[top2[0]], pid)
        s2 = read_summary(fold[top2[1]], pid)
        if not (s1 and s2 and s1.get("future_directions") and s2.get("future_directions")):
            continue
        sources = {
            "llm1": directions_of(s1),
            "llm2": directions_of(s2),
            "s2": s2_whitespace_source(pid, s1, neigh, args.neighbours),
        }
        q = build_consensus_query(pid, sources, gt_texts=gt[pid],
                                  tau=args.tau, threshold=0.5)
        queries.append(q)
    print(f"   built {len(queries)} 3-source queries "
          f"(llm1 + llm2 + S2 white-space), clustered + matched to GT")

    # ---- STAGE 3: consensus RRF + CV best-of fallback ------------------- #
    print("\n" + "=" * 64)
    print(f"STAGE 3 — RANK FUSION: consensus RRF (consensus.py, MRR@{K}) "
          f"+ CV best-of fallback")
    print("=" * 64)
    cut = len(queries) // 2
    dev, test = queries[:cut], queries[cut:]
    srcs = ["llm1", "llm2", "s2"]

    def consensus_builder(train):
        return tune_consensus(train, llm_sources=("llm1", "llm2"), s2_source="s2",
                              mrr_k=K, per_direction=PD, n_random=120,
                              coord_rounds=2, restarts=1).rank_fn()

    def uniform_builder(train):
        return make_rrf_rank_fn({s: 1.0 for s in srcs}, 60.0)

    fused_builders = {"uniform RRF": uniform_builder, "consensus RRF": consensus_builder}
    # show what consensus learns on the full dev set
    cfg = tune_consensus(dev, llm_sources=("llm1", "llm2"), s2_source="s2",
                         mrr_k=K, per_direction=PD, n_random=200, coord_rounds=3, restarts=2)
    print(f"   consensus (full dev): w=" +
          ", ".join(f"{s}={cfg.weights[s]:.2f}" for s in srcs) +
          f", k={cfg.k:g}, eta1={cfg.eta1} eta3={cfg.eta3} rho={cfg.rho}")

    # CV selection: re-fit each fusion per fold; prefer the simplest strategy
    # within `margin` of the best cross-validated MRR -> robust to dev/test gap.
    chosen = select_best_strategy_cv(dev, srcs, fused_builders=fused_builders,
                                     mrr_k=K, per_direction=PD, n_folds=5, margin=0.01)
    print(f"\n   5-fold CV MRR@{K} on dev:")
    for r in chosen["table"]:
        mark = "  <- chosen" if r["name"] == chosen["best_name"] else ""
        print(f"      {r['name']:<22}{r['cv_mrr']:>8.4f}{mark}")
    print(f"   fallback emits: {chosen['best_name']}  "
          f"(simplest within {0.01:.2f} of best CV MRR)")

    # ---- STAGE 4: ablation table on TEST -------------------------------- #
    print("\n" + "=" * 64)
    print(f"STAGE 4 — EVALUATION on TEST ({len(test)} papers), MRR@{K}")
    print("=" * 64)
    fused_full = {n: b(dev) for n, b in fused_builders.items()}
    rows = compare_strategies(test, srcs, fused_rank_fns=fused_full, mrr_k=K, per_direction=PD)
    best_single = max(r["mrr"] for r in rows if r["name"].startswith("single:"))
    print(f"   {'system':<22}{'test MRR':>10}")
    print("   " + "-" * 32)
    for r in rows:
        tag = "  *fusion" if "RRF" in r["name"] else ""
        print(f"   {r['name']:<22}{r['mrr']:>10.4f}{tag}")
    chosen_mrr = corpus_mrr_at_k(test, chosen["best_rank_fn"], K, PD)
    print("   " + "-" * 32)
    print(f"   {'SELECTED (CV best-of)':<22}{chosen_mrr:>10.4f}   <- emitted "
          f"({chosen['best_name']})")
    print(f"\n   emitted vs best single source on test: "
          f"{chosen_mrr - best_single:+.4f}  (CV picks the generaliser)")

    # ---- MLRC-style benchmark: improvement over the random baseline ----- #
    print("\n   MLRC benchmark (improvement over random baseline):")
    methods = {f"single:{s}": make_single_source_rank_fn(s) for s in srcs}
    methods.update(fused_full)
    methods[f"SELECTED:{chosen['best_name']}"] = chosen["best_rank_fn"]
    report = mlrc_benchmark(test, methods, mrr_k=K, per_direction=PD)
    print(format_benchmark(report))


if __name__ == "__main__":
    main()
