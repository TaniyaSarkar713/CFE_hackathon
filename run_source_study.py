"""Source study: which S2 source is better, and can fusion beat every single source?

Builds, per paper, ALL sources on a shared clustered candidate pool:
  - 5 LLM agents (their ranked future_directions)
  - s2_agentC   : neighbour papers ranked by similarity to the seed (Agent C's
                  topology signal), candidate text = neighbour topic  (offline proxy)
  - s2_white    : the white-space bridge generator

Then reports (test split, MRR@K):
  1. per-source MRR  -> is Agent C better than white-space?
  2. fusion experiments (RRF over subsets, incl. DIVERSE LLM pairs and +S2)
     -> can any fusion beat the best single source?

    python run_source_study.py [--papers 150] [--tau 0.10] [--mrr-k 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    Query, match_relevance, cluster_sources, make_single_source_rank_fn,
    make_rrf_rank_fn, corpus_mrr_at_k, tune_consensus,
    whitespace_candidates, whitespace_source, keyphrases,
)

DATA = os.path.join(os.path.dirname(__file__), "Data")
SUMM = os.path.join(DATA, "Paper_summaries")
GT_DIR = os.path.join(DATA, "Ground Truth Entities", "future_sec_NER_ground_truth")
PD = True


def load_gt():
    gt = {}
    for f in glob.glob(os.path.join(GT_DIR, "*_FD_filtered_entities.json")):
        pid = os.path.basename(f).split("_")[0]
        try:
            dirs = json.load(open(f)).get("future_directions", [])
        except Exception:
            continue
        t = [d.get("direction", "") for d in dirs if d.get("direction")]
        if t:
            gt[pid] = t
    return gt


def llm_folders():
    return [d for d in sorted(glob.glob(os.path.join(SUMM, "*_fd")))
            if os.path.isdir(d) and " " not in os.path.basename(d)]


def pids_in(folder):
    return {os.path.basename(f).replace("_future_directions.json", "")
            for f in glob.glob(os.path.join(folder, "*_future_directions.json"))}


def read(folder, pid):
    fp = os.path.join(folder, f"{pid}_future_directions.json")
    try:
        return json.load(open(fp)) if os.path.exists(fp) else None
    except Exception:
        return None


def directions(summary):
    return [f"{d.get('title','')}. {d.get('rationale','')}"
            for d in summary.get("future_directions", [])]


class Neigh:
    def __init__(self, folder, pids):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids = sorted(pids)
        self.blob = {}
        for p in self.ids:
            s = read(folder, p) or {}
            self.blob[p] = s.get("topic_and_scope", "") + " " + " ".join(s.get("main_themes", []))
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform([self.blob[i] for i in self.ids])
        self.pos = {i: j for j, i in enumerate(self.ids)}

    def order(self, pid, k):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.mat[self.pos[pid]], self.mat)[0]
        return [self.ids[j] for j in sims.argsort()[::-1] if self.ids[j] != pid][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=150)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--mrr-k", type=int, default=3)
    ap.add_argument("--neighbours", type=int, default=30)
    args = ap.parse_args()
    K = args.mrr_k

    gt = load_gt()
    folders = llm_folders()
    names = [os.path.basename(f).replace("_fd", "") for f in folders]
    fold = dict(zip(names, folders))
    common = sorted(set.intersection(*[pids_in(f) for f in folders]) & set(gt))[:args.papers]
    neigh = Neigh(fold[names[0]], set(common))
    print(f"{len(names)} LLMs {names}; {len(common)} papers; MRR@{K}\n")

    src_names = [f"llm:{n}" for n in names] + ["s2_agentC", "s2_white"]
    queries = []
    for pid in common:
        summ = {n: read(fold[n], pid) for n in names}
        if any(s is None or not s.get("future_directions") for s in summ.values()):
            continue
        source_texts = {f"llm:{n}": directions(summ[n]) for n in names}
        # Agent-C proxy: neighbour papers ranked by similarity to the seed
        nb = neigh.order(pid, args.neighbours)
        source_texts["s2_agentC"] = [neigh.blob[p] for p in nb]
        # white-space bridges
        seed_ents = keyphrases(neigh.blob[pid])
        gap_ents = keyphrases(" ".join(g["description"] for g in summ[names[0]].get("key_gaps", [])))
        ws = whitespace_candidates(seed_ents, [neigh.blob[p] for p in nb],
                                   gap_entities=gap_ents, min_support=2, top_k=15)
        source_texts["s2_white"] = whitespace_source(ws)

        cids, rankings, ctext = cluster_sources(source_texts, threshold=0.5)
        rel = match_relevance(cids, [ctext[c] for c in cids], gt[pid], tau=args.tau)
        queries.append(Query(pid, cids, rankings, rel))

    cut = len(queries) // 2
    dev, test = queries[:cut], queries[cut:]

    def mrr(fn, qs=test):
        return corpus_mrr_at_k(qs, fn, K, PD)

    # --- 1) per-source MRR ------------------------------------------------ #
    print("Per-source MRR@%d (test):" % K)
    per = sorted(((s, mrr(make_single_source_rank_fn(s))) for s in src_names),
                 key=lambda x: -x[1])
    for s, m in per:
        print(f"   {s:<16}{m:.4f}")
    best_single_name, best_single = per[0]
    agentc = dict(per)["s2_agentC"]
    white = dict(per)["s2_white"]
    print(f"\n   >> Agent C ({agentc:.4f}) vs white-space ({white:.4f}): "
          f"{'Agent C better' if agentc > white else 'white-space better'}")

    # --- 2) fusion experiments ------------------------------------------- #
    llm_srcs = [f"llm:{n}" for n in names]
    llm_by_mrr = sorted(llm_srcs, key=lambda s: -mrr(make_single_source_rank_fn(s)))
    best_s2 = "s2_agentC" if agentc >= white else "s2_white"
    combos = {
        "RRF all 5 LLMs": llm_srcs,
        "RRF top-2 LLMs": llm_by_mrr[:2],
        "RRF top-3 LLMs": llm_by_mrr[:3],
        "RRF best+worst LLM (diverse)": [llm_by_mrr[0], llm_by_mrr[-1]],
        "RRF 5 LLMs + best S2": llm_srcs + [best_s2],
    }
    print(f"\nFusion experiments (uniform RRF, test MRR@{K}):")
    best_fuse = ("", 0.0)
    for label, subset in combos.items():
        m = mrr(make_rrf_rank_fn({s: 1.0 for s in subset}, 60.0))
        tag = "  ** beats best single" if m > best_single + 1e-9 else ""
        print(f"   {label:<30}{m:.4f}{tag}")
        if m > best_fuse[1]:
            best_fuse = (label, m)
    # tuned consensus over the strongest combo
    cfg = tune_consensus(dev, llm_sources=(llm_by_mrr[0], llm_by_mrr[1]),
                         s2_source=best_s2, mrr_k=K, per_direction=PD,
                         n_random=200, coord_rounds=3, restarts=2)
    mc = mrr(cfg.rank_fn())
    print(f"   {'tuned consensus (2 LLMs + S2)':<30}{mc:.4f}"
          f"{'  ** beats best single' if mc > best_single + 1e-9 else ''}")

    print(f"\nbest single source: {best_single_name} = {best_single:.4f}")
    print(f"best fusion:        {best_fuse[0]} = {best_fuse[1]:.4f}  "
          f"({best_fuse[1]-best_single:+.4f} vs best single)")


if __name__ == "__main__":
    main()
