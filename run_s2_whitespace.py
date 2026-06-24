"""Pull connected papers from Semantic Scholar -> white-space future directions.

Pipeline:
  1. resolve the seed paper's S2 CorpusId (via Naz's S2Client),
  2. pull its CONNECTED papers = references + citations (title + abstract),
  3. feed those neighbour docs to rank_fusion.whitespace_candidates, which mines
     the entity co-occurrence graph for "adjacent-but-unexplored" bridge
     directions (boosted near the seed's open gaps).

Alternatively (same neighbourhood, different method) it runs Naz's Agent C
(S2DirectionAgent), which ranks the neighbour TITLES by graph signals.

    python run_s2_whitespace.py [PMCID]        # default 11816194

Needs network + an S2 API key (env S2_API_KEY). Falls back to the local corpus
as the neighbourhood if S2 is unreachable, so it always produces output.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import whitespace_candidates, keyphrases

DATA = os.path.join(os.path.dirname(__file__), "Data")
S2_KEY = os.environ.get("S2_API_KEY", "s2k-VqZPfeS1kZk3Mqd5ac7i1PKN0rv15d8wrKm7LNUR")


def seed_info(pid):
    s = json.load(open(os.path.join(DATA, "Paper_summaries", f"{pid}_future_directions.json")))
    title = open(os.path.join(DATA, "papers", f"{pid}.txt")).readline().strip()
    seed_ents = keyphrases(s.get("topic_and_scope", "") + " " + " ".join(s.get("main_themes", [])))
    gap_ents = keyphrases(" ".join(g["description"] for g in s.get("key_gaps", [])))
    return title, seed_ents, gap_ents


def pull_connected_papers(title, limit=20):
    """references + citations from S2 (each carries title + abstract). Returns
    (corpus_id, [neighbour dicts]) or (None, []) on failure."""
    from semantic_scholar_ranking.s2_client import S2Client
    client = S2Client(api_key=S2_KEY, rps=0.5)
    cid = client.resolve_corpus_id(title)
    if not cid:
        return None, []
    refs = client.get_references(cid, limit=limit) or []
    cites = client.get_citations(cid, limit=limit) or []
    seen, neighbours = set(), []
    for p in refs + cites:
        t = (p or {}).get("title")
        if t and t not in seen:
            seen.add(t)
            neighbours.append(p)
    return cid, neighbours


def local_neighbourhood(pid, k=30):
    """Offline fallback: top-k similar local papers as the 'connected' set."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    docs, ids = {}, []
    for f in glob.glob(os.path.join(DATA, "Paper_summaries", "*_future_directions.json")):
        p = os.path.basename(f).replace("_future_directions.json", "")
        s = json.load(open(f))
        docs[p] = s.get("topic_and_scope", "") + " " + " ".join(s.get("main_themes", []))
        ids.append(p)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([docs[i] for i in ids])
    pos = {i: j for j, i in enumerate(ids)}
    sims = cosine_similarity(mat[pos[pid]], mat)[0]
    order = [ids[j] for j in sims.argsort()[::-1] if ids[j] != pid][:k]
    return [{"title": docs[n][:80], "abstract": docs[n]} for n in order]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pid", nargs="?", default="11816194")
    ap.add_argument("--limit", type=int, default=20, help="refs+cites to pull per side")
    # min_support=2 (recurring concepts) is cleaner than 1, which floods with
    # single-paper fragments from one high-overlap neighbour. Real NER as the
    # entity_fn is what would most improve bridge quality.
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()
    pid = args.pid

    title, seed_ents, gap_ents = seed_info(pid)
    print(f"SEED {pid}: {title[:88]}")

    print("\n--- pulling connected papers from Semantic Scholar ---")
    try:
        cid, neighbours = pull_connected_papers(title, limit=args.limit)
    except Exception as e:
        print(f"   [s2] failed: {type(e).__name__}: {e}")
        cid, neighbours = None, []
    if neighbours:
        print(f"pulled {len(neighbours)} connected papers (CorpusID {cid}); e.g.:")
        for p in neighbours[:4]:
            print(f"   - {p['title'][:78]}")
        source = "Semantic Scholar (references + citations)"
    else:
        print("   S2 unreachable/empty -> falling back to local corpus neighbourhood")
        neighbours = local_neighbourhood(pid, k=30)
        source = "local corpus (offline fallback)"

    neighbour_docs = [f"{p.get('title','')}. {p.get('abstract','') or ''}" for p in neighbours]

    print(f"\n--- white-space future directions (from {source}) ---")
    ws = whitespace_candidates(seed_ents, neighbour_docs, gap_entities=gap_ents,
                               min_support=args.min_support, top_k=args.top_k)
    if not ws:
        print("   (no bridge entities found)")
    for i, c in enumerate(ws[:args.top_k], 1):
        print(f"  {i:>2}. [{c.score:.2f}] {c.entity:<26} (connects: {c.connects[:2]})")

    # alternatively: Agent C ranks the neighbour TITLES by graph signals
    if cid:
        print("\n--- alternatively: Agent C (S2DirectionAgent) on the same graph ---")
        try:
            from semantic_scholar_ranking.s2_client import S2Client, GraphSignalExtractor
            from semantic_scholar_ranking.s2_agent import S2DirectionAgent
            client = S2Client(api_key=S2_KEY, rps=0.5)
            ext = GraphSignalExtractor(client)
            ctx = ext.build_context(cid)
            out = S2DirectionAgent(ext, max_hops=1, max_per_hop=8).run(cid, ctx)
            for i, t in enumerate(out[1][:5], 1):
                print(f"  {i}. {t[:74]}")
        except Exception as e:
            print(f"   [agent C] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
