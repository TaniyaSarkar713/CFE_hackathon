"""Synthetic ranking generator for validating the fusion engine.

We need data where (a) each source is individually noisy but informative,
(b) fusing diverse sources beats the best single source, and (c) the sources
have *different* reliabilities so MRR-tuned weights beat uniform weights.

Model (per paper):
  - A candidate pool of `pool_size` directions, of which a few are "true"
    (each true candidate IS its own ground-truth direction g_j, exact-id match).
  - Each true candidate has a latent quality q ~ N(1, .) ; distractors ~ N(0, .).
  - Each source m has a skill `s_m`: it observes  score = s_m * q + noise_m,
    where noise_m has an independent part (diverse errors) and a small shared
    part (sources partly agree). The source ranks candidates by observed score.

Higher skill -> true candidates float to the top -> higher single-source MRR.
Independent noise -> fusion cancels errors. Skill spread -> non-uniform optimal
weights. Defaults give: best single < uniform RRF < tuned RRF.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from .schema import Dataset, Query

DEFAULT_SKILLS: Dict[str, float] = {
    "gpt": 1.7,        # strongest ranker
    "claude": 1.3,     # strong, diverse errors vs gpt
    "retrieval": 0.9,  # weaker future-work-bank kNN
    "citation": 0.5,   # weakest / sparse forward-citation signal
}


def make_synthetic_dataset(
    n_papers: int = 300,
    pool_size: int = 50,
    skills: Optional[Dict[str, float]] = None,
    min_true: int = 1,
    max_true: int = 3,
    noise_sd: float = 1.0,
    shared_noise_sd: float = 0.4,
    split: Sequence[float] = (0.5, 0.25, 0.25),
    attach_features: bool = False,
    seed: int = 7,
) -> Dataset:
    """Build a `Dataset` of synthetic ranking instances with a temporal-style split.

    If `attach_features=True`, each candidate also gets horizon-style features
    (adjacency / novelty / popularity / recency) on `Query.features`, so the
    explainability layer (rank_fusion.explain) has signals to reason over.
    """
    skills = dict(skills or DEFAULT_SKILLS)
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    queries: List[Query] = []
    for p in range(n_papers):
        pid = f"P{p:04d}"
        cand_ids = [f"{pid}_c{j:02d}" for j in range(pool_size)]

        n_true = pyrng.randint(min_true, max_true)
        true_idx = set(pyrng.sample(range(pool_size), n_true))

        # latent quality: true candidates higher than distractors
        q = rng.normal(0.0, 1.0, size=pool_size)
        for j in true_idx:
            q[j] = rng.normal(2.0, 0.5)

        # shared confounder so sources partly agree (realistic, not independent)
        shared = rng.normal(0.0, shared_noise_sd, size=pool_size)

        rankings: Dict[str, List[str]] = {}
        for src, skill in skills.items():
            observed = skill * q + shared + rng.normal(0.0, noise_sd, size=pool_size)
            order = np.argsort(-observed)
            rankings[src] = [cand_ids[i] for i in order]

        # exact-id ground truth: each true candidate is its own direction g_j
        relevance = {f"g{n}": {cand_ids[j]} for n, j in enumerate(sorted(true_idx))}

        features: Dict[str, Dict[str, float]] = {}
        if attach_features:
            for j, cid in enumerate(cand_ids):
                is_true = j in true_idx
                features[cid] = {
                    # true candidates are more on-topic (higher adjacency)
                    "adjacency": float(rng.uniform(0.4, 0.9) if is_true
                                       else rng.uniform(0.1, 0.6)),
                    "sim_to_paper": float(rng.uniform(0.4, 0.9) if is_true
                                          else rng.uniform(0.1, 0.6)),
                    "novelty": float(rng.uniform(0.0, 1.0)),
                    "popularity": float(rng.uniform(0.0, 1.0)),
                    "recency": float(rng.uniform(0.0, 1.0)),
                }

        queries.append(Query(pid, cand_ids, rankings, relevance, features=features))

    # temporal-style split: oldest papers train, newest test (by index order)
    n = len(queries)
    n_tr = int(split[0] * n)
    n_dev = int(split[1] * n)
    ids = [q.query_id for q in queries]
    train_ids = ids[:n_tr]
    dev_ids = ids[n_tr:n_tr + n_dev]
    test_ids = ids[n_tr + n_dev:]

    return Dataset(queries, train_ids, dev_ids, test_ids)


# Default candidate generators: name -> (per-direction recall, #distractors).
# LLM brainstorm has the highest recall; citation snippets the lowest — so the
# *union* recall is well above any single source (Level 1's headline).
DEFAULT_GENERATORS: Dict[str, tuple] = {
    "gen_llm": (0.75, 18),
    "gen_retrieval": (0.55, 14),
    "gen_citation": (0.35, 10),
    "gen_limitation_tmpl": (0.45, 8),
}


def make_candidate_generation_dataset(
    n_papers: int = 200,
    n_true: int = 3,
    generators: Optional[Dict[str, tuple]] = None,
    ranker_skill: float = 1.3,
    noise_sd: float = 1.0,
    split: Sequence[float] = (0.5, 0.25, 0.25),
    seed: int = 13,
) -> Dataset:
    """Model the *candidate generation* stage to expose pool recall (Level 1).

    Each generator recalls a true direction with its own probability and adds
    distractors; the candidate pool is their union, so some true directions can
    be missed entirely (oracle MRR < 1). Each Query carries `candidate_sources`
    (provenance) and a single noisy "ranker" over the pool, so you can show that
    union recall is the ceiling and ordering only matters up to it.
    """
    generators = dict(generators or DEFAULT_GENERATORS)
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    queries: List[Query] = []
    for p in range(n_papers):
        pid = f"P{p:04d}"
        true_ids = [f"{pid}_t{j}" for j in range(n_true)]

        candidate_sources: Dict[str, List[str]] = {}
        pool: List[str] = []
        for gname, (recall, n_dist) in generators.items():
            got: List[str] = []
            for tid in true_ids:                       # maybe recall each true dir
                if rng.random() < recall:
                    got.append(tid)
            for d in range(n_dist):                    # generator-specific distractors
                got.append(f"{pid}_{gname}_d{d}")
            candidate_sources[gname] = got
            for c in got:
                if c not in pool:
                    pool.append(c)

        # one noisy ranker over the pool (true ids score higher)
        latent = {c: rng.normal(2.0, 0.5) if c in true_ids else rng.normal(0.0, 1.0)
                  for c in pool}
        observed = {c: ranker_skill * latent[c] + rng.normal(0.0, noise_sd)
                    for c in pool}
        ranking = sorted(pool, key=lambda c: -observed[c])

        # relevance references the canonical true id (empty match if pool missed it)
        relevance = {f"g{j}": ({true_ids[j]} if true_ids[j] in pool else set())
                     for j in range(n_true)}

        queries.append(Query(pid, pool, {"ranker": ranking}, relevance,
                             candidate_sources=candidate_sources))

    n = len(queries)
    n_tr, n_dev = int(split[0] * n), int(split[1] * n)
    ids = [q.query_id for q in queries]
    return Dataset(queries, ids[:n_tr], ids[n_tr:n_tr + n_dev], ids[n_tr + n_dev:])


def make_consensus_dataset(
    n_papers: int = 300,
    pool_size: int = 40,
    min_true: int = 1,
    max_true: int = 3,
    llm_skill: float = 1.5,
    s2_skill: float = 0.7,
    noise_sd: float = 1.0,
    salient_frac: float = 0.6,
    split: Sequence[float] = (0.5, 0.25, 0.25),
    seed: int = 23,
) -> Dataset:
    """Synthetic data for consensus-aware fusion (sources: llm1, llm2, s2).

    Some true directions are *salient*: both LLMs rank them at the top (so
    LLM-LLM agreement fires) AND S2 supports them — making the agreement term and
    the agreement×S2 interaction genuinely predictive, so a dev-MRR tuner learns
    positive eta/rho. Non-salient trues get only a weak S2 signal; distractors get
    independent noise (rarely both LLMs at once).
    """
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    queries: List[Query] = []
    for p in range(n_papers):
        pid = f"P{p:04d}"
        cand = [f"{pid}_c{j:02d}" for j in range(pool_size)]
        n_true = pyrng.randint(min_true, max_true)
        true_idx = set(pyrng.sample(range(pool_size), n_true))
        q = rng.normal(0.0, 1.0, pool_size)
        for j in true_idx:
            q[j] = rng.normal(2.0, 0.4)
        salient = {j for j in true_idx if rng.random() < salient_frac}

        l1 = llm_skill * q + rng.normal(0, noise_sd, pool_size)
        l2 = llm_skill * q + rng.normal(0, noise_sd, pool_size)
        s2 = rng.normal(0, noise_sd, pool_size)            # s2 is mostly noise...
        for j in salient:                                   # ...except it backs consensus
            bump = rng.normal(3.0, 0.3)
            l1[j] += bump
            l2[j] += bump
            s2[j] += s2_skill * 2.5
        for j in true_idx - salient:
            s2[j] += s2_skill * q[j]

        def order(v):
            return [cand[i] for i in np.argsort(-v)]
        rankings = {"llm1": order(l1), "llm2": order(l2), "s2": order(s2)}
        relevance = {f"g{n}": {cand[j]} for n, j in enumerate(sorted(true_idx))}
        queries.append(Query(pid, cand, rankings, relevance))

    n = len(queries)
    n_tr, n_dev = int(split[0] * n), int(split[1] * n)
    ids = [x.query_id for x in queries]
    return Dataset(queries, ids[:n_tr], ids[n_tr:n_tr + n_dev], ids[n_tr + n_dev:])


def make_redundancy_dataset(
    n_papers: int = 200,
    n_clusters: int = 6,
    cluster_size: int = 4,
    n_true: int = 2,
    emb_dim: int = 8,
    emb_noise: float = 0.25,
    skills: Optional[Dict[str, float]] = None,
    noise_sd: float = 0.2,
    split: Sequence[float] = (0.5, 0.25, 0.25),
    seed: int = 21,
) -> Dataset:
    """Synthetic data with near-duplicate candidate clusters (for MMR, §9.3).

    Each paper has `n_clusters` concept clusters of `cluster_size` near-duplicate
    candidates (similar embeddings). `n_true` *distinct* clusters each contain one
    true direction (its cleanest member). True clusters have different *salience*,
    so a relevance-only ranking fills the top with one salient cluster's
    near-duplicates and buries the other true direction — exactly the case MMR
    fixes. Candidate embeddings are stored on `Query.features` as `emb_*` dims so
    `feature_embedding_sim` can build the similarity for MMR.
    """
    skills = dict(skills or DEFAULT_SKILLS)
    dim = max(emb_dim, n_clusters)
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    queries: List[Query] = []
    for p in range(n_papers):
        pid = f"P{p:04d}"
        true_clusters = pyrng.sample(range(n_clusters), n_true)
        # decreasing salience so one true cluster dominates the top
        salience = {c: s for c, s in zip(true_clusters,
                    [1.0, 0.65, 0.5, 0.4][:n_true])}

        cand_ids, embs, true_ids = [], [], {}
        for c in range(n_clusters):
            center = np.zeros(dim)
            center[c] = 1.0
            for m in range(cluster_size):
                cid = f"{pid}_c{c}m{m}"
                if m == 0:
                    e = center.copy()                 # cleanest member = the true one
                else:
                    e = center + emb_noise * rng.normal(0, 1, dim)
                e = e / (np.linalg.norm(e) + 1e-9)
                cand_ids.append(cid)
                embs.append(e)
                if c in true_clusters and m == 0:
                    true_ids[c] = cid

        embs = np.asarray(embs)
        # quality: proximity to a true cluster centre, weighted by its salience
        quality = np.zeros(len(cand_ids))
        for c, sal in salience.items():
            center = np.zeros(dim)
            center[c] = 1.0
            quality += sal * (embs @ center)
        qz = (quality - quality.mean()) / (quality.std() + 1e-9)

        rankings: Dict[str, List[str]] = {}
        for src, skill in skills.items():
            obs = skill * qz + rng.normal(0, noise_sd, len(cand_ids))
            rankings[src] = [cand_ids[i] for i in np.argsort(-obs)]

        relevance = {f"g{n}": {true_ids[c]} for n, c in enumerate(true_clusters)}
        features = {cid: {f"emb_{j}": float(embs[i, j]) for j in range(dim)}
                    for i, cid in enumerate(cand_ids)}
        queries.append(Query(pid, cand_ids, rankings, relevance, features=features))

    n = len(queries)
    n_tr, n_dev = int(split[0] * n), int(split[1] * n)
    ids = [q.query_id for q in queries]
    return Dataset(queries, ids[:n_tr], ids[n_tr:n_tr + n_dev], ids[n_tr + n_dev:])
