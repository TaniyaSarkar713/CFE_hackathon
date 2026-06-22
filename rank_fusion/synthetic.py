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
