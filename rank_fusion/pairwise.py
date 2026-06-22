"""Swiss-system tournament: aggregate pairwise comparisons into a global rank.

If an Agent-2 ranker does *pairwise* judging ("is direction A a more likely next
step than B?", Qin et al. PRP), we need to turn O(n) sparse comparisons into one
global ordering. A full round-robin is O(n²) LLM calls; Bradley-Terry / Elo need
many comparisons to converge. A **Swiss tournament** is the cheap middle ground
(as used by Si et al. 2024 to rank generated ideas):

  - every item starts at 0 points;
  - each of N rounds, sort by accumulated points and pair neighbours
    (similar-scored items play each other), the winner gains a point;
  - after N≈5 rounds, total points give the global ranking.

Cost is N·n/2 comparisons (≈ 2.5·n at N=5) instead of n²/2, and pairing
similar-scored items spends the comparisons where they actually separate ties.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple


def swiss_tournament(
    items: Sequence,
    compare: Callable[[object, object], bool],
    n_rounds: int = 5,
    seed: int = 0,
    avoid_rematches: bool = True,
) -> Tuple[List, Dict]:
    """Run a Swiss tournament. `compare(a, b)` returns True iff a beats b.

    Returns (ranked items best-first, points dict). A bye (odd item out) scores
    a point, mirroring a win.
    """
    items = list(items)
    rng = random.Random(seed)
    pts: Dict = {it: 0.0 for it in items}
    played = {it: set() for it in items}

    for _ in range(n_rounds):
        order = sorted(items, key=lambda it: (-pts[it], rng.random()))
        used = set()
        i = 0
        while i < len(order):
            a = order[i]
            if a in used:
                i += 1
                continue
            # find the next available opponent, preferring an un-played one
            b = None
            for j in range(i + 1, len(order)):
                cand = order[j]
                if cand in used:
                    continue
                if avoid_rematches and cand in played[a] and b is None:
                    b = cand            # remember as fallback, keep looking
                    continue
                b = cand
                break
            if b is None:               # a is the odd one out -> bye
                pts[a] += 1.0
                used.add(a)
                i += 1
                continue
            used.add(a)
            used.add(b)
            played[a].add(b)
            played[b].add(a)
            if compare(a, b):
                pts[a] += 1.0
            else:
                pts[b] += 1.0
            i += 1

    ranked = sorted(items, key=lambda it: (-pts[it], rng.random()))
    return ranked, pts


def comparator_from_scores(scores: Dict, noise: float = 0.0, seed: int = 0):
    """A `compare(a, b)` from latent scores + logistic noise (for tests/sims).

    P(a beats b) = sigmoid((score_a - score_b)/temperature); higher noise =
    noisier (less reliable) pairwise judge.
    """
    import math
    rng = random.Random(seed)
    temp = max(noise, 1e-6)

    def compare(a, b) -> bool:
        diff = (scores[a] - scores[b]) / temp
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, diff))))
        return rng.random() < p

    return compare
