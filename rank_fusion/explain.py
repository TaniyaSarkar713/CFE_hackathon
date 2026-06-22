"""Explainability layer — Direction Cards (plan §6).

A ranked list isn't enough for a research tool: a scientist must trust and
verify each suggestion. Every output direction therefore ships as a
**Direction Card** with a faithful, evidence-grounded rationale. The design
borrows directly from interpretability practice (Ivanitskiy, *Mechanistic
Interpretability*; Lundberg & Lee, SHAP):

  1. Exact additive attribution  — decompose the fused score into per-source /
     per-feature contributions. For RRF the score is *literally* a sum, so the
     decomposition is exact; for the logistic meta-ranker the logit is linear in
     standardised features, so SHAP values are exact (`coef_j * x_j`). This is
     the "input attribution / saliency" layer — faithful by construction.
  2. Causal ablation probe        — leave-one-signal-out: re-rank with a source
     removed and report the rank change ("remove citation -> drops to rank 7").
     This is a mechanistic intervention, not a correlational story.
  3. Logical reasoning            — a defeasible argumentation engine turns the
     signals into PRO/CON arguments, resolves attacks under grounded semantics,
     and emits a verdict (recommend / caveat / abstain) with a reasoning trace.
  4. Confidence & abstention      — a calibrated score (ECE reported); thin
     evidence yields "insufficient evidence" rather than a confident guess.
  5. Faithfulness check           — verify a displayed rationale's claimed drivers
     match the actual top attribution drivers (guards against post-hoc stories /
     Goodharting the explanation).
  6. Evidence grounding           — link a direction to the specific limitations
     / gaps it addresses, with quoted spans (uses the real summary `key_gaps`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fusion import weighted_rrf
from .schema import Query


# --------------------------------------------------------------------------- #
# Direction Card schema (plan §6) + interpretability extensions
# --------------------------------------------------------------------------- #
@dataclass
class DirectionCard:
    direction: str
    candidate_id: str
    rank: int
    score: float
    confidence: str                                  # high | medium | low | abstain
    verdict: str                                     # recommend | recommend_with_caveat | abstain
    rationale: str                                   # generated FROM attribution -> faithful
    addresses: List[str] = field(default_factory=list)
    evidence_spans: List[str] = field(default_factory=list)
    supported_by: List[str] = field(default_factory=list)
    agent_view: Dict = field(default_factory=dict)
    horizon: Dict = field(default_factory=dict)
    top_factors: List[Tuple[str, float]] = field(default_factory=list)
    attribution: Dict[str, float] = field(default_factory=dict)
    ablation: Dict[str, int] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)
    faithfulness: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["top_factors"] = [[n, round(v, 4)] for n, v in self.top_factors]
        return d


# --------------------------------------------------------------------------- #
# 1. Attribution  (exact additive decomposition — "saliency")
# --------------------------------------------------------------------------- #
def rrf_attribution(query: Query, cand_id: str,
                    weights: Optional[Dict[str, float]] = None, k=60.0
                    ) -> Dict[str, float]:
    """Per-source contribution to the RRF score: w_s / (k_s + rank_s(d)).

    These sum *exactly* to the fused score, so the explanation cannot drift from
    the computation (the interpretability ideal of a faithful decomposition).
    """
    def k_of(s):
        return float(k.get(s, 60.0)) if isinstance(k, dict) else float(k)

    contrib: Dict[str, float] = {}
    for src, ids in query.rankings.items():
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        if w == 0.0:
            continue
        try:
            r = ids.index(cand_id) + 1
        except ValueError:
            continue
        contrib[src] = w / (k_of(src) + r)
    return contrib


def linear_shap(meta_ranker, query: Query, cand_id: str) -> Dict[str, float]:
    """Exact SHAP values for the logistic meta-ranker (background = train mean).

    For a linear logit  b + sum_j coef_j * z_j  with standardised z (zero mean),
    the SHAP value of feature j is exactly  coef_j * z_j  (Lundberg & Lee 2017).
    """
    from .meta_ranker import _query_features, feature_names

    if getattr(meta_ranker, "model_kind", "logistic") != "logistic":
        return {}
    cand_ids, X = _query_features(query, meta_ranker.sources, meta_ranker.k)
    if cand_id not in cand_ids:
        return {}
    i = cand_ids.index(cand_id)
    keep = meta_ranker.keep_cols
    names = [n for n, kp in zip(feature_names(meta_ranker.sources), keep) if kp]
    z = (X[i, keep] - meta_ranker.scaler.mean_) / meta_ranker.scaler.scale_
    coef = meta_ranker.model.coef_[0]
    return {n: float(coef[j] * z[j]) for j, n in enumerate(names)}


# --------------------------------------------------------------------------- #
# 2. Causal ablation probe  (leave-one-signal-out intervention)
# --------------------------------------------------------------------------- #
def ablation_probe(query: Query, cand_id: str,
                   weights: Optional[Dict[str, float]] = None, k=60.0
                   ) -> Dict[str, int]:
    """Rank change when each source is removed from the fusion.

    Positive delta = the candidate *drops* that many places without the source
    (i.e. the source was holding it up). A causal, leave-one-out intervention.
    """
    base_order, _ = weighted_rrf(query.rankings, weights, k, query.candidate_ids)
    base_rank = base_order.index(cand_id) + 1 if cand_id in base_order else None
    if base_rank is None:
        return {}
    deltas: Dict[str, int] = {}
    for src in query.rankings:
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        if w == 0.0:
            continue
        ablated = {s: ids for s, ids in query.rankings.items() if s != src}
        order, _ = weighted_rrf(ablated, weights, k, query.candidate_ids)
        new_rank = order.index(cand_id) + 1 if cand_id in order else len(order) + 1
        deltas[src] = new_rank - base_rank          # >0 means it dropped
    return deltas


# --------------------------------------------------------------------------- #
# 3. Logical reasoning  (defeasible argumentation, grounded semantics)
# --------------------------------------------------------------------------- #
@dataclass
class _Rule:
    name: str
    polarity: str                # PRO | CON
    claim: str                   # recommend | caveat | abstain
    needs: List[str]             # predicate names that must be true
    attacks: List[str]           # rule names this one attacks
    template: str                # human-readable justification


# Ordered rule base. PRO rules argue *for* a claim; CON rules attack them.
_RULES: List[_Rule] = [
    _Rule("strong_consensus", "PRO", "recommend",
          ["agent_agreement", "multi_source"], [],
          "GPT and Claude agree and >=3 sources rank it -> strong consensus."),
    _Rule("grounded_in_gap", "PRO", "recommend",
          ["addresses_explicit", "retrieval_support"], [],
          "Addresses an explicit limitation AND is supported by retrieval "
          "neighbours -> evidence-grounded next step."),
    _Rule("exploratory_novelty", "PRO", "caveat",
          ["novel_adjacent"], [],
          "Adjacent to the paper but in a sparse region -> exploratory 'white "
          "space' direction."),
    _Rule("thin_evidence", "CON", "abstain",
          ["thin"], ["strong_consensus", "grounded_in_gap", "exploratory_novelty"],
          "Backed by too few signals -> insufficient evidence; abstain."),
    _Rule("ranker_conflict", "CON", "caveat",
          ["disagreement"], ["strong_consensus"],
          "GPT and Claude strongly disagree -> needs a tie-break; downgrade to "
          "caveat."),
    _Rule("popularity_bias", "CON", "caveat",
          ["popularity_only"], ["strong_consensus", "grounded_in_gap"],
          "Carried mainly by popularity (a bias signal, not truth) -> flag, "
          "don't trust the rank."),
]


def _grounded_extension(active: List[str], attacks: List[Tuple[str, str]]):
    """Standard grounded labelling: IN if every attacker is OUT."""
    active_set = set(active)
    IN: set = set()
    OUT: set = set()
    changed = True
    while changed:
        changed = False
        for a in active_set - IN - OUT:
            attackers = [x for (x, y) in attacks if y == a and x in active_set]
            if all(x in OUT for x in attackers):     # vacuously true if no attackers
                IN.add(a)
                changed = True
        for a in active_set - IN - OUT:
            if any(x in IN for (x, y) in attacks if y == a):
                OUT.add(a)
                changed = True
    return IN, OUT


def reason_logically(predicates: Dict[str, bool]) -> Tuple[str, List[str], List[str]]:
    """Run the argumentation engine; return (verdict, trace, justified_rules).

    verdict in {recommend, recommend_with_caveat, abstain}.
    """
    active = [r.name for r in _RULES if all(predicates.get(p, False) for p in r.needs)]
    by_name = {r.name: r for r in _RULES}
    attacks = [(r.name, t) for r in _RULES if r.name in active
               for t in r.attacks if t in active]
    IN, OUT = _grounded_extension(active, attacks)

    trace: List[str] = []
    for name in active:
        r = by_name[name]
        status = "ACCEPTED" if name in IN else ("DEFEATED" if name in OUT else "UNDECIDED")
        trace.append(f"[{status}] ({r.polarity}->{r.claim}) {r.template}")

    justified = [n for n in active if n in IN]
    claims = {by_name[n].claim for n in justified}
    if "abstain" in claims:
        verdict = "abstain"
    elif "recommend" in claims and "caveat" not in claims:
        verdict = "recommend"
    elif "recommend" in claims and "caveat" in claims:
        verdict = "recommend_with_caveat"
    elif "caveat" in claims:
        verdict = "recommend_with_caveat"
    else:
        verdict = "abstain"
    if not trace:
        trace.append("[NEUTRAL] No decisive signal fired -> default abstain.")
    return verdict, trace, justified


# --------------------------------------------------------------------------- #
# 4. Calibration / confidence  (model-risk-aware)
# --------------------------------------------------------------------------- #
class Calibrator:
    """Map a raw fused score to a calibrated P(relevant), and report ECE."""

    def __init__(self):
        self.model = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]):
        from sklearn.linear_model import LogisticRegression
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        if len(set(y.tolist())) < 2:
            self.model = None
            return self
        self.model = LogisticRegression(max_iter=1000).fit(s, y)
        return self

    def prob(self, score: float) -> float:
        if self.model is None:
            return float(score)
        return float(self.model.predict_proba([[score]])[0, 1])

    def ece(self, scores: Sequence[float], labels: Sequence[int], n_bins: int = 10
            ) -> float:
        probs = np.array([self.prob(s) for s in scores])
        y = np.asarray(labels, dtype=int)
        if len(y) == 0:
            return 0.0
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            m = (probs >= bins[b]) & (probs < bins[b + 1] + (b == n_bins - 1) * 1e-9)
            if m.sum() == 0:
                continue
            ece += (m.sum() / len(y)) * abs(probs[m].mean() - y[m].mean())
        return float(ece)


# --------------------------------------------------------------------------- #
# 5. Faithfulness check
# --------------------------------------------------------------------------- #
def faithfulness_check(claimed_factors: Sequence[str],
                       actual_top_factors: Sequence[str]) -> Dict:
    """Does a displayed rationale cite the factors that actually drove the rank?

    Returns overlap score + a `faithful` flag. Use it to audit an LLM's
    free-text justification against the exact attribution drivers (so the shown
    reason can't be a plausible-but-wrong post-hoc story).
    """
    claimed = {c.lower() for c in claimed_factors}
    actual = {a.lower() for a in actual_top_factors}
    if not claimed:
        return {"score": 1.0, "faithful": True, "claimed": [], "actual": list(actual)}
    overlap = len(claimed & actual) / len(claimed)
    return {"score": round(overlap, 3), "faithful": overlap >= 0.5,
            "claimed": sorted(claimed), "actual": sorted(actual)}


# --------------------------------------------------------------------------- #
# 6. Evidence grounding (extractive, against the summary's key_gaps)
# --------------------------------------------------------------------------- #
def ground_evidence(direction_text: str, gaps: Sequence[Dict], tau: float = 0.10
                    ) -> Tuple[List[str], List[str]]:
    """Link a direction to the limitations/gaps it addresses (TF-IDF cosine).

    `gaps` is a list of dicts like the summary's key_gaps
    ({"description":..., "status":"explicit|implicit"}). Returns
    (addressed labels, quoted spans).
    """
    gap_texts = [g.get("description", "") for g in gaps]
    if not direction_text or not any(gap_texts):
        return [], []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([direction_text] + gap_texts)
    sims = cosine_similarity(mat[0:1], mat[1:])[0]
    addressed, spans = [], []
    for j, s in enumerate(sims):
        if s >= tau:
            status = gaps[j].get("status", "?")
            addressed.append(f"{status} gap: {gap_texts[j][:60]}")
            spans.append(gap_texts[j])
    return addressed, spans


# --------------------------------------------------------------------------- #
# Predicate extraction + card assembly
# --------------------------------------------------------------------------- #
def _signals(query: Query, cand_id: str,
             weights: Optional[Dict[str, float]], k) -> Dict:
    pos = {s: (ids.index(cand_id) + 1 if cand_id in ids else None)
           for s, ids in query.rankings.items()}
    n_sources = sum(1 for r in pos.values() if r is not None)
    gpt_r, claude_r = pos.get("gpt"), pos.get("claude")
    support = sum(1 for s in ("retrieval", "citation")
                  if pos.get(s) is not None and pos[s] <= 15)
    feats = query.features.get(cand_id, {})
    return {"pos": pos, "n_sources": n_sources, "gpt_rank": gpt_r,
            "claude_rank": claude_r, "support": support, "feats": feats}


def _predicates(sig: Dict, evidence_addressed: Sequence[str]) -> Dict[str, bool]:
    gpt_r, claude_r = sig["gpt_rank"], sig["claude_rank"]
    both = gpt_r is not None and claude_r is not None
    agree = both and abs(gpt_r - claude_r) <= 3 and max(gpt_r, claude_r) <= 10
    disagree = both and abs(gpt_r - claude_r) >= 15
    feats = sig["feats"]
    novelty = feats.get("novelty", 0.0)
    adjacency = feats.get("adjacency", feats.get("sim_to_paper", 0.0))
    popularity = feats.get("popularity", 0.0)
    has_explicit = any("explicit" in a for a in evidence_addressed)
    return {
        "agent_agreement": agree,
        "disagreement": disagree,
        "multi_source": sig["n_sources"] >= 3,
        "retrieval_support": sig["support"] >= 2,
        "thin": sig["n_sources"] <= 1,
        "novel_adjacent": novelty >= 0.6 and adjacency >= 0.5,
        "popularity_only": popularity >= 0.6 and sig["support"] == 0 and not agree,
        "addresses_explicit": has_explicit,
    }


_CONF_BANDS = [(0.66, "high"), (0.33, "medium"), (0.0, "low")]


def _confidence(prob: float, verdict: str) -> str:
    if verdict == "abstain":
        return "abstain"
    for thr, label in _CONF_BANDS:
        if prob >= thr:
            return label
    return "low"


def build_direction_card(
    query: Query,
    cand_id: str,
    rank: int,
    direction_text: str = "",
    weights: Optional[Dict[str, float]] = None,
    k=60.0,
    meta_ranker=None,
    calibrator: Optional[Calibrator] = None,
    gaps: Optional[Sequence[Dict]] = None,
    score: Optional[float] = None,
) -> DirectionCard:
    """Assemble a full Direction Card for one candidate."""
    sig = _signals(query, cand_id, weights, k)

    # attribution: prefer the learned model's SHAP, else exact RRF decomposition
    attribution = linear_shap(meta_ranker, query, cand_id) if meta_ranker else {}
    if not attribution:
        attribution = rrf_attribution(query, cand_id, weights, k)
    top_factors = sorted(attribution.items(), key=lambda kv: -abs(kv[1]))[:3]

    ablation = ablation_probe(query, cand_id, weights, k)

    addressed, spans = ([], [])
    if gaps and direction_text:
        addressed, spans = ground_evidence(direction_text, gaps)

    predicates = _predicates(sig, addressed)
    verdict, trace, _ = reason_logically(predicates)

    if score is None:
        _, scores = weighted_rrf(query.rankings, weights, k, query.candidate_ids)
        score = scores.get(cand_id, 0.0)
    prob = calibrator.prob(score) if calibrator else min(1.0, max(0.0, score * 8))
    confidence = _confidence(prob, verdict)

    # provenance
    supported_by = []
    for gname, ids in query.candidate_sources.items():
        if cand_id in ids:
            supported_by.append(f"source:{gname}")
    for s in ("retrieval", "citation"):
        if sig["pos"].get(s) is not None:
            supported_by.append(f"{s}@{sig['pos'][s]}")

    # rationale generated FROM the attribution -> faithful by construction
    drivers = ", ".join(f"{n} ({v:+.3f})" for n, v in top_factors) or "no dominant driver"
    carrier, carrier_delta = (None, 0)
    if ablation:
        carrier, carrier_delta = max(ablation.items(), key=lambda kv: kv[1])
    rationale = (f"Ranked #{rank} ({verdict.replace('_', ' ')}). "
                 f"Top drivers: {drivers}."
                 + (f" Removing '{carrier}' would demote it most "
                    f"(+{carrier_delta} ranks)." if carrier and carrier_delta > 0 else ""))
    faith = faithfulness_check([n for n, _ in top_factors],
                               list(attribution.keys()))

    horizon = {kk: round(sig["feats"].get(kk, 0.0), 3)
               for kk in ("recency", "popularity", "novelty")
               if kk in sig["feats"]}

    return DirectionCard(
        direction=direction_text or cand_id,
        candidate_id=cand_id, rank=rank, score=round(float(score), 4),
        confidence=confidence, verdict=verdict, rationale=rationale,
        addresses=addressed, evidence_spans=spans, supported_by=supported_by,
        agent_view={"gpt_rank": sig["gpt_rank"], "claude_rank": sig["claude_rank"],
                    "agreement": "high" if predicates["agent_agreement"]
                    else ("low" if predicates["disagreement"] else "mixed")},
        horizon=horizon,
        top_factors=[(n, round(v, 4)) for n, v in top_factors],
        attribution={n: round(v, 4) for n, v in attribution.items()},
        ablation=ablation, reasoning_trace=trace, faithfulness=faith,
    )


def explain_ranking(
    query: Query, ranked_ids: Sequence[str], top_k: int = 3,
    direction_texts: Optional[Dict[str, str]] = None, **kwargs
) -> List[DirectionCard]:
    """Build Direction Cards for the top-K of a fused ranking."""
    texts = direction_texts or {}
    cards = []
    for rank, cid in enumerate(ranked_ids[:top_k], start=1):
        cards.append(build_direction_card(
            query, cid, rank, direction_text=texts.get(cid, ""), **kwargs))
    return cards


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_card_markdown(card: DirectionCard) -> str:
    L = []
    badge = {"recommend": "✅ RECOMMEND",
             "recommend_with_caveat": "⚠️ RECOMMEND (caveat)",
             "abstain": "⛔ INSUFFICIENT EVIDENCE"}.get(card.verdict, card.verdict)
    L.append(f"### #{card.rank}  {card.direction}")
    L.append(f"**{badge}**  ·  confidence: *{card.confidence}*  ·  "
             f"score {card.score}")
    L.append(f"\n_{card.rationale}_")
    if card.addresses:
        L.append("\n**Addresses:** " + "; ".join(card.addresses))
    if card.supported_by:
        L.append("**Provenance:** " + ", ".join(card.supported_by))
    av = card.agent_view
    L.append(f"**Agent view:** gpt_rank={av.get('gpt_rank')} "
             f"claude_rank={av.get('claude_rank')} ({av.get('agreement')})")
    if card.horizon:
        L.append("**Horizon:** " + ", ".join(f"{k}={v}" for k, v in card.horizon.items()))
    L.append("**Attribution (top factors):** "
             + ", ".join(f"{n} {v:+.3f}" for n, v in card.top_factors))
    if card.ablation:
        abl = ", ".join(f"{s}:{d:+d}" for s, d in sorted(
            card.ablation.items(), key=lambda kv: -kv[1]))
        L.append(f"**Ablation (Δrank if removed):** {abl}")
    L.append("**Reasoning trace:**")
    for step in card.reasoning_trace:
        L.append(f"  - {step}")
    f = card.faithfulness
    L.append(f"**Faithfulness:** rationale↔drivers overlap {f['score']} "
             f"({'OK' if f['faithful'] else 'MISMATCH'})")
    return "\n".join(L)
