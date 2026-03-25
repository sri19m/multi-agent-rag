"""
RL-inspired reward ranker for agent response selection.
Scores candidate responses on relevance, completeness, and confidence,
then selects the best response — mimicking a reward model in RLHF.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Container for a candidate response from any agent."""
    content: str
    agent_name: str
    retrieval_source: str = "hybrid"          # bm25 | vector | hybrid | tool
    metadata: dict = field(default_factory=dict)

    # Scores (filled by RewardRanker)
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    confidence_score: float = 0.0
    total_reward: float = 0.0


class RewardRanker:
    """
    Multi-criteria reward ranker for agent response selection.

    Scoring dimensions:
    - Relevance:     Keyword overlap between query and response (proxy for faithfulness)
    - Completeness:  Response length and structural richness (paragraphs, numbers, etc.)
    - Confidence:    Absence of hedging language; presence of specific claims

    Final reward: weighted sum of all three scores.
    Mimics the reward model in RLHF — selects the highest-reward response
    to surface to the user.
    """

    # Hedging phrases reduce confidence score
    HEDGE_PATTERNS = [
        r"\bi (don'?t|cannot|can'?t) (know|say|tell|confirm)\b",
        r"\bit'?s? (unclear|uncertain|hard to say)\b",
        r"\bmight be\b", r"\bperhaps\b", r"\bmaybe\b",
        r"\bi'?m not sure\b", r"\bpossibly\b",
    ]

    # Positive signal: numbers, citations, structured claims
    CONFIDENCE_SIGNALS = [
        r"\$[\d,.]+[BMK]?\b",           # Dollar amounts
        r"\d+\.?\d*\s?%",               # Percentages
        r"\bQ[1-4]\s?\d{4}\b",          # Quarter references
        r"\bFY\d{4}\b",                 # Fiscal year
        r"\b(increased|decreased|grew|declined)\b",  # Directional claims
    ]

    def __init__(
        self,
        relevance_weight: float = 0.5,
        completeness_weight: float = 0.3,
        confidence_weight: float = 0.2,
        min_score_threshold: float = 0.3,
    ):
        if abs(relevance_weight + completeness_weight + confidence_weight - 1.0) > 1e-6:
            raise ValueError("Reward weights must sum to 1.0")

        self.w_relevance = relevance_weight
        self.w_completeness = completeness_weight
        self.w_confidence = confidence_weight
        self.threshold = min_score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self, query: str, responses: List[AgentResponse]
    ) -> List[AgentResponse]:
        """
        Score all candidate responses and return sorted list (best first).

        Args:
            query: Original user query.
            responses: List of candidate AgentResponse objects.

        Returns:
            Responses sorted by total_reward descending.
        """
        if not responses:
            return []

        for resp in responses:
            resp.relevance_score = self._score_relevance(query, resp.content)
            resp.completeness_score = self._score_completeness(resp.content)
            resp.confidence_score = self._score_confidence(resp.content)
            resp.total_reward = round(
                self.w_relevance * resp.relevance_score
                + self.w_completeness * resp.completeness_score
                + self.w_confidence * resp.confidence_score,
                4,
            )

        ranked = sorted(responses, key=lambda r: r.total_reward, reverse=True)

        logger.info(
            f"RewardRanker: scored {len(ranked)} responses. "
            f"Best agent='{ranked[0].agent_name}' score={ranked[0].total_reward:.3f}"
        )
        return ranked

    def select_best(
        self, query: str, responses: List[AgentResponse]
    ) -> Optional[AgentResponse]:
        """Rank and return single best response above threshold."""
        ranked = self.rank(query, responses)
        if not ranked:
            return None

        best = ranked[0]
        if best.total_reward < self.threshold:
            logger.warning(
                f"Best response score {best.total_reward:.3f} below threshold "
                f"{self.threshold}. Response may be low quality."
            )
        return best

    def get_score_report(self, responses: List[AgentResponse]) -> str:
        """Human-readable score breakdown for all responses."""
        if not responses:
            return "No responses to report."

        lines = ["=" * 60, "REWARD RANKER SCORE REPORT", "=" * 60]
        for i, r in enumerate(responses, 1):
            lines.append(
                f"\n#{i} Agent: {r.agent_name} | Source: {r.retrieval_source}"
                f"\n   Relevance:    {r.relevance_score:.3f}"
                f"\n   Completeness: {r.completeness_score:.3f}"
                f"\n   Confidence:   {r.confidence_score:.3f}"
                f"\n   TOTAL REWARD: {r.total_reward:.3f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scoring heuristics
    # ------------------------------------------------------------------

    def _score_relevance(self, query: str, response: str) -> float:
        """
        Unigram overlap between query terms and response (Jaccard-like).
        Simple but effective proxy for faithfulness without calling LLM.
        """
        q_tokens = set(self._tokenize(query))
        r_tokens = set(self._tokenize(response))

        if not q_tokens:
            return 0.0

        overlap = len(q_tokens & r_tokens)
        score = overlap / len(q_tokens)
        return min(score * 1.5, 1.0)  # Boost and cap

    def _score_completeness(self, response: str) -> float:
        """
        Heuristic completeness score based on response structure and length.
        """
        if not response.strip():
            return 0.0

        score = 0.0
        length = len(response.split())

        # Length score: optimal zone is 80-400 words
        if length < 20:
            score += 0.1
        elif length < 80:
            score += 0.3
        elif length <= 400:
            score += 0.6
        else:
            score += 0.5  # Very long may be verbose

        # Structural richness
        if re.search(r"\n", response):
            score += 0.1        # Multi-line structure
        if re.search(r"[\-\•\*]\s", response):
            score += 0.1        # Bullet points
        if re.search(r"\d+\.", response):
            score += 0.1        # Numbered items
        if len(response.split(".")) > 3:
            score += 0.1        # Multiple sentences

        return min(score, 1.0)

    def _score_confidence(self, response: str) -> float:
        """
        Confidence score: penalize hedging, reward specific factual claims.
        """
        score = 0.5  # Neutral baseline

        # Penalize hedging
        for pattern in self.HEDGE_PATTERNS:
            matches = len(re.findall(pattern, response, re.IGNORECASE))
            score -= matches * 0.08

        # Reward specific financial signals
        for pattern in self.CONFIDENCE_SIGNALS:
            matches = len(re.findall(pattern, response, re.IGNORECASE))
            score += matches * 0.06

        return max(0.0, min(score, 1.0))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]
