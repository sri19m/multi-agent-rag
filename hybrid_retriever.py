"""
Hybrid retriever combining BM25 (sparse) and dense vector retrieval.
Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from both retrievers.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .bm25_retriever import BM25Document, BM25Retriever
from .vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Unified retrieval result from hybrid retriever."""
    content: str
    metadata: Dict[str, Any]
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    source: str = "hybrid"


class HybridRetriever:
    """
    Fuses BM25 sparse retrieval and dense vector retrieval using
    weighted Reciprocal Rank Fusion (RRF) for robust hybrid search.

    Weighted RRF formula:
        score(d) = bm25_weight * (1 / (k + rank_bm25(d)))
                 + vector_weight * (1 / (k + rank_vector(d)))
    """

    RRF_K = 60  # Standard RRF constant

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        """
        Args:
            bm25_retriever: Initialized BM25Retriever.
            vector_retriever: Initialized VectorRetriever.
            bm25_weight: Weight for BM25 contribution (0-1).
            vector_weight: Weight for vector contribution (0-1).
        """
        if abs(bm25_weight + vector_weight - 1.0) > 1e-6:
            raise ValueError("bm25_weight + vector_weight must sum to 1.0")

        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Run hybrid retrieval for a query.

        Returns:
            Top-k RetrievedChunks ranked by fused hybrid score.
        """
        fetch_k = max(top_k * 3, 15)  # Over-fetch for fusion

        # --- BM25 results ---
        bm25_results: List[Tuple[BM25Document, float]] = self.bm25.retrieve(
            query, top_k=fetch_k
        )
        # --- Vector results ---
        vector_results: List[Tuple[Document, float]] = self.vector.retrieve(
            query, top_k=fetch_k
        )

        # --- Score fusion ---
        fused = self._rrf_fusion(bm25_results, vector_results)

        # --- Return top_k ---
        top_results = fused[:top_k]
        logger.info(
            f"Hybrid retrieval: {len(bm25_results)} BM25, "
            f"{len(vector_results)} vector → {len(top_results)} fused results"
        )
        return top_results

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[BM25Document, float]],
        vector_results: List[Tuple[Document, float]],
    ) -> List[RetrievedChunk]:
        """Weighted Reciprocal Rank Fusion."""
        scores: Dict[str, Dict] = {}

        # Index BM25 by content
        for rank, (doc, score) in enumerate(bm25_results):
            key = self._content_key(doc.content)
            rrf_score = self.bm25_weight * (1.0 / (self.RRF_K + rank + 1))
            scores[key] = {
                "content": doc.content,
                "metadata": doc.metadata,
                "bm25_score": score,
                "vector_score": 0.0,
                "rrf": rrf_score,
            }

        # Merge vector results
        for rank, (doc, score) in enumerate(vector_results):
            key = self._content_key(doc.page_content)
            rrf_score = self.vector_weight * (1.0 / (self.RRF_K + rank + 1))
            if key in scores:
                scores[key]["vector_score"] = score
                scores[key]["rrf"] += rrf_score
            else:
                scores[key] = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "bm25_score": 0.0,
                    "vector_score": score,
                    "rrf": rrf_score,
                }

        # Build final results
        chunks = [
            RetrievedChunk(
                content=v["content"],
                metadata=v["metadata"],
                bm25_score=round(v["bm25_score"], 4),
                vector_score=round(v["vector_score"], 4),
                hybrid_score=round(v["rrf"], 6),
                source="hybrid" if v["bm25_score"] > 0 and v["vector_score"] > 0
                       else ("bm25" if v["bm25_score"] > 0 else "vector"),
            )
            for v in scores.values()
        ]

        return sorted(chunks, key=lambda c: c.hybrid_score, reverse=True)

    @staticmethod
    def _content_key(content: str) -> str:
        """Derive a short dedup key from content."""
        return content[:120].strip().lower()

    # ------------------------------------------------------------------
    # Convenience: single-mode retrieval for ablation
    # ------------------------------------------------------------------

    def retrieve_bm25_only(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Use only BM25 (for ablation studies)."""
        results = self.bm25.retrieve(query, top_k=top_k)
        return [
            RetrievedChunk(
                content=doc.content,
                metadata=doc.metadata,
                bm25_score=score,
                hybrid_score=score,
                source="bm25",
            )
            for doc, score in results
        ]

    def retrieve_vector_only(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Use only vector search (for ablation studies)."""
        results = self.vector.retrieve(query, top_k=top_k)
        return [
            RetrievedChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                vector_score=score,
                hybrid_score=score,
                source="vector",
            )
            for doc, score in results
        ]
