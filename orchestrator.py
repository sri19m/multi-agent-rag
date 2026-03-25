"""
Orchestrator Agent — the central coordinator for the multi-agent RAG system.
Routes queries, delegates to specialized agents, applies reward ranking,
and returns the best final response.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from crewai import Agent, Crew, Process, Task
from langchain_core.language_models import BaseLanguageModel

from src.agents.reasoning_agent import ReasoningAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reward_ranker import AgentResponse, RewardRanker
from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Final result returned to the user after multi-agent processing."""
    query: str
    answer: str
    agent_name: str
    retrieval_mode: str
    chunks_used: List[RetrievedChunk]
    reward_score: float
    latency_ms: float
    all_responses: List[AgentResponse] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Query routing heuristics
# ------------------------------------------------------------------

RETRIEVAL_KEYWORDS = [
    "what", "who", "when", "how much", "how many", "revenue",
    "earnings", "profit", "loss", "margin", "eps", "guidance",
    "report", "filing", "quarter", "annual", "fiscal",
]

TOOL_KEYWORDS = [
    "current price", "latest news", "today", "real-time",
    "stock price", "compare", "recent", "this week",
]


class OrchestratorAgent:
    """
    Multi-agent orchestrator that:
    1. Classifies query intent (document RAG vs live tool use)
    2. Delegates to RetrievalAgent + ReasoningAgent pipeline
    3. Can run multiple retrieval strategies in parallel (ablation mode)
    4. Applies RewardRanker to select the best response
    5. Returns a structured OrchestratorResult

    Architecture:
        OrchestratorAgent
            ├── RetrievalAgent  (hybrid, bm25-only, vector-only)
            ├── ReasoningAgent  (LLM synthesis)
            └── RewardRanker    (response selection)
    """

    AGENT_ROLE = "Multi-Agent RAG Orchestrator"
    AGENT_GOAL = (
        "Coordinate retrieval and reasoning agents to produce the most accurate, "
        "complete, and well-supported answer to financial queries. "
        "Select the highest-quality response using reward heuristics."
    )
    AGENT_BACKSTORY = (
        "You are the chief research coordinator at a top-tier investment bank. "
        "You manage a team of specialized analysts and know exactly which expert "
        "to consult for each type of financial query. You synthesize diverse "
        "information sources and ensure the final answer is accurate and actionable."
    )

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        reasoning_agent: ReasoningAgent,
        reward_ranker: RewardRanker,
        llm: BaseLanguageModel,
        multi_strategy: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            retrieval_agent: Initialized RetrievalAgent.
            reasoning_agent: Initialized ReasoningAgent.
            reward_ranker: Initialized RewardRanker.
            llm: LLM for the orchestrator agent itself.
            multi_strategy: If True, run hybrid + BM25-only + vector-only
                            and rank all three responses (ablation mode).
            verbose: Print agent logs.
        """
        self.retrieval_agent = retrieval_agent
        self.reasoning_agent = reasoning_agent
        self.reward_ranker = reward_ranker
        self.llm = llm
        self.multi_strategy = multi_strategy
        self.verbose = verbose
        self._agent: Optional[Agent] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, query: str, top_k: int = 5) -> OrchestratorResult:
        """
        Process a query through the full multi-agent RAG pipeline.

        Args:
            query: Natural language question.
            top_k: Number of chunks to retrieve.

        Returns:
            OrchestratorResult with best answer and full diagnostics.
        """
        start = time.time()
        logger.info(f"\n{'='*60}\nOrchestrator received query: {query}\n{'='*60}")

        intent = self._classify_intent(query)
        logger.info(f"Query intent classified as: {intent}")

        if self.multi_strategy:
            result = self._run_multi_strategy(query, top_k, intent)
        else:
            result = self._run_single_strategy(query, top_k, intent, mode="hybrid")

        result.latency_ms = round((time.time() - start) * 1000, 2)
        logger.info(
            f"Orchestrator completed in {result.latency_ms}ms | "
            f"Reward score: {result.reward_score:.3f}"
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline variants
    # ------------------------------------------------------------------

    def _run_single_strategy(
        self, query: str, top_k: int, intent: str, mode: str = "hybrid"
    ) -> OrchestratorResult:
        """Standard single-pass pipeline: retrieve → reason → rank."""
        chunks = self.retrieval_agent.retrieve(query, mode=mode, top_k=top_k)
        context = self.retrieval_agent.format_context(chunks)

        response = self.reasoning_agent.reason(
            query=query,
            context_chunks=chunks,
            formatted_context=context,
        )

        ranked = self.reward_ranker.rank(query, [response])
        best = ranked[0]

        return OrchestratorResult(
            query=query,
            answer=best.content,
            agent_name=best.agent_name,
            retrieval_mode=mode,
            chunks_used=chunks,
            reward_score=best.total_reward,
            latency_ms=0.0,
            all_responses=ranked,
            metadata={
                "intent": intent,
                "strategy": "single",
                "retrieval_mode": mode,
                "num_chunks": len(chunks),
            },
        )

    def _run_multi_strategy(
        self, query: str, top_k: int, intent: str
    ) -> OrchestratorResult:
        """
        Multi-strategy pipeline: run hybrid, BM25-only, and vector-only
        in sequence, then rank all three responses with RewardRanker.
        The highest-scoring response is returned.
        """
        all_responses: List[AgentResponse] = []
        all_chunks: List[RetrievedChunk] = []

        for mode in ["hybrid", "bm25", "vector"]:
            logger.info(f"  → Running {mode} retrieval strategy...")
            chunks = self.retrieval_agent.retrieve(query, mode=mode, top_k=top_k)
            context = self.retrieval_agent.format_context(chunks)
            response = self.reasoning_agent.reason(
                query=query,
                context_chunks=chunks,
                formatted_context=context,
            )
            response.agent_name = f"ReasoningAgent[{mode}]"
            response.retrieval_source = mode
            all_responses.append(response)
            if mode == "hybrid":
                all_chunks = chunks  # Keep hybrid chunks for the result

        ranked = self.reward_ranker.rank(query, all_responses)
        best = ranked[0]

        if self.verbose:
            print(self.reward_ranker.get_score_report(ranked))

        return OrchestratorResult(
            query=query,
            answer=best.content,
            agent_name=best.agent_name,
            retrieval_mode=best.retrieval_source,
            chunks_used=all_chunks,
            reward_score=best.total_reward,
            latency_ms=0.0,
            all_responses=ranked,
            metadata={
                "intent": intent,
                "strategy": "multi",
                "winning_retriever": best.retrieval_source,
                "num_candidates": len(ranked),
            },
        )

    # ------------------------------------------------------------------
    # CrewAI-based full Crew pipeline (advanced usage)
    # ------------------------------------------------------------------

    def run_as_crew(self, query: str, top_k: int = 5) -> str:
        """
        Run the full multi-agent pipeline as a CrewAI Crew.
        This enables agent-to-agent delegation and richer orchestration.
        Use for complex multi-step analytical tasks.
        """
        # Step 1: Retrieve context
        chunks = self.retrieval_agent.retrieve(query, top_k=top_k)
        context = self.retrieval_agent.format_context(chunks)

        # Step 2: Create tasks
        retrieval_task = Task(
            description=(
                f"Retrieve and curate the most relevant context for this query:\n{query}\n\n"
                f"Retrieved context (pre-fetched):\n{context}"
            ),
            agent=self.retrieval_agent.agent,
            expected_output="Curated, relevant document context for the query.",
        )

        reasoning_task = Task(
            description=(
                f"Using the retrieved context, answer this financial question precisely:\n{query}"
            ),
            agent=self.reasoning_agent.agent,
            expected_output="A precise, data-backed answer with specific financial figures.",
            context=[retrieval_task],
        )

        orchestration_task = Task(
            description=(
                "Review the retrieval and reasoning outputs. "
                "Synthesize a final, high-quality answer. "
                "Ensure accuracy, completeness, and clear formatting."
            ),
            agent=self.agent,
            expected_output="Final polished answer ready for the end user.",
            context=[retrieval_task, reasoning_task],
        )

        crew = Crew(
            agents=[
                self.retrieval_agent.agent,
                self.reasoning_agent.agent,
                self.agent,
            ],
            tasks=[retrieval_task, reasoning_task, orchestration_task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        return str(result)

    # ------------------------------------------------------------------
    # CrewAI Agent (self)
    # ------------------------------------------------------------------

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(
                role=self.AGENT_ROLE,
                goal=self.AGENT_GOAL,
                backstory=self.AGENT_BACKSTORY,
                llm=self.llm,
                verbose=self.verbose,
                allow_delegation=True,
            )
        return self._agent

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, query: str) -> str:
        """
        Rule-based intent classifier.
        Returns: "retrieval" | "tool" | "hybrid_intent"
        """
        q_lower = query.lower()

        tool_match = sum(1 for kw in TOOL_KEYWORDS if kw in q_lower)
        retrieval_match = sum(1 for kw in RETRIEVAL_KEYWORDS if kw in q_lower)

        if tool_match > retrieval_match:
            return "tool"
        elif retrieval_match > 0:
            return "retrieval"
        else:
            return "hybrid_intent"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def format_result(self, result: OrchestratorResult) -> str:
        """Pretty-print an OrchestratorResult for CLI output."""
        lines = [
            f"\n{'='*60}",
            f"QUERY:   {result.query}",
            f"AGENT:   {result.agent_name}",
            f"MODE:    {result.retrieval_mode}",
            f"SCORE:   {result.reward_score:.3f}",
            f"LATENCY: {result.latency_ms}ms",
            f"CHUNKS:  {len(result.chunks_used)}",
            f"{'='*60}",
            f"\nANSWER:\n{result.answer}",
            f"\n{'='*60}",
        ]
        return "\n".join(lines)
