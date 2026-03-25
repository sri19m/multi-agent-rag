"""
Multi-Agent RAG Orchestration System — Main Entry Point

Usage:
    python main.py --query "What was Apple's revenue in Q3 2024?"
    python main.py --query "Summarize key risks" --vectorstore chroma --multi-strategy
    python main.py --query "Compare Apple and Microsoft margins" --crew-mode
"""

import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel

from src.agents.orchestrator import OrchestratorAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reward_ranker import RewardRanker
from src.llm.bedrock_client import BedrockClient
from src.retrieval.bm25_retriever import BM25Document, BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.vectorstore_factory import VectorstoreFactory

load_dotenv()
console = Console()
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s"
)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# System bootstrap
# ------------------------------------------------------------------

def bootstrap_system(config: dict, vectorstore_type: str) -> OrchestratorAgent:
    """
    Initialize the full multi-agent RAG system.
    Loads vectorstore, builds BM25 index, wires all agents.
    """
    persist_path = config["vectorstore"]["persist_path"]
    top_k = config["retrieval"]["top_k"]

    # 1. LLM & Embeddings
    console.print("[dim]Initializing AWS Bedrock client...[/dim]")
    bedrock = BedrockClient(
        model_id=config["llm"]["model_id"],
        embed_model_id=config["embeddings"]["model_id"],
        region=config["llm"]["region"],
        max_tokens=config["llm"]["max_tokens"],
        temperature=config["llm"]["temperature"],
    )
    llm = bedrock.chat_model

    # 2. Sample documents (fallback if no index exists)
    sample_chunks = _get_sample_chunks(config)

    # 3. Vectorstore
    console.print(f"[dim]Loading {vectorstore_type.upper()} vectorstore...[/dim]")
    vectorstore = VectorstoreFactory.create(
        store_type=vectorstore_type,
        embeddings=bedrock.embeddings,
        documents=sample_chunks if not _index_exists(persist_path, vectorstore_type) else None,
        persist_path=persist_path,
    )

    # 4. BM25 index
    console.print("[dim]Building BM25 index...[/dim]")
    bm25_docs = [
        BM25Document(content=c.page_content, metadata=c.metadata)
        for c in sample_chunks
    ]
    bm25_retriever = BM25Retriever()
    bm25_retriever.index_documents(bm25_docs)

    # 5. Hybrid retriever
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=VectorRetriever(vectorstore),
        bm25_weight=config["retrieval"]["bm25_weight"],
        vector_weight=config["retrieval"]["vector_weight"],
    )

    # 6. Agents
    reward_cfg = config["reward_ranker"]
    retrieval_agent = RetrievalAgent(
        hybrid_retriever=hybrid_retriever,
        llm=llm,
        top_k=top_k,
        verbose=config["agents"]["verbose"],
    )
    reasoning_agent = ReasoningAgent(
        llm=llm,
        use_financial_tools=True,
        use_web_search=False,
        verbose=config["agents"]["verbose"],
    )
    reward_ranker = RewardRanker(
        relevance_weight=reward_cfg["relevance_weight"],
        completeness_weight=reward_cfg["completeness_weight"],
        confidence_weight=reward_cfg["confidence_weight"],
        min_score_threshold=reward_cfg["min_score_threshold"],
    )

    return OrchestratorAgent(
        retrieval_agent=retrieval_agent,
        reasoning_agent=reasoning_agent,
        reward_ranker=reward_ranker,
        llm=llm,
        verbose=config["agents"]["verbose"],
    )


def _index_exists(persist_path: str, vectorstore_type: str) -> bool:
    if vectorstore_type == "faiss":
        return os.path.exists(os.path.join(persist_path, "faiss_index"))
    elif vectorstore_type == "chroma":
        return os.path.exists(os.path.join(persist_path, "chroma_db"))
    return False


def _get_sample_chunks(config: dict) -> list[Document]:
    """Return sample financial document chunks for demo/fallback."""
    raw_docs = [
        Document(
            page_content=(
                "Apple Inc. Q3 2024 Earnings Report. Revenue: $85.8 billion, up 4.9% YoY. "
                "EPS: $1.40 vs $1.35 expected. Gross margin: 46.3%. Services revenue: $24.2B (record). "
                "iPhone revenue: $39.3B. Mac revenue: $7.0B. iPad: $7.2B. Wearables: $8.1B. "
                "FY2025 guidance raised to $390-395B. Key risks: FX headwinds -$2.1B, supply chain pressure."
            ),
            metadata={"source": "apple_q3_2024.txt", "page": 1},
        ),
        Document(
            page_content=(
                "Microsoft Corp. Q3 2024: Revenue $61.9B, up 17% YoY. Azure cloud grew 21% YoY. "
                "EPS: $2.94 vs $2.82 expected. Operating income: $27.6B. Operating margin: 44.6%. "
                "AI services contributing ~4pp to Azure growth. LinkedIn revenue: $4.0B. "
                "Risks: regulatory scrutiny, FX exposure, cybersecurity threats."
            ),
            metadata={"source": "msft_q3_2024.txt", "page": 1},
        ),
        Document(
            page_content=(
                "Risk factors analysis FY2024: FX headwinds impact major tech companies. "
                "Regulatory risk: EU Digital Markets Act compliance costs for Apple and Microsoft. "
                "Supply chain risk: Semiconductor shortages affecting hardware product margins. "
                "Credit risk: Minimal for both companies given strong investment-grade ratings. "
                "Operational risk: Cloud infrastructure security and data privacy compliance."
            ),
            metadata={"source": "risk_summary.txt", "page": 1},
        ),
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["retrieval"]["chunk_size"],
        chunk_overlap=config["retrieval"]["chunk_overlap"],
    )
    return splitter.split_documents(raw_docs)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG Orchestration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "What was Apple's Q3 2024 revenue?"
  python main.py --query "Summarize all key risks" --multi-strategy
  python main.py --query "Compare Apple vs Microsoft margins" --vectorstore chroma
  python main.py --query "Analyze Azure growth drivers" --crew-mode
        """,
    )
    parser.add_argument("--query", required=True, help="Your financial question")
    parser.add_argument(
        "--vectorstore", default=None, choices=["faiss", "chroma"],
        help="Vectorstore backend (overrides config)"
    )
    parser.add_argument(
        "--multi-strategy", action="store_true",
        help="Run all retrieval strategies and rank responses"
    )
    parser.add_argument(
        "--crew-mode", action="store_true",
        help="Run full CrewAI multi-agent Crew pipeline"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of chunks to retrieve (overrides config)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose agent logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.vectorstore:
        config["vectorstore"]["type"] = args.vectorstore
    if args.top_k:
        config["retrieval"]["top_k"] = args.top_k
    if args.verbose:
        config["agents"]["verbose"] = True
        logging.getLogger().setLevel(logging.INFO)

    vectorstore_type = config["vectorstore"]["type"]

    console.rule("[bold blue]Multi-Agent RAG Orchestration System[/bold blue]")
    console.print(Panel(f"[bold]Query:[/bold] {args.query}", expand=False))
    console.print(
        f"  Vectorstore: [cyan]{vectorstore_type.upper()}[/cyan] | "
        f"Multi-strategy: [cyan]{args.multi_strategy}[/cyan] | "
        f"Crew mode: [cyan]{args.crew_mode}[/cyan]"
    )

    # Bootstrap
    console.print("\n[bold]Initializing system...[/bold]")
    try:
        orchestrator = bootstrap_system(config, vectorstore_type)
        orchestrator.multi_strategy = args.multi_strategy
        console.print("[green]✓[/green] System ready.\n")
    except Exception as e:
        console.print(f"[red]System initialization failed: {e}[/red]")
        console.print(
            "[yellow]Make sure AWS credentials are configured:\n"
            "  aws configure\n"
            "Or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env[/yellow]"
        )
        sys.exit(1)

    # Run query
    if args.crew_mode:
        console.print("[bold]Running CrewAI multi-agent pipeline...[/bold]")
        try:
            answer = orchestrator.run_as_crew(args.query, top_k=config["retrieval"]["top_k"])
            console.print(Panel(answer, title="[bold green]Final Answer (Crew)[/bold green]"))
        except Exception as e:
            console.print(f"[red]CrewAI pipeline failed: {e}[/red]")
    else:
        result = orchestrator.run(args.query, top_k=config["retrieval"]["top_k"])
        console.print(orchestrator.format_result(result))


if __name__ == "__main__":
    main()
