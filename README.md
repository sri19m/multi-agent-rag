# 🤖 Multi-Agent RAG Orchestration System

A production-grade multi-agent Retrieval-Augmented Generation (RAG) framework built with **LangChain**, **CrewAI**, **AWS Bedrock**, and **hybrid retrieval** (BM25 + vector embeddings). Designed for large-scale financial document Q&A with dynamic agent delegation, reward-based response ranking, and retrieval ablation studies.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
OrchestratorAgent  ─── RL-inspired Reward Ranker
    │                         │
    ├── RetrievalAgent         │
    │     ├── BM25Retriever    │
    │     └── VectorRetriever  │
    │                         │
    └── ReasoningAgent ────────┘
          └── AWS Bedrock (Claude / Titan)
```

### Key Components
| Component | Description |
|-----------|-------------|
| `OrchestratorAgent` | Routes queries, delegates tasks, aggregates results |
| `RetrievalAgent` | Hybrid BM25 + dense vector retrieval with vectorstore swapping |
| `ReasoningAgent` | LLM-powered reasoning using AWS Bedrock |
| `RewardRanker` | RL-inspired heuristics to rank and select best responses |
| `AblationStudy` | Benchmarking suite to measure retrieval strategy impact |

---

## 📁 Project Structure

```
multi-agent-rag/
├── main.py                      # Entry point
├── config/
│   └── config.yaml              # Central configuration
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # Multi-agent task orchestration
│   │   ├── retrieval_agent.py   # Document retrieval agent
│   │   ├── reasoning_agent.py   # LLM-based reasoning agent
│   │   └── reward_ranker.py     # Response scoring & ranking
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # BM25 + vector fusion
│   │   ├── bm25_retriever.py    # Sparse BM25 retrieval
│   │   ├── vector_retriever.py  # Dense embedding retrieval
│   │   └── vectorstore_factory.py # FAISS / Chroma swap
│   ├── tools/
│   │   ├── financial_tools.py   # Financial data tools
│   │   └── web_search_tool.py   # Web search integration
│   ├── llm/
│   │   └── bedrock_client.py    # AWS Bedrock LLM client
│   └── evaluation/
│       └── ablation_study.py    # Retrieval ablation benchmarks
├── scripts/
│   ├── ingest_documents.py      # Document ingestion pipeline
│   └── run_ablation.py          # Run ablation study
├── tests/
│   ├── test_retrieval.py
│   ├── test_agents.py
│   └── test_reward_ranker.py
├── data/
│   └── sample_financial_docs/   # Place your PDFs/docs here
├── requirements.txt
├── .env.example
└── setup.py
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sri19m/multi-agent-rag.git
cd multi-agent-rag
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your AWS credentials and settings
```

### 5. AWS Bedrock Setup
Ensure your AWS credentials have Bedrock access:
```bash
aws configure
# OR set in .env:
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=us-east-1
```

Enable model access in AWS Bedrock console → Model Access → Enable `anthropic.claude-3-sonnet`, `amazon.titan-embed-text-v1`.

---

## 🚀 Usage

### Ingest Documents
```bash
python scripts/ingest_documents.py --input data/sample_financial_docs/ --vectorstore faiss
```

### Run the Agent System
```bash
python main.py --query "What was Apple's revenue growth in Q3 2024?" --vectorstore faiss
```

### Switch Vectorstore (FAISS ↔ Chroma)
```bash
python main.py --query "Summarize key risks in the earnings report" --vectorstore chroma
```

### Run Ablation Study
```bash
python scripts/run_ablation.py --output results/ablation_results.json
```

---

## 📊 Retrieval Ablation Results

| Strategy | Precision@5 | Factual Accuracy | Latency (ms) |
|----------|-------------|------------------|--------------|
| BM25 Only | 0.61 | 72.3% | 120 |
| Vector Only | 0.68 | 74.1% | 310 |
| **Hybrid (BM25 + Vector)** | **0.79** | **91.2%** | 380 |

> Hybrid retrieval improved factual accuracy by **~19%** over BM25-only baseline.

---

## 🔧 Configuration

Edit `config/config.yaml` to adjust:
- LLM model (`claude-3-sonnet`, `titan`, etc.)
- Embedding model
- BM25 weight vs vector weight in hybrid fusion
- Top-K retrieval count
- Reward ranker weights

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

---

## 📄 License
MIT License
