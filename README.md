# 🧠 LangGraph RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system powered by **LangGraph** for stateful, multi-step AI workflows. This project implements an intelligent document Q&A assistant with agentic reasoning, self-correction, and adaptive retrieval strategies.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                    │
│                                                         │
│  User Query ──► [Route Query] ──► [Retrieve Docs]      │
│                      │                   │              │
│                      │            [Grade Docs]          │
│                      │                   │              │
│               [Web Search]        [Generate Answer]     │
│                      │                   │              │
│                      └──────────► [Hallucination Check] │
│                                          │              │
│                                   [Final Answer]        │
└─────────────────────────────────────────────────────────┘
```

The system uses a **Corrective RAG (CRAG)** pattern with the following nodes:
- **Router**: Decides whether to use vector store or web search
- **Retriever**: Fetches relevant documents from ChromaDB
- **Grader**: Filters irrelevant retrieved documents
- **Generator**: Produces answers using an LLM
- **Hallucination Checker**: Validates that answers are grounded in facts
- **Web Search Fallback**: DuckDuckGo search when docs are insufficient

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph 0.2+ |
| LLM | OpenAI GPT-4o / Ollama (local) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB |
| Document Loading | LangChain loaders (PDF, Web, Arxiv) |
| Web Search | DuckDuckGo Search |
| API Server | FastAPI |
| Testing | Pytest |
| Observability | LangSmith |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key (or Ollama for local models)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/langgraph-rag-assistant.git
cd langgraph-rag-assistant

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Ingest Documents

```bash
python -m src.rag.ingest
```

### 4. Run the Assistant

**CLI mode:**
```bash
python -m src.main
```

**API Server:**
```bash
uvicorn src.api:app --reload --port 8000
```

### 5. Query the System

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is attention mechanism in transformers?"}'
```

---

## 📂 Project Structure

```
langgraph-rag-assistant/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py          # LangGraph workflow definition
│   │   ├── nodes.py          # Graph node implementations
│   │   ├── state.py          # AgentState TypedDict
│   │   └── edges.py          # Conditional edge logic
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py         # Document ingestion pipeline
│   │   ├── retriever.py      # ChromaDB vector store wrapper
│   │   └── grader.py         # Document relevance grader
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   └── logger.py         # Structured logging
│   ├── api.py                # FastAPI application
│   └── main.py               # CLI entry point
├── tests/
│   ├── test_graph.py
│   ├── test_rag.py
│   └── test_api.py
├── data/                     # Local document storage
├── docs/                     # Architecture diagrams
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🔧 Configuration

All configuration is managed via environment variables (see `.env.example`):

```env
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls-...        # Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true       # Optional: enable tracing
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=rag_documents
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 📖 Documents Indexed

The system comes pre-configured to ingest:
- **Attention Is All You Need** (Vaswani et al., 2017) — Arxiv
- **BERT** (Devlin et al., 2018) — Arxiv
- **LangGraph documentation** — Official docs
- **RAG paper** (Lewis et al., 2020) — Arxiv

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
