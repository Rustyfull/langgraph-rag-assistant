"""
FastAPI REST API for the LangGraph RAG Assistant.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents import run_query
from src.rag import run_ingestion, vectorstore_exists
from src.utils import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app:FastAPI) -> AsyncGenerator:
    logger.info("Starting LangGraph RAG Assistant API...")
    if not vectorstore_exists():
        logger.info("No vector store found — running initial ingestion...")
        run_ingestion()
    logger.info("API ready ✓")
    yield
    logger.info("Shutting down...")
    
    
    
# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app =  FastAPI(
    title="LangGraph RAG Assistant",
    description="Production RAG system with LangGraph stateful workflows",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(...,min_length=3,max_length=1000, description="User question")

class SourceDocument(BaseModel):
    content:str
    source: str
    source_type: str



class QueryResponse(BaseModel):
    question:str
    answer:str
    sources:List[SourceDocument]
    hallucination_score: str
    retry_count: int
    latency_ms: float



class HealthResponse(BaseModel):
    status: str
    model: str
    vectorstore_ready: bool
    

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check()    -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model=settings.openai_model,
        vectorstore_ready=vectorstore_exists()
    )


@app.post("/query",response_model=QueryResponse, tags=["RAG"])
async  def query(request:QueryRequest)  -> QueryResponse:
    """
    Run a question through the RAG pipeline.

    The LangGraph workflow will:
    1. Retrieve relevant documents
    2. Grade their relevance
    3. Optionally fall back to web search
    4. Generate an answer
    5. Validate for hallucinations
    """
    logger.info("POST /query — question: %s", request.question[:80])
    start = time.perf_counter()

    try:
        state = run_query(request.question)
    except Exception as exc:
        logger.info("Graph execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - start)*1000

    sources = [
        SourceDocument(
            content=doc.page_content[:300],
            source=doc.metadata.get("source", "unknow"),
            source_type=doc.metadata.get("source_type", "Unknow")

        )
        for doc in state.get("documents",[])
    ]
    return QueryResponse(
        question=request.question,
        answer=state.get("generation",""),
        sources=sources,
        hallucination_score=state.get("hallucination_score","unknow"),
        retry_count=state.get("retry_count",0),
        latency_ms=round(latency_ms,2)
    )


@app.post("/ingest",tags=["Admin"])
async def trigger_ingestion(background_tasks:BackgroundTasks, force:bool = False):
    """
        Trigger document re-ingestion in the background.

        Args:
            force: If True, re-ingest even if vector store already exists.
    """
    background_tasks.add_task(run_ingestion,force=force)
    return {
        "message":"Ingestion started in background",
        "force":force
    }