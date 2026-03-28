"""
LangGraph node implementations.

Each function takes an AgentState and returns a partial state update.
LangGraph merges these updates automatically.
"""

from __future__ import annotations

from importlib.metadata import metadata
from multiprocessing.connection import answer_challenge
from typing import Any, Dict

from langchain_classic.chains.hyde.prompts import web_search
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agents import AgentState
from src.rag import DocumentGrader, RAGRetriever
from src.utils import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Shared singletons (lazy-initialized to avoid startup cost in tests)
# ─────────────────────────────────────────────────────────────────────────────
_retriever: RAGRetriever | None = None
_grader: DocumentGrader | None = None
_llm: ChatOpenAI | None = None
_search: DuckDuckGoSearchRun | None = None


def _get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def _get_grader()   -> DocumentGrader:
    global _grader
    if _grader is None:
        _grader = DocumentGrader()
    return _grader

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.2
        )
    return _llm


def _get_search() -> DuckDuckGoSearchRun:
    global _search
    if _search is None:
        _search = DuckDuckGoSearchRun()
    return _search

# ─────────────────────────────────────────────────────────────────────────────
# Node: retrieve
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(state:AgentState) -> Dict[str, Any]:
    """Retrieve relevant documents from the vector store."""
    logger.info("[Node] retrieve - question : %s", state["question"][:80])
    docs = _get_retriever().retrieve(state["question"])
    return {
        "documents":docs,
        "retry_count":state.get('retry_count',0)
    }



# ─────────────────────────────────────────────────────────────────────────────
# Node: grade_documents
# ─────────────────────────────────────────────────────────────────────────────
def grade_documents(state:AgentState) -> Dict[str, Any]:
    """
    Grade retrieved documents for relevance.
    If fewer than half pass grading, flag for web search fallback.
    """
    logger.info("[Node] grade_documents")
    question = state["question"]
    docs = state["documents"]

    relevant_docs = _get_grader().filter_relevant(question, docs)

    # Trigger web search if we have too few relevant docs
    web_search_needed = len(relevant_docs) < max(1,len(relevant_docs)//2)
    if web_search_needed:
        logger.info("  → Web search fallback triggered (insufficient relevant docs)")

    return {
        "documents":relevant_docs,
        "web_search_needed":web_search_needed
    }



# ─────────────────────────────────────────────────────────────────────────────
# Node: web_search
# ─────────────────────────────────────────────────────────────────────────────
def web_search(state:AgentState) -> Dict[str, Any]:
    """Augment document list with DuckDuckGo web search results."""
    logger.info("[Node] web_search")
    query = state["question"]
    search_result = _get_search().run(query)

    web_doc = Document(
        page_content=search_result,
        metadata={
            "source":"duckduckgo_search",
            "query":query
        }

    )

    # Merge web result with any surviving graded docs
    existing_docs = state.get('documents',[])
    return {
        "documents":existing_docs + [web_doc]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: generate
# ─────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """
You are an expert AI assistant specializing in machine learning, natural language processing, and AI engineering.

Answer the user's  question using ONLY the provided context documents.
If the context is insufficient, say so honestly.

Guidelines:
- Be precise and technical when appropriate
- Cite relevant details from the documents
- Structure your answer clearly
- Do not hallucinate facts not present in the context
"""

RAG_HUMAN_PROMPT = """
Context Documents: {context}

Question: {question}

Answer: """

def generate(state:AgentState) -> Dict[str,Any]:
    """Generate an answer from retrieved documents using the LLM."""
    logger.info("[Node] generate")
    docs = state["documents"]
    question = state["question"]

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknow')}]\n{doc.page_content}"
        for doc in docs
    )

    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system",RAG_SYSTEM_PROMPT),
                ("human", RAG_HUMAN_PROMPT)
            ]
        )
        | _get_llm()
        | StrOutputParser()
    )

    generation = chain.invoke({
        "context":context,
        "question":question
    })
    logger.info("  → Generated %d chars", len(generation))
    return {"generation":generation}


# ─────────────────────────────────────────────────────────────────────────────
# Node: check_hallucination
# ─────────────────────────────────────────────────────────────────────────────
class HallucinationScore(BaseModel):
    grounded: bool = Field(
        description="True if the answer is fully grounded in the provided documents."
    )

class AnswerScore(BaseModel):
    addresses_question:bool = Field(
        description="True if the answer actually addresses the user's question."
    )
    
    
def check_hallucination(state:AgentState) -> Dict[str,Any]:
    """Check whether the generation is grounded in retrieved documents."""
    logger.info("[Node] check_hallucination")
    llm = _get_llm()
    docs = state["documents"]
    generation = state["generation"]

    context = "\n\n".join(d.page_content[:500] for d in docs)

    # Check 1: Is the answer grounded in the documents?
    hallucination_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Assess whether the answer is supported yb the provided documents. Respond with JSON."),
                ("human","Documents:\n{context}\n\nAnswer:\n{generation}\n\nIs the answer grounded in these documents?")
            ]
        ) | llm.with_structured_output(HallucinationScore)
    )
    h_score = hallucination_chain.invoke({"context":context, "generation":generation})

    # Check 2 : Does the answer address the question
    answer_chain = ChatPromptTemplate.from_messages(
        [
            ("system","Assess whether the answer addresses the question. Respond with JSON."),
            ("human", "Question:: {question}\n\nAnswer: {generation}\n\nDoes the answer address the question?"),
        ]
    ) | llm.with_structured_output(AnswerScore)

    a_score = answer_chain.invoke(
        {
            "question":state["question"],
            "generation":generation
        }
    )

    hallucination_score = "yes" if h_score.grounded else "no"
    answer_addresses = "yes" if a_score.addresses_question else "no"

    logger.info(
        "  → grounded=%s, addresses_question=%s",
        hallucination_score,
        answer_addresses,
    )

    return {
        "hallucination_score":hallucination_score,
        "answer_addresses_question":answer_addresses,
        "retry_count":state.get("retry_count",0)+1,
    }