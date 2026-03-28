"""
LangGraph AgentState - the shared state passed between all graph nodes.

Using TypedDict with Annotated fields allows LangGraph to merge
list fields across parallel branches (add_messages_pattern).
"""

from __future__ import annotations

from typing import List, Literal

from langchain_core.documents import Document
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state for the RAG agent graph.

    Fields
    ------
    question: str
        The original user question.
    documents: List[Document]
        Documents retrieved (and graded) from the vector store.
    generation: str
        The LLM-generated answer.
    web_search_needed: bool
        Set to True when retrieved docs are insufficient.
    hallucination_score: Literal["yes", "no"]
        Whether the generation is grounded in retrieved documents.
    answer_addresses_question: Literal["yes","no"]
        Whether the generation actually answers the question.
    retry_count: int
        Number of self-correction cycles performed so far.
    """
    question: str
    documents: List[Document]
    generation:str
    web_search_needed: bool
    hallucination_score: Literal["yes", "no"]
    answer_addresses_question: Literal["yes", "no"]
    retry_count: int