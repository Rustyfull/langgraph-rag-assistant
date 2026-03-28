"""
ChromaDB retriever wrapper with similarity scoring.
"""

from __future__ import annotations

from typing import Tuple, List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.rag.ingest import get_embeddings, load_vectorstore, vectorstore_exists, run_ingestion
from src.utils.config import  get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGRetriever:
    """
    Thin wrapper around ChromaDB with relevance scaring.

    Usage:
        retriever = RAGRetriever()
        docs = retriever.retrieve("What is attention mechanism?")
    """
    def __init__(self, vectorstore:Chroma | None = None)    -> None:
        if vectorstore is not None:
            self._vs = vectorstore
        elif vectorstore_exists():
            self._vs = load_vectorstore()
        else:
            logger.warning("No vector store found - running ingestion...")
            self._vs = run_ingestion()

    def retrieve(self, query:str, k:int | None = None) -> List[Document]:
        """Retrieve top-k relevant documents for a query"""
        k = k or settings.retriever_k
        retriever = self._vs.as_retriever(
                search_type="mmr", # Maximal Marginal Relevance for Diversity,
                search_kwargs={
                    "k":k,
                    "fetch_k":k*3
                }
        )
        docs = retriever.invoke(query)
        logger.debug(f"Retrieved {len(docs)} documents for query: '{query[:60]}...'")
        return docs


    def retrieve_with_score(
            self,
            query: str,
            k:int | None = None
                            ) -> List[Tuple[Document, float]]:
        """Retrieve documents with cosine similarity scores."""
        k = k or settings.retriever_k
        results = self._vs.similarity_search_with_relevance_scores(query,k=k)
        return results


    def add_documents(
            self,
            docs: List[Document]
    ) -> None:
        """Add new documents to the vector store at runtime."""
        self._vs.add_documents(docs)
        logger.info(f"Added {len(docs)} new documents to the vector store")