"""
Document ingestion pipeline

Loads real publicly available documents from:
    - ArXiv papers (via arxiv loader)
    - Web pages (via WebBaseLoader)

Then chunks and embeds them into ChromaDb
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import ArxivLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import get_settings, get_logger

load_dotenv()
logger = get_logger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Public sources indexed by this system
# ─────────────────────────────────────────────────────────────────────────────
ARXIV_PAPERS = [
    "1706.03762",
    "1810.04805",
    "2005.11401",
    "2312.10997"
]

WEB_SOURCES = [
    # Langgraph official docs
     "https://langchain-ai.github.io/langgraph/concepts/"
    # LangChain RAG tutorial
      "https://python.langchain.com/docs/tutorials/rag/",
    # ChromaDB docs
    "https://docs.trychroma.com/docs/overview/introduction",

]



def load_arxiv_papers(paper_ids:List[str]) -> List[Document]:
    """Load papers from ArXiv by ID."""
    docs: List[Document] = []
    for paper_id in paper_ids:
        try:
            logger.info(f"Loading ArXIv paper: {paper_id}")
            loader = ArxivLoader(query=paper_id, load_max_docs=1)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_type"] = "arxiv"
                doc.metadata["paper_id"] = paper_id
            docs.extend(loaded)
            logger.info(f"✓ Loader {len(loaded)} doc(s) for {paper_id}")
        except Exception as e:
            logger.warning(f"   x Failed to load {paper_id}: {e}")
    return docs


def load_web_pages(urls:List[str]) -> List[Document]:
    """Load web pages using WebBaseLoader."""
    docs: List[Document] = []
    for url in urls:
        try:
            logger.info(f"Loading web page: {url}")
            loader = WebBaseLoader(url)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_type"] = "web"
                doc.metadata["url"] = url
            docs.extend(loaded)
            logger.info(f"  ✓ Loaded {len(loaded)} doc(s) from {url}")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load {url}: {e}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks")
    return chunks

def get_embeddings() -> OpenAIEmbeddings:
    """Return the embedding model."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,

    )

def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    logger.info(f"Building vector store with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,

    )
    logger.info(f"  ✓ Vector store saved to '{settings.chroma_persist_dir}'")
    return vectorstore



def load_vectorstore() -> Chroma:
    """Load existing vector store from disk."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir
    )


def vectorstore_exists() -> bool:
    """Check if a persisted vector store already exists."""
    return os.path.isdir(settings.chroma_persist_dir) and bool(
        os.listdir(settings.chroma_persist_dir)
    )


def run_ingestion(force: bool = False) -> Chroma:
    """
    Full ingestion pipeline.
    
    Args:
         force: Re-ingest even if vector store already exists.

    Returns:
        Populated Chroma vector store.
    """
    if vectorstore_exists() and not force:
        logger.info("Vector store already exists - skipping ingestion. Use force=True to re-ingest.")
        return load_vectorstore()
    logger.info("=== Starting document ingestion ===")

    # 1. Load raw documents
    raw_docs: List[Document] = []
    raw_docs.extend(load_arxiv_papers(ARXIV_PAPERS))
    raw_docs.extend(load_web_pages(WEB_SOURCES))
    
    if not raw_docs:
        raise RuntimeError(
            "No documents were loaded. Check your internet connection and sources."
        )
    logger.info(f"Total raw documents loaded: {len(raw_docs)}")


    # 2. Chunk
    chunks = split_documents(raw_docs)

    # 3. Embed & persist
    vectorstore = build_vectorstore(chunks)

    logger.info("=== Ingestion complete ===")
    return vectorstore


if __name__ == "__main__":
    run_ingestion(force=True)
