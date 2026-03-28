"""
Configuration management using Pydantic Settings.
All values are loaded from environment variables / .env file.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    # Ollama (local fallback)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    # LangSmith
    langchain_tracking_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="langgraph-rag-assistant", alias="LANGCHAIN_PROJECT"
    )

    # Vector Store
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="rag_documents",alias="COLLECTION_NAME")

    # Retrieval
    retriever_k: int = Field(default=4, alias="RETRIEVER_K")
    relevance_threshold: float = Field(default=0.7, alias="RELEVANCE_THRESHOLD")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # App
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    max_retries: int = Field(default=10, alias="MAX_RETRIES")
    
    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))
    
    
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton"""
    return Settings()