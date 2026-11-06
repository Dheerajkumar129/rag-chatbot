from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "FastAPI RAG Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # OpenAI API
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Vector Store & Storage
    VECTOR_STORE_TYPE: str = "chroma"
    BASE_DIR: Path = Path(__file__).resolve().parent
    CHROMA_PERSIST_DIRECTORY: str = str(Path(__file__).resolve().parent / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # Chunking & Retrieval Parameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 4
    SIMILARITY_THRESHOLD: float = 0.25

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
