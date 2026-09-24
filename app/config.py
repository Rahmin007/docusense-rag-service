from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DocuSense"
    environment: str = "development"
    docs_dir: str = "data/docs"

    # Embeddings: openai, sentence_transformer, or tfidf
    embedding_provider: str = "tfidf"
    openai_embedding_model: str = "text-embedding-3-small"
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM: openai, anthropic, or extractive
    llm_provider: str = "extractive"
    openai_chat_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_chat_model: str = "claude-sonnet-5"
    anthropic_api_key: str | None = None

    top_k: int = 4
    similarity_threshold: float = 0.20
    chunk_size: int = 800
    chunk_overlap: int = 120
    snippet_chars: int = 260

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def resolved_docs_dir(self) -> Path:
        return Path(self.docs_dir)
