# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class RAGConfig(BaseSettings):
    CHILD_COLL: str = "the_batch_children"
    PARENT_COLL: str = "the_batch_parents"

    QDRANT_URL: str = "http://qdrant_db:6333"
    GR_TOKEN: str
    QDRANT_TIMEOUT: int = 30
    QDRANT_RETRIES: int = 3
    COLLECTION_NAME: str = "the_batch_production"

    LLM_MODEL_NAME: str = "qwen2.5:1.5b"  # qwen2.5:3b can be struggled
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    OlLAMA_URL_COLLAB: str = "http://host.docker.internal:11434"

    EMBED_MODEL_NAME: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    RERANKER_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    TEXT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    IMAGE_MODEL_NAME: str = "sentence-transformers/clip-ViT-B-32"
    VARIATIONS_LLAMA_MODEL_NAME = os.getenv("MODEL_NAME", "qwen-tun")

    SPARSE_MODEL_NAME: str = "Qdrant/bm25"

    CHUNK_SIZE_PARENTS: int = 2000
    CHUNK_OVERLAP_PARENTS: int = 200
    CHUNK_SIZE_CHILD: int = 500
    CHUNK_OVERLAP_CHILD: int = 50
    CHUNK_BATCH_SIZE: int = 128
    CHUNK_OVERLAP_BATCH_SIZE: int = 128
    TOP_P: float = 0.4  # use top 40% words
    REPEAT_PENALTY: float = 1.1  # penalty for repeated words
    THRESHOLD: float = 0.7

    CACHE_THRESHOLD: float = 0.95

    TEMPERATURE: float = 0.2
    TOP_K: int = 8
    NUM_PREDICT: int = 1024

    RERANKER_K: int = 40

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = RAGConfig()
