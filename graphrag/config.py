"""Central configuration. Everything is read from .env so students never edit code to switch backends."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    # paths
    root: Path = ROOT
    raw_dir: Path = ROOT / "data" / "raw"
    seed_file: Path = ROOT / "data" / "seed_triples.json"
    index_dir: Path = ROOT / "data" / "index"

    # llm
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_models: list[str] = field(default_factory=lambda: _list(
        "GROQ_MODELS",
        "openai/gpt-oss-120b,openai/gpt-oss-20b,llama-3.3-70b-versatile,llama-3.1-8b-instant"))
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # graph
    graph_backend: str = os.getenv("GRAPH_BACKEND", "networkx").lower()
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password123")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # embeddings / retrieval
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k_chunks: int = int(os.getenv("TOP_K_CHUNKS", 4))
    top_k_entities: int = int(os.getenv("TOP_K_ENTITIES", 6))
    graph_hops: int = int(os.getenv("GRAPH_HOPS", 2))
    max_facts: int = int(os.getenv("MAX_FACTS", 60))


settings = Settings()
settings.index_dir.mkdir(parents=True, exist_ok=True)
