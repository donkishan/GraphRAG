from ..config import settings
from .base import Fact, GraphStore


def get_store(backend: str | None = None) -> GraphStore:
    backend = (backend or settings.graph_backend).lower()
    if backend == "neo4j":
        from .neo4j_store import Neo4jStore
        return Neo4jStore()
    from .networkx_store import NetworkXStore
    return NetworkXStore()


__all__ = ["get_store", "Fact", "GraphStore"]
