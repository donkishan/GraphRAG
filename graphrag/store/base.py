"""Common interface so the rest of the code never cares which graph DB is behind it."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schema import Edge, Node

# Hub nodes (an Act connects to every section) would flood the context if we walked through them.
NO_TRAVERSE_TYPES = {"Act"}
SKIP_RELS = {"SOURCE"}


@dataclass
class Fact:
    source: dict        # node dict
    rel: str
    target: dict
    evidence: str = ""
    hop: int = 1

    def render(self) -> str:
        def fmt(n: dict) -> str:
            act = f" [{n['act']}]" if n.get("act") else ""
            return f"{n['name']}{act} ({n['type']})"
        ev = f'  // "{self.evidence}"' if self.evidence else ""
        return f"{fmt(self.source)} -[{self.rel}]-> {fmt(self.target)}{ev}"


class GraphStore(ABC):
    @abstractmethod
    def upsert_nodes(self, nodes: list[Node]) -> None: ...

    @abstractmethod
    def upsert_edges(self, edges: list[Edge]) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> dict | None: ...

    @abstractmethod
    def all_nodes(self) -> list[dict]: ...

    @abstractmethod
    def all_edges(self) -> list[dict]: ...

    @abstractmethod
    def expand(self, seed_ids: list[str], hops: int, max_facts: int) -> list[Fact]:
        """k-hop neighbourhood of the seed entities, nearest facts first."""

    @abstractmethod
    def clear(self) -> None: ...

    def save(self) -> None:  # NetworkX persists to disk; Neo4j is already persistent
        pass

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        nodes, edges = self.all_nodes(), self.all_edges()
        by_type: dict[str, int] = {}
        for n in nodes:
            by_type[n["type"]] = by_type.get(n["type"], 0) + 1
        return {"nodes": len(nodes), "edges": len(edges), "by_type": by_type}
