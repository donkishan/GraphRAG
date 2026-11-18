"""Pure-Python knowledge graph (zero install). Persisted as JSON in data/index/graph.json."""
from __future__ import annotations

import json
from collections import deque

import networkx as nx

from ..config import settings
from ..schema import Edge, Node
from .base import NO_TRAVERSE_TYPES, SKIP_RELS, Fact, GraphStore


class NetworkXStore(GraphStore):
    def __init__(self):
        self.path = settings.index_dir / "graph.json"
        self.g = nx.MultiDiGraph()
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.g = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")

    # ---------- writes ----------
    def upsert_nodes(self, nodes: list[Node]) -> None:
        for n in nodes:
            if n.id in self.g:
                cur = self.g.nodes[n.id]
                # keep the richer description when the same entity is seen twice
                if len(n.description) > len(cur.get("description", "")):
                    cur["description"] = n.description
                cur.update({k: v for k, v in n.props.items()})
            else:
                base = {k: v for k, v in n.to_dict().items() if k != "props"}
                self.g.add_node(n.id, **{**base, **n.props})

    def upsert_edges(self, edges: list[Edge]) -> None:
        for e in edges:
            if e.source not in self.g or e.target not in self.g:
                continue
            if self.g.has_edge(e.source, e.target, key=e.type):
                if e.evidence and not self.g[e.source][e.target][e.type].get("evidence"):
                    self.g[e.source][e.target][e.type]["evidence"] = e.evidence
                continue
            self.g.add_edge(e.source, e.target, key=e.type, **e.to_dict())

    # ---------- reads ----------
    def _node(self, nid: str) -> dict:
        d = dict(self.g.nodes[nid])
        d.setdefault("id", nid)
        return d

    def get_node(self, node_id: str) -> dict | None:
        return self._node(node_id) if node_id in self.g else None

    def all_nodes(self) -> list[dict]:
        return [self._node(n) for n in self.g.nodes if self.g.nodes[n].get("type") != "Chunk"]

    def all_edges(self) -> list[dict]:
        return [d for _, _, d in self.g.edges(data=True) if d.get("type") not in SKIP_RELS]

    def expand(self, seed_ids: list[str], hops: int = 2, max_facts: int = 60) -> list[Fact]:
        """Breadth-first walk (ignoring edge direction) so nearer facts come first."""
        facts: list[Fact] = []
        seen_edges: set[tuple] = set()
        visited = {s for s in seed_ids if s in self.g}
        queue = deque((s, 0) for s in visited)
        while queue and len(facts) < max_facts:
            nid, depth = queue.popleft()
            if depth >= hops:
                continue
            if depth > 0 and self.g.nodes[nid].get("type") in NO_TRAVERSE_TYPES:
                continue
            incident = [(nid, v, k, d) for _, v, k, d in self.g.out_edges(nid, keys=True, data=True)]
            incident += [(u, nid, k, d) for u, _, k, d in self.g.in_edges(nid, keys=True, data=True)]
            for u, v, k, d in incident:
                if k in SKIP_RELS or (u, v, k) in seen_edges:
                    continue
                if k == "PART_OF" and self.g.nodes[v].get("type") in NO_TRAVERSE_TYPES:
                    continue   # "Section X is part of Act Y" adds nothing; act is already on the node
                seen_edges.add((u, v, k))
                facts.append(Fact(self._node(u), k, self._node(v), d.get("evidence", ""), depth + 1))
                other = v if u == nid else u
                if other not in visited:
                    visited.add(other)
                    queue.append((other, depth + 1))
                if len(facts) >= max_facts:
                    break
        return facts

    def chunks_of(self, node_id: str) -> list[str]:
        if node_id not in self.g:
            return []
        return [v for _, v, k in self.g.out_edges(node_id, keys=True) if k == "SOURCE"]

    # ---------- admin ----------
    def clear(self) -> None:
        self.g = nx.MultiDiGraph()
        if self.path.exists():
            self.path.unlink()

    def save(self) -> None:
        self.path.write_text(json.dumps(nx.node_link_data(self.g, edges="links")))
