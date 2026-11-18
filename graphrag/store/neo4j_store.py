"""Neo4j (Community Edition, local) implementation of the same interface.

Every entity gets the label :Entity plus its type label (e.g. :Entity:Section) and a unique `id`.
"""
from __future__ import annotations

import re
from collections import defaultdict

from neo4j import GraphDatabase

from ..config import settings
from ..schema import NODE_TYPES, REL_TYPES, Edge, Node
from .base import NO_TRAVERSE_TYPES, SKIP_RELS, Fact, GraphStore

WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL\s+dbms|CALL\s+apoc|CALL\s+db\.)\b", re.I)


class Neo4jStore(GraphStore):
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        self.driver.verify_connectivity()
        self.db = settings.neo4j_database
        self._run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

    def _run(self, cypher: str, **params):
        records, _, _ = self.driver.execute_query(cypher, params, database_=self.db)
        return records

    # ---------- writes (batched with UNWIND) ----------
    def upsert_nodes(self, nodes: list[Node]) -> None:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for n in nodes:
            if n.type in NODE_TYPES:     # whitelist => safe to put the label in the query string
                by_type[n.type].append({**n.to_dict(), **n.props, "props": None})
        for label, rows in by_type.items():
            self._run(f"""
                UNWIND $rows AS r
                MERGE (n:Entity {{id: r.id}})
                SET n:{label}, n.type = r.type, n.name = r.name, n.act = r.act,
                    n.text = coalesce(r.text, n.text), n.doc = coalesce(r.doc, n.doc),
                    n.description = CASE WHEN size(coalesce(n.description,'')) >= size(r.description)
                                         THEN n.description ELSE r.description END
            """, rows=rows)

    def upsert_edges(self, edges: list[Edge]) -> None:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            if e.type in REL_TYPES:
                by_type[e.type].append(e.to_dict())
        for rel, rows in by_type.items():
            self._run(f"""
                UNWIND $rows AS r
                MATCH (a:Entity {{id: r.source}}), (b:Entity {{id: r.target}})
                MERGE (a)-[x:{rel}]->(b)
                SET x.evidence = CASE WHEN coalesce(x.evidence,'') = '' THEN r.evidence ELSE x.evidence END,
                    x.source_doc = r.source_doc
            """, rows=rows)

    # ---------- reads ----------
    @staticmethod
    def _props(node) -> dict:
        d = dict(node)
        return {k: d.get(k, "") for k in ("id", "type", "name", "description", "act")}

    def get_node(self, node_id: str) -> dict | None:
        rec = self._run("MATCH (n:Entity {id:$id}) RETURN n", id=node_id)
        return self._props(rec[0]["n"]) if rec else None

    def all_nodes(self) -> list[dict]:
        return [self._props(r["n"]) for r in self._run("MATCH (n:Entity) WHERE n.type <> 'Chunk' RETURN n")]

    def all_edges(self) -> list[dict]:
        return [dict(r) for r in self._run("""
            MATCH (a:Entity)-[r]->(b:Entity) WHERE type(r) <> 'SOURCE'
            RETURN a.id AS source, b.id AS target, type(r) AS type, r.evidence AS evidence""")]

    def expand(self, seed_ids: list[str], hops: int = 2, max_facts: int = 60) -> list[Fact]:
        hops = max(1, min(int(hops), 3))   # int-cast: variable-length bounds can't be parameters
        no_trav = list(NO_TRAVERSE_TYPES)
        recs = self._run(f"""
            MATCH p = (s:Entity)-[*1..{hops}]-(m:Entity)
            WHERE s.id IN $ids
              AND none(r IN relationships(p) WHERE type(r) IN $skip
                       OR (type(r) = 'PART_OF' AND endNode(r).type IN $no_trav))
              AND none(n IN nodes(p)[1..-1] WHERE n.type IN $no_trav)
            WITH p ORDER BY length(p) LIMIT $lim
            UNWIND range(0, length(p)-1) AS i
            WITH relationships(p)[i] AS r, i
            RETURN startNode(r) AS a, type(r) AS rel, endNode(r) AS b,
                   r.evidence AS ev, min(i)+1 AS hop
            ORDER BY hop LIMIT $maxf
        """, ids=seed_ids, skip=list(SKIP_RELS), no_trav=no_trav, lim=max_facts * 3, maxf=max_facts)
        return [Fact(self._props(r["a"]), r["rel"], self._props(r["b"]), r["ev"] or "", r["hop"]) for r in recs]

    def chunks_of(self, node_id: str) -> list[str]:
        return [r["id"] for r in self._run(
            "MATCH (:Entity {id:$id})-[:SOURCE]->(c:Entity) RETURN c.id AS id", id=node_id)]

    # ---------- Text2Cypher execution (read-only) ----------
    def run_readonly(self, cypher: str, limit: int = 50) -> list[dict]:
        if WRITE_KEYWORDS.search(cypher):
            raise ValueError("Blocked: only read-only Cypher is allowed.")
        if not re.search(r"\bLIMIT\b", cypher, re.I):
            cypher = cypher.rstrip().rstrip(";") + f" LIMIT {limit}"
        with self.driver.session(database=self.db, default_access_mode="READ") as s:
            return s.execute_read(lambda tx: [r.data() for r in tx.run(cypher)])

    # ---------- admin ----------
    def clear(self) -> None:
        self._run("MATCH (n:Entity) DETACH DELETE n")

    def close(self) -> None:
        self.driver.close()
