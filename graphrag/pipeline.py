"""The GraphRAG engine: build the graph, retrieve a sub-graph for a question, answer with citations,
and check the answer is grounded. Also contains a plain-RAG baseline for side-by-side comparison."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .config import settings
from .embeddings import VectorIndex, embed
from .extractor import chunk_anchor, llm_extract, load_seed, rule_extract
from .loaders import Chunk, load_dir, load_path
from .schema import NODE_TYPES, REL_TYPES, make_id
from .store import Fact, get_store

EXPLICIT_REF = re.compile(
    r"\b(?:section|sec\.?|s\.)\s*(\d{1,3}[A-Z]{0,3}(?:\([0-9a-z]{1,4}\))*)|\b(GSTR-\d{1,2}[A-Z]?|Form\s+\d{1,2}[A-Z]{0,3})\b",
    re.I)

ANSWER_SYSTEM = """You are a careful Indian tax-compliance assistant (GST and Income-tax).
Answer ONLY from the FACTS (knowledge-graph relationships) and SOURCES (statute text) provided.
Rules:
1. After every claim, cite its support like [F3] or [S2]. A claim with no citation is not allowed.
2. If the context does not contain the answer, say exactly: "Not found in the knowledge graph." Do not use outside knowledge.
3. Mention section numbers and their Act exactly as they appear in the context.
4. Format:
   **Short answer** (2-3 sentences)
   **Applicable provisions** (bullets)
   **Compliance steps** (numbered)
   **Watch-outs / exceptions** (bullets)
5. End with: "_Educational output - verify with a qualified professional._"
"""

CYPHER_SYSTEM = f"""You translate questions into a single READ-ONLY Neo4j Cypher query.
Graph schema:
- Every node has label :Entity plus one type label from: {", ".join(t for t in NODE_TYPES if t != "Chunk")}.
- Node properties: id, name, type, act, description. Section names look like 'Section 16(2)'; act is e.g. 'CGST Act 2017' or 'Income-tax Act 1961'.
- Relationship types: {", ".join(r for r in REL_TYPES if r != "SOURCE")}.
Rules: never use CREATE/MERGE/DELETE/SET/REMOVE/CALL. Match names case-insensitively with toLower(..) CONTAINS.
Return RELATIONSHIP-rich rows (names of both ends and type(r)). Always end with LIMIT 50.
Return JSON: {{"cypher": "...", "explanation": "..."}}"""


@dataclass
class Answer:
    question: str
    answer: str
    facts: list[Fact] = field(default_factory=list)
    sources: list[Chunk] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    latency_s: float = 0.0
    mode: str = "graphrag"


class GraphRAG:
    def __init__(self, backend: str | None = None, llm=None, embed_fn=embed):
        self.store = get_store(backend)
        self.entity_index = VectorIndex("entities", embed_fn)
        self.chunk_index = VectorIndex("chunks", embed_fn)
        self.chunks_path = settings.index_dir / "chunks.json"
        self.chunks: dict[str, Chunk] = {}
        if self.chunks_path.exists():
            self.chunks = {d["id"]: Chunk(**d) for d in json.loads(self.chunks_path.read_text())}
        self._llm = llm

    # ------------------------------------------------------------------ helpers
    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLM
            self._llm = LLM()
        return self._llm

    def _persist(self) -> None:
        self.store.save()
        self.entity_index.save()
        self.chunk_index.save()
        self.chunks_path.write_text(json.dumps([c.__dict__ for c in self.chunks.values()]))

    # ------------------------------------------------------------------ BUILD
    def build(self, use_llm: bool = False, reset: bool = True, log=print) -> dict:
        if reset:
            self.store.clear()
            self.entity_index.clear()
            self.chunk_index.clear()
            self.chunks = {}
        chunks = load_dir(settings.raw_dir)
        log(f"Loaded {len(chunks)} provision chunks from {settings.raw_dir}")
        seed_nodes, seed_edges = load_seed()
        self.store.upsert_nodes(seed_nodes)
        log(f"Seed graph: {len(seed_nodes)} nodes, {len(seed_edges)} edges")
        self._ingest_chunks(chunks, seed_nodes, use_llm, log)
        self.store.upsert_edges(seed_edges)        # after chunk nodes exist
        self._index_entities()
        self._persist()
        st = self.store.stats()
        log(f"Done: {st['nodes']} nodes, {st['edges']} edges")
        return st

    def ingest_file(self, path, use_llm: bool = True, log=print) -> dict:
        chunks = load_path(path)
        log(f"{path.name}: {len(chunks)} chunks")
        known = [_as_node(n) for n in self.store.all_nodes()]
        self._ingest_chunks(chunks, known, use_llm, log)
        self._index_entities()
        self._persist()
        return self.store.stats()

    def _ingest_chunks(self, chunks: list[Chunk], known_nodes, use_llm: bool, log) -> None:
        nodes, edges = rule_extract(chunks, known_nodes)
        self.store.upsert_nodes(nodes)
        self.store.upsert_edges(edges)
        log(f"Rule extraction: {len(nodes)} nodes, {len(edges)} edges")
        if use_llm:
            for i, c in enumerate(chunks, 1):
                try:
                    n2, e2 = llm_extract(c, self.llm)
                    self.store.upsert_nodes(n2)
                    self.store.upsert_edges(e2)
                    log(f"  LLM [{i}/{len(chunks)}] {c.label}: +{len(n2)} entities, +{len(e2)} relations")
                except Exception as ex:   # one bad chunk should not kill ingestion
                    log(f"  LLM [{i}/{len(chunks)}] {c.label}: skipped ({ex})")
        for c in chunks:
            self.chunks[c.id] = c
        self.chunk_index.add([c.id for c in chunks], [f"{c.heading}\n{c.text}" for c in chunks])

    def _index_entities(self) -> None:
        nodes = self.store.all_nodes()
        self.entity_index.add(
            [n["id"] for n in nodes],
            [f"{n['name']} ({n['type']}{', ' + n['act'] if n.get('act') else ''}): {n.get('description', '')}"
             for n in nodes])

    # ------------------------------------------------------------------ RETRIEVE
    def link_entities(self, question: str) -> list[str]:
        """Map the question onto graph nodes: exact references first, then semantic matches."""
        seeds: list[str] = []
        names = {n["id"]: n for n in self.store.all_nodes()}
        for m in EXPLICIT_REF.finditer(question):
            if m.group(1):
                ref = m.group(1).lower()
                for nid, n in names.items():
                    if n["type"] == "Section" and n["name"].lower() == f"section {ref}":
                        seeds.append(nid)
            elif m.group(2):
                fid = make_id("Form", m.group(2))
                if fid in names:
                    seeds.append(fid)
        for nid, score in self.entity_index.search(question, settings.top_k_entities):
            if score >= 0.30 and nid in names and nid not in seeds:
                seeds.append(nid)
        # the provision behind the best matching text passage is also a good entry point
        for cid, score in self.chunk_index.search(question, 2):
            if cid in self.chunks and score >= 0.30:
                a = chunk_anchor(self.chunks[cid]).id
                if a in names and a not in seeds:
                    seeds.append(a)
        return seeds

    def retrieve(self, question: str) -> tuple[list[str], list[Fact], list[Chunk]]:
        seeds = self.link_entities(question)
        facts = self.store.expand(seeds, settings.graph_hops, settings.max_facts)
        # evidence passages: text of seed provisions first, then provisions reached by the walk, then vector hits
        ordered: list[str] = []
        touched = seeds + [f.source["id"] for f in facts] + [f.target["id"] for f in facts]
        for nid in touched:
            for cid in self.store.chunks_of(nid):
                if cid in self.chunks and cid not in ordered:
                    ordered.append(cid)
        vec = [cid for cid, _ in self.chunk_index.search(question, settings.top_k_chunks)]
        final = [c for c in vec[:2] if c in self.chunks]
        final += [c for c in ordered if c not in final]
        final += [c for c in vec if c not in final and c in self.chunks]
        return seeds, facts, [self.chunks[c] for c in final[: settings.top_k_chunks + 2]]

    # ------------------------------------------------------------------ ANSWER
    @staticmethod
    def build_context(facts: list[Fact], sources: list[Chunk]) -> str:
        f = "\n".join(f"[F{i}] {fact.render()}" for i, fact in enumerate(facts, 1)) or "(none)"
        s = "\n\n".join(f"[S{i}] {c.heading}\n{c.text[:1500]}" for i, c in enumerate(sources, 1)) or "(none)"
        return f"FACTS (knowledge graph):\n{f}\n\nSOURCES (statute text):\n{s}"

    def ask(self, question: str) -> Answer:
        t0 = time.time()
        seeds, facts, sources = self.retrieve(question)
        ctx = self.build_context(facts, sources)
        text = self.llm.chat(ANSWER_SYSTEM, f"{ctx}\n\nQUESTION: {question}")
        ans = Answer(question, text, facts, sources, seeds, mode="graphrag")
        ans.grounding = grounding_check(text, facts, sources, question)
        ans.latency_s = round(time.time() - t0, 2)
        return ans

    def ask_plain_rag(self, question: str) -> Answer:
        """Baseline: vector search over chunks only, no graph. Used to SHOW why GraphRAG helps."""
        t0 = time.time()
        sources = [self.chunks[c] for c, _ in self.chunk_index.search(question, settings.top_k_chunks)
                   if c in self.chunks]
        ctx = self.build_context([], sources)
        text = self.llm.chat(ANSWER_SYSTEM, f"{ctx}\n\nQUESTION: {question}")
        ans = Answer(question, text, [], sources, [], mode="plain_rag")
        ans.grounding = grounding_check(text, [], sources, question)
        ans.latency_s = round(time.time() - t0, 2)
        return ans

    # ------------------------------------------------------------------ TEXT2CYPHER
    def text_to_cypher(self, question: str) -> dict:
        if not hasattr(self.store, "run_readonly"):
            return {"error": "Text2Cypher needs GRAPH_BACKEND=neo4j"}
        out = self.llm.chat_json(CYPHER_SYSTEM, question)
        cypher = out.get("cypher", "")
        try:
            rows = self.store.run_readonly(cypher)
        except Exception as e:
            return {"cypher": cypher, "error": str(e)}
        summary = self.llm.chat(
            "Summarise these Neo4j query results to answer the question. Use only the rows. "
            "If rows are empty say the graph has no matching data.",
            f"QUESTION: {question}\nCYPHER: {cypher}\nROWS: {json.dumps(rows[:50], default=str)}")
        return {"cypher": cypher, "explanation": out.get("explanation", ""), "rows": rows, "summary": summary}


# ---------------------------------------------------------------------- guardrail
SECTION_MENTION = re.compile(r"\bSection\s+(\d{1,3}[A-Z]{0,3}(?:\([0-9a-z]{1,4}\))*)", re.I)


def grounding_check(answer: str, facts: list[Fact], sources: list[Chunk], question: str = "") -> dict:
    """Cheap, deterministic hallucination detector:
    - every [F#]/[S#] citation must exist
    - every 'Section X' the model mentions must appear somewhere in the retrieved context
    - share of sentences carrying a citation"""
    cites = re.findall(r"\[(F|S)(\d+)\]", answer)
    invalid = sorted({f"{k}{n}" for k, n in cites
                      if (k == "F" and not 1 <= int(n) <= len(facts)) or (k == "S" and not 1 <= int(n) <= len(sources))})
    ctx = " ".join([question] + [f.render() for f in facts] + [c.heading + " " + c.text for c in sources]).lower()
    mentioned = {m.group(1) for m in SECTION_MENTION.finditer(answer)}
    unsupported = sorted(s for s in mentioned if f"section {s.lower()}" not in ctx and f"sections {s.lower()}" not in ctx)
    body = re.split(r"_Educational output", answer)[0]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", body) if len(s.strip()) > 25 and not s.strip().startswith("**")]
    cited = [s for s in sentences if re.search(r"\[(F|S)\d+\]", s)]
    ratio = round(len(cited) / len(sentences), 2) if sentences else 0.0
    refused = "not found in the knowledge graph" in answer.lower()
    ok = not invalid and not unsupported and (ratio >= 0.6 or refused)
    return {"grounded": ok, "citation_coverage": ratio, "invalid_citations": invalid,
            "unsupported_sections": unsupported, "refused": refused}


def _as_node(d: dict):
    from .schema import Node
    return Node(d["id"], d["type"], d["name"], d.get("description", ""), d.get("act", ""))