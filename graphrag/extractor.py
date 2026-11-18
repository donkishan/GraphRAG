"""Turn text chunks into nodes + edges.

Three layers, from most to least deterministic:
  1. Seed triples      - curated by a domain expert (data/seed_triples.json)
  2. Rule extraction   - regex for cross-references ("Section 16(2)"), forms ("GSTR-3B"), known entity mentions
  3. LLM extraction    - schema-constrained JSON from Groq/Ollama for anything new
"""
from __future__ import annotations

import json
import re

from .config import settings
from .loaders import Chunk
from .schema import NODE_TYPES, REL_TYPES, Edge, Node, make_id

SECTION_REF = re.compile(r"\bSections?\s+(\d{1,3}[A-Z]{0,3}(?:\([0-9a-z]{1,4}\))*)")
FOREIGN_ACT = re.compile(r"^\s*(?:of|under)\s+the\s+(IGST|MSMED|Companies|SGST|UTGST)", re.I)
FORM_REF = re.compile(r"\b(GSTR-\d{1,2}[A-Z]?|CMP-\d{2}|Form\s+\d{1,2}[A-Z]{0,3})\b")


# ---------------------------------------------------------------- 1. seed
def load_seed(path=None) -> tuple[list[Node], list[Edge]]:
    data = json.loads((path or settings.seed_file).read_text())
    key2id, nodes = {}, []
    for n in data["nodes"]:
        nid = make_id(n["type"], n["name"], n.get("act", ""))
        key2id[n["k"]] = nid
        nodes.append(Node(nid, n["type"], n["name"], n.get("desc", ""), n.get("act", "")))
    edges = [Edge(key2id[e[0]], key2id[e[2]], e[1], e[3] if len(e) > 3 else "", "seed")
             for e in data["edges"]]
    return nodes, edges


# ---------------------------------------------------------------- 2. rules
def chunk_anchor(c: Chunk) -> Node:
    """The node a chunk is 'about' (taken from its header)."""
    label = c.label
    if label.lower().startswith("notification"):
        return Node(make_id("Notification", label), "Notification", label, c.title)
    if label.lower().startswith("rule"):
        return Node(make_id("Rule", label, c.act), "Rule", label, c.title, c.act)
    if label.lower() == "act":
        return Node(make_id("Act", c.act), "Act", c.act, c.title)
    if label.lower().startswith("section"):
        return Node(make_id("Section", label, c.act), "Section", label, c.title, c.act)
    return Node(make_id("Concept", c.heading), "Concept", c.heading[:80], c.title)


def resolve_section(ref: str, act: str, known: set[str]) -> str:
    """'16(2)(c)' -> most specific node that exists: 16(2)(c) -> 16(2) -> 16."""
    cand = ref
    while True:
        nid = make_id("Section", f"Section {cand}", act)
        if nid in known or "(" not in cand:
            return nid
        cand = cand[: cand.rfind("(")]


def rule_extract(chunks: list[Chunk], known_nodes: list[Node]) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    known_ids = {n.id for n in known_nodes}
    # long names first so "Section 194C(6)" is matched before "Section 194C"
    mentionable = sorted([n for n in known_nodes if n.type in {"Party", "Concept", "Form", "Notification"}],
                         key=lambda n: -len(n.name))

    for c in chunks:
        anchor = chunk_anchor(c)
        nodes.append(anchor)
        known_ids.add(anchor.id)
        nodes.append(Node(c.id, "Chunk", c.heading[:120], c.text[:200], c.act,
                          props={"text": c.text, "doc": c.doc}))
        edges.append(Edge(anchor.id, c.id, "SOURCE", source_doc=c.doc))
        if anchor.type in {"Section", "Rule"} and c.act:
            act_id = make_id("Act", c.act)
            nodes.append(Node(act_id, "Act", c.act))
            edges.append(Edge(anchor.id, act_id, "PART_OF", source_doc=c.doc))

        # cross references inside the same Act
        for m in SECTION_REF.finditer(c.text):
            if FOREIGN_ACT.match(c.text[m.end(): m.end() + 30]):
                continue
            tgt = resolve_section(m.group(1), c.act, known_ids)
            if tgt == anchor.id:
                continue
            if tgt not in known_ids:
                nodes.append(Node(tgt, "Section", f"Section {m.group(1).split('(')[0]}", "", c.act))
                known_ids.add(tgt)
            snippet = c.text[max(0, m.start() - 60): m.end() + 40].replace("\n", " ").strip()
            edges.append(Edge(anchor.id, tgt, "REFERS_TO", snippet, c.doc))

        # forms
        for f in {m.group(1) for m in FORM_REF.finditer(c.text)}:
            fid = make_id("Form", f)
            if fid not in known_ids:
                nodes.append(Node(fid, "Form", f))
                known_ids.add(fid)
            edges.append(Edge(anchor.id, fid, "REQUIRES_FORM", f"{f} mentioned in {c.label}", c.doc))

        # mentions of known parties/concepts => provenance link to the chunk
        low = c.text.lower()
        for n in mentionable:
            if n.name.lower() in low:
                edges.append(Edge(n.id, c.id, "SOURCE", source_doc=c.doc))
    return nodes, edges


# ---------------------------------------------------------------- 3. LLM
EXTRACT_SYSTEM = f"""You are a legal knowledge-graph extraction engine for Indian tax law (GST, Income-tax).
Extract entities and relationships from the provision text.
Allowed entity types: {", ".join(t for t in NODE_TYPES if t != "Chunk")}.
Allowed relationship types: {", ".join(f"{k} ({v})" for k, v in REL_TYPES.items() if k != "SOURCE")}.
Rules:
- Use ONLY the allowed types. Never invent facts that are not in the text.
- Section names look like "Section 16(2)"; always set "act" for Section/Rule.
- Put exact numbers (rates, limits, deadlines) into Threshold entities.
- "evidence" must be a short quote (<= 20 words) from the text.
Return JSON: {{"entities":[{{"name":"","type":"","act":"","description":""}}],
"relations":[{{"source":"","source_type":"","target":"","target_type":"","type":"","evidence":""}}]}}"""


def llm_extract(chunk: Chunk, llm) -> tuple[list[Node], list[Edge]]:
    user = f"ACT: {chunk.act}\nPROVISION: {chunk.label} - {chunk.title}\nTEXT:\n{chunk.text[:6000]}"
    data = llm.chat_json(EXTRACT_SYSTEM, user)
    nodes, edges, name2id = [], [], {}

    def nid(name: str, typ: str, act: str = "") -> str | None:
        if typ not in NODE_TYPES or typ == "Chunk" or not name:
            return None
        act = act or (chunk.act if typ in {"Section", "Rule"} else "")
        return make_id(typ, name, act)

    for e in data.get("entities", []):
        i = nid(e.get("name", ""), e.get("type", ""), e.get("act", ""))
        if i:
            act = e.get("act") or (chunk.act if e["type"] in {"Section", "Rule"} else "")
            nodes.append(Node(i, e["type"], e["name"], e.get("description", "")[:300], act))
            name2id[(e["name"].lower(), e["type"])] = i
            edges.append(Edge(i, chunk.id, "SOURCE", source_doc=chunk.doc))

    for r in data.get("relations", []):
        if r.get("type") not in REL_TYPES or r["type"] == "SOURCE":
            continue
        s = name2id.get((str(r.get("source", "")).lower(), r.get("source_type"))) or nid(r.get("source", ""), r.get("source_type", ""))
        t = name2id.get((str(r.get("target", "")).lower(), r.get("target_type"))) or nid(r.get("target", ""), r.get("target_type", ""))
        if s and t and s != t:
            edges.append(Edge(s, t, r["type"], str(r.get("evidence", ""))[:200], chunk.doc))
    return nodes, edges
