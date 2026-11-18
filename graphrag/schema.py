"""The ontology: the fixed vocabulary of node types and relationship types.

Teaching point: a GraphRAG system is only as good as its schema. Restricting the LLM
to THESE labels during extraction is what makes the graph deterministic and queryable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

NODE_TYPES = {
    "Act":          "A statute, e.g. CGST Act 2017, Income-tax Act 1961",
    "Section":      "A numbered provision of an Act, e.g. Section 16, Section 194C",
    "Rule":         "A rule framed under an Act, e.g. Rule 36(4) CGST Rules",
    "Notification": "A government notification / circular",
    "Form":         "A return / form / certificate, e.g. GSTR-3B, Form 26Q",
    "Concept":      "A legal concept, e.g. Input Tax Credit, Reverse Charge, TDS",
    "Party":        "A type of person / business, e.g. Goods Transport Agency, Composition Dealer",
    "Threshold":    "A numeric limit, rate or deadline, e.g. Rs 20 lakh turnover, 30% disallowance",
    "Chunk":        "A raw text passage (source evidence)",
}

REL_TYPES = {
    "PART_OF":          "Section/Rule -> Act",
    "REFERS_TO":        "Provision cross-references another provision",
    "AMENDS":           "Provision/Notification/Act amends another",
    "REPLACES":         "New Act/provision supersedes an old one",
    "RESTRICTS":        "Provision blocks or limits another (e.g. 17(5) restricts 16)",
    "EXCEPTION_TO":     "Provision carves an exception out of another",
    "DEFINES":          "Provision defines a Concept",
    "GOVERNS":          "Provision governs a Concept",
    "APPLIES_TO":       "Provision/Concept applies to a Party",
    "LIABLE_UNDER":     "Party is liable/obligated under a Provision",
    "REQUIRES_FORM":    "Provision requires filing of a Form",
    "HAS_THRESHOLD":    "Provision has a limit / rate / deadline",
    "CONSEQUENCE_OF":   "Penalty/interest/disallowance is a consequence of breaching a provision",
    "NOTIFIED_UNDER":   "Notification issued under a Section",
    "SOURCE":           "Entity -> Chunk where it is stated (provenance)",
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def make_id(node_type: str, name: str, act: str = "") -> str:
    """Deterministic IDs => the same entity extracted twice merges into ONE node (entity resolution)."""
    prefix = f"{slug(act)}:" if act and node_type in {"Section", "Rule"} else ""
    return f"{node_type.lower()}:{prefix}{slug(name)}"


@dataclass
class Node:
    id: str
    type: str
    name: str
    description: str = ""
    act: str = ""
    props: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edge:
    source: str
    target: str
    type: str
    evidence: str = ""          # short quote / reason -> shown to user as provenance
    source_doc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
