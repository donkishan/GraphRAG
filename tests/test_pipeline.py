"""Offline tests: no network, no API key. A hashing embedder stands in for MiniLM and a fake LLM
stands in for Groq, so we test OUR logic (graph build, linking, traversal, guardrail).

Run:  pytest -q
"""
import hashlib
import re

import numpy as np
import pytest

from graphrag.config import settings


def hash_embed(texts):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


class FakeLLM:
    model = "fake"

    def chat(self, system, user, **kw):
        facts = re.findall(r"^\[F(\d+)\]", user, re.M)
        return ("**Short answer** The recipient pays GST under reverse charge [F1]. "
                f"Registration is compulsory under Section 24 [F{facts[-1] if facts else 1}].")

    def chat_json(self, system, user):
        return {"entities": [], "relations": []}


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    settings.index_dir = tmp_path_factory.mktemp("idx")
    from graphrag.pipeline import GraphRAG
    r = GraphRAG(backend="networkx", llm=FakeLLM(), embed_fn=hash_embed)
    r.build(use_llm=False, log=lambda *_: None)
    return r


def test_graph_built(rag):
    st = rag.store.stats()
    assert st["nodes"] > 80 and st["edges"] > 150


def test_cross_reference_rule(rag):
    edges = rag.store.all_edges()
    assert any(e["source"] == "section:cgst_act_2017:section_17_5" and e["type"] == "RESTRICTS"
               and e["target"] == "section:cgst_act_2017:section_16" for e in edges)
    assert any(e["type"] == "REFERS_TO" and e["source"].endswith("section_24")
               and e["target"].endswith("section_9_3") for e in edges)


def test_entity_linking_explicit(rag):
    seeds = rag.link_entities("What does Section 194C say about transporters?")
    assert "section:income_tax_act_1961:section_194c" in seeds


def test_multihop_gta(rag):
    seeds, facts, sources = rag.retrieve("We pay a Goods Transport Agency for freight. GST and TDS?")
    names = {f.source["name"] for f in facts} | {f.target["name"] for f in facts}
    assert "Goods Transport Agency" in names
    assert "Section 194C" in names            # income-tax side
    assert "Notification 13/2017-CT(Rate)" in names   # GST side
    assert sources


def test_guardrail_flags_hallucinated_section(rag):
    from graphrag.pipeline import grounding_check
    _, facts, sources = rag.retrieve("input tax credit conditions Section 16")
    bad = "ITC is allowed under Section 999 of the Act [F1]. Also see [F9999]."
    g = grounding_check(bad, facts, sources)
    assert not g["grounded"] and "999" in g["unsupported_sections"] and "F9999" in g["invalid_citations"]


def test_ask_end_to_end(rag):
    ans = rag.ask("Do I need GST registration if I pay tax under reverse charge?")
    assert ans.facts and ans.answer and "grounded" in ans.grounding


def test_networkx_persistence(rag):
    from graphrag.store.networkx_store import NetworkXStore
    assert NetworkXStore().stats()["nodes"] == rag.store.stats()["nodes"]


def test_guardrail_allows_refusal_echoing_question(rag):
    from graphrag.pipeline import grounding_check
    q = "What does Section 80C say about GST?"
    g = grounding_check("Section 80C is not covered. Not found in the knowledge graph.", [], [], q)
    assert g["grounded"] and g["refused"]
