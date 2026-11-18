"""Streamlit UI for Regulatory GraphRAG.   Run:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from graphrag.config import settings
from graphrag.pipeline import GraphRAG
from graphrag.visualize import facts_to_html, legend_html

st.set_page_config(page_title="Regulatory GraphRAG", page_icon="⚖️", layout="wide")

EXAMPLES = [
    "We are a company paying a Goods Transport Agency for freight. What GST and income-tax compliance applies?",
    "Can we claim input tax credit on GST paid for a company car used by directors?",
    "Our supplier has not filed GSTR-1. Can we still take ITC? What else is needed?",
    "We forgot to deduct TDS on payments to our advocate. What are the consequences?",
    "My turnover is Rs 25 lakh and I only make intra-State supplies but I pay tax under reverse charge. Do I need GST registration?",
    "A composition dealer sells to us. Can we take ITC on their invoice?",
]


@st.cache_resource(show_spinner="Loading graph + embedding model...")
def get_engine() -> GraphRAG:
    return GraphRAG()


def render_answer(ans, show_graph: bool = True, key: str = ""):
    g = ans.grounding
    badge = "✅ Grounded" if g.get("grounded") else "⚠️ Check grounding"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Guardrail", badge)
    c2.metric("Citation coverage", f"{int(g.get('citation_coverage', 0) * 100)}%")
    c3.metric("Graph facts used", len(ans.facts))
    c4.metric("Latency", f"{ans.latency_s}s")
    if g.get("unsupported_sections"):
        st.warning(f"Sections mentioned but NOT in retrieved context (possible hallucination): "
                   f"{', '.join(g['unsupported_sections'])}")
    if g.get("invalid_citations"):
        st.warning(f"Citations that point to nothing: {', '.join(g['invalid_citations'])}")
    st.markdown(ans.answer)

    if show_graph and ans.facts:
        with st.expander(f"🕸️ Reasoning sub-graph ({len(ans.facts)} facts)", expanded=True):
            st.markdown(legend_html(), unsafe_allow_html=True)
            components.html(facts_to_html(ans.facts, ans.seeds), height=580, scrolling=False)
    if ans.facts:
        with st.expander("📋 Facts passed to the LLM [F#]"):
            st.dataframe(pd.DataFrame([{
                "#": f"F{i}", "source": f.source["name"], "relation": f.rel, "target": f.target["name"],
                "hop": f.hop, "evidence": f.evidence} for i, f in enumerate(ans.facts, 1)]),
                use_container_width=True, hide_index=True)
    with st.expander(f"📄 Statute passages [S#] ({len(ans.sources)})"):
        for i, c in enumerate(ans.sources, 1):
            st.markdown(f"**[S{i}] {c.heading}**  \n<small>{c.doc}</small>", unsafe_allow_html=True)
            st.text(c.text[:1500])


# ------------------------------------------------------------------ sidebar
engine = get_engine()
with st.sidebar:
    st.title("⚖️ Regulatory GraphRAG")
    st.caption("GST + Income-tax compliance Q&A grounded in a knowledge graph")
    stats = engine.store.stats()
    st.write(f"**Graph backend:** `{settings.graph_backend}`")
    st.write(f"**LLM provider:** `{settings.llm_provider}`")
    try:
        st.write(f"**Model:** `{engine.llm.model}`")
    except Exception as e:
        st.error(f"LLM not ready: {e}")
    st.write(f"**Embeddings:** `{settings.embed_model.split('/')[-1]}` (local)")
    st.divider()
    st.metric("Nodes", stats["nodes"])
    st.metric("Relationships", stats["edges"])
    if stats["by_type"]:
        st.bar_chart(pd.Series(stats["by_type"]).drop("Chunk", errors="ignore"))
    st.divider()
    use_llm = st.checkbox("Use LLM extraction when building", value=False,
                          help="Seed + rules need no API calls. LLM extraction spends Groq free-tier quota.")
    if st.button("🔨 Build / rebuild graph", use_container_width=True):
        logs = st.empty(); lines = []
        def log(msg):
            lines.append(msg); logs.code("\n".join(lines[-12:]))
        engine.build(use_llm=use_llm, log=log)
        st.success("Graph built"); st.rerun()

if stats["nodes"] == 0:
    st.info("👈 The graph is empty. Click **Build / rebuild graph** in the sidebar first.")
    st.stop()

tab_ask, tab_cmp, tab_explore, tab_cypher, tab_ingest, tab_how = st.tabs(
    ["💬 Ask", "🆚 GraphRAG vs RAG", "🕸️ Explore graph", "🔎 Text2Cypher", "📥 Add documents", "📘 How it works"])

# ------------------------------------------------------------------ Ask
with tab_ask:
    if "history" not in st.session_state:
        st.session_state.history = []
    ex = st.selectbox("Try an example scenario", ["—"] + EXAMPLES)
    q = st.chat_input("Ask a compliance question...")
    if ex != "—" and st.button("Ask this example"):
        q = ex
    for old in st.session_state.history[-3:]:
        with st.chat_message("user"):
            st.write(old.question)
        with st.chat_message("assistant"):
            render_answer(old, show_graph=False)
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Linking entities → walking the graph → generating..."):
                try:
                    ans = engine.ask(q)
                    render_answer(ans)
                    st.session_state.history.append(ans)
                except Exception as e:
                    st.error(f"Error: {e}")

# ------------------------------------------------------------------ Compare
with tab_cmp:
    st.markdown("Same question, same LLM. Left: vector search over text only. Right: graph + text.")
    cq = st.text_area("Question", EXAMPLES[0], key="cmpq")
    if st.button("Run both"):
        l, r = st.columns(2)
        with l:
            st.subheader("Plain RAG")
            with st.spinner("..."):
                render_answer(engine.ask_plain_rag(cq), show_graph=False)
        with r:
            st.subheader("GraphRAG")
            with st.spinner("..."):
                render_answer(engine.ask(cq), show_graph=True)

# ------------------------------------------------------------------ Explore
with tab_explore:
    nodes = sorted(engine.store.all_nodes(), key=lambda n: (n["type"], n["name"]))
    types = st.multiselect("Filter types", sorted({n["type"] for n in nodes}),
                           default=["Section", "Party", "Concept"])
    opts = {f"{n['type']}: {n['name']}" + (f" [{n['act']}]" if n.get("act") else ""): n["id"]
            for n in nodes if n["type"] in types}
    keys = list(opts)
    default = next((i for i, k in enumerate(keys) if k.startswith("Section: Section 16 [")), 0)
    pick = st.selectbox("Centre node", keys, index=default)
    hops = st.slider("Hops", 1, 3, 1)
    if pick:
        facts = engine.store.expand([opts[pick]], hops, 120)
        st.markdown(legend_html(), unsafe_allow_html=True)
        components.html(facts_to_html(facts, [opts[pick]], 620), height=640)
        st.caption(f"{len(facts)} relationships shown")

# ------------------------------------------------------------------ Text2Cypher
with tab_cypher:
    if settings.graph_backend != "neo4j":
        st.info("Text2Cypher needs Neo4j. Set `GRAPH_BACKEND=neo4j` in `.env`, start Neo4j, rebuild the graph.")
    cq2 = st.text_input("Ask the graph directly", "Which parties are covered by reverse charge notifications?")
    if st.button("Generate & run Cypher", disabled=settings.graph_backend != "neo4j"):
        out = engine.text_to_cypher(cq2)
        if out.get("cypher"):
            st.code(out["cypher"], language="cypher")
        if out.get("error"):
            st.error(out["error"])
        else:
            st.markdown(out.get("summary", ""))
            st.dataframe(pd.DataFrame(out.get("rows", [])), use_container_width=True)

# ------------------------------------------------------------------ Ingest
with tab_ingest:
    st.markdown("Upload a GST / Income-tax PDF or text file (e.g. a CBIC circular). "
                "Rules extract cross-references; the LLM extracts entities & relations into the ontology.")
    up = st.file_uploader("Document", type=["pdf", "txt", "md"])
    llm_on = st.checkbox("LLM extraction (recommended for new docs)", value=True)
    if up and st.button("Ingest"):
        dest = settings.raw_dir / up.name
        dest.write_bytes(up.getvalue())
        box = st.empty(); lines = []
        def log2(msg):
            lines.append(msg); box.code("\n".join(lines[-15:]))
        st.json(engine.ingest_file(dest, use_llm=llm_on, log=log2))
        st.success(f"Saved to data/raw/{up.name} and merged into the graph.")

# ------------------------------------------------------------------ How it works
with tab_how:
    st.markdown("""
### Pipeline
```
 Documents (.txt/.pdf)
     │ 1. provision-aware chunking (one chunk per Section)
     ▼
 Extraction ──► seed triples (expert) + regex rules (cross-refs, forms) + LLM (schema-constrained JSON)
     ▼
 Knowledge Graph (NetworkX / Neo4j)   +   Vector index (all-MiniLM-L6-v2, local)
     │
 Question ─► 2. entity linking (regex "Section 16", "GSTR-3B" + semantic match)
          ─► 3. k-hop graph walk from those entry nodes (facts F1..Fn)
          ─► 4. fetch statute text of the provisions reached (S1..Sn)
          ─► 5. LLM answers ONLY from F/S with citations
          ─► 6. guardrail: citations valid? sections mentioned exist in context?
```
### Why the graph removes hallucinations
* **Multi-hop**: "GTA freight" → *Notification 13/2017* → *Section 9(3)* → *Section 24* (must register) — plain vector
  search finds the notification but rarely the registration consequence two hops away.
* **Exceptions & overrides** are explicit edges (`RESTRICTS`, `EXCEPTION_TO`) — e.g. 17(5) restricts 16.
* **Deterministic provenance**: every fact the LLM may use is a typed edge with evidence, and the answer must cite it.
""")
