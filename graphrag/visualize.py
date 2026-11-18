"""Interactive graph rendering with pyvis (vis.js) -> HTML string for Streamlit."""
from __future__ import annotations

from pyvis.network import Network

from .store.base import Fact

COLORS = {
    "Act": "#6b7280", "Section": "#2563eb", "Rule": "#0891b2", "Notification": "#7c3aed",
    "Form": "#16a34a", "Concept": "#ea580c", "Party": "#db2777", "Threshold": "#ca8a04",
}


def _label(n: dict) -> str:
    return n["name"] if len(n["name"]) <= 32 else n["name"][:30] + "..."


def facts_to_html(facts: list[Fact], seeds: list[str] | None = None, height: int = 560) -> str:
    seeds = set(seeds or [])
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111827",
                  cdn_resources="in_line")
    net.set_options("""{
      "physics": {"barnesHut": {"gravitationalConstant": -4000, "centralGravity": 0.4, "springLength": 120},
                  "stabilization": {"enabled": true, "iterations": 400, "fit": true}},
      "edges": {"smooth": {"type": "continuous"}},
      "nodes": {"font": {"size": 12}},
      "interaction": {"hover": true, "navigationButtons": true}
    }""")
    added = set()
    for f in facts:
        for n in (f.source, f.target):
            if n["id"] not in added:
                added.add(n["id"])
                title = f"{n['type']}: {n['name']}\n{n.get('act', '')}\n{n.get('description', '')}"
                net.add_node(n["id"], label=_label(n), title=title, color=COLORS.get(n["type"], "#9ca3af"),
                             size=26 if n["id"] in seeds else 14,
                             borderWidth=4 if n["id"] in seeds else 1, shape="dot")
        net.add_edge(f.source["id"], f.target["id"], label=f.rel, title=f.evidence or f.rel,
                     font={"size": 9, "align": "middle"}, arrows="to", color="#9ca3af")
    html = net.generate_html()
    # Fit after layout settles (the Streamlit iframe gets its size late, so the default fit misses)
    # and Streamlit tabs render hidden iframes at 0x0 -> re-fit whenever the container is resized/shown.
    fit_js = """<script>
      (function f(n){ if (typeof network === 'undefined' || !network) { if (n < 50) setTimeout(() => f(n+1), 100); return; }
        network.once('stabilizationIterationsDone', () => network.fit());
        let last = 0;
        setInterval(() => { const w = document.getElementById('mynetwork').clientWidth;
          if (w > 0 && w !== last) { last = w; network.setSize('100%', network.body.container.style.height || '100%');
                                     network.redraw(); network.fit(); } }, 400); })(0);
    </script></body>"""
    return html.replace("</body>", fit_js)


def legend_html() -> str:
    chips = "".join(
        f'<span style="display:inline-block;margin:2px 8px 2px 0"><span style="display:inline-block;width:10px;'
        f'height:10px;border-radius:50%;background:{c};margin-right:4px"></span>{t}</span>'
        for t, c in COLORS.items())
    return f'<div style="font-size:12px">{chips} &nbsp;(large node = entry point matched from the question)</div>'
