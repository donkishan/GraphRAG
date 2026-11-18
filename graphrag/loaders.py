"""Load .txt / .md / .pdf and split them into provision-aware chunks.

Legal text should be chunked by PROVISION (section), not by a fixed character count,
otherwise a condition and its proviso end up in different chunks.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Our curated corpus uses:  ## <Act> | <Section/Rule/Notification label> | <title>
HEADER_RE = re.compile(r"^##\s*(?P<act>[^|]+)\|\s*(?P<label>[^|]+)\|\s*(?P<title>.+)$", re.M)
# Generic statute PDFs: lines like "16. Eligibility and conditions ..." or "Section 194C. Payments to contractors"
GENERIC_RE = re.compile(r"^\s*(?:Section\s+)?(?P<num>\d{1,3}[A-Z]{0,3})\.\s+(?P<title>[A-Z][^\n]{3,120})$", re.M)


@dataclass
class Chunk:
    id: str
    doc: str
    act: str
    label: str      # e.g. "Section 16"
    title: str
    text: str

    @property
    def heading(self) -> str:
        return f"{self.act} {self.label} - {self.title}".strip()


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _cid(doc: str, label: str, text: str) -> str:
    return "chunk:" + hashlib.md5(f"{doc}|{label}|{text[:200]}".encode()).hexdigest()[:12]


def _window(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def chunk_text(text: str, doc: str, default_act: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    headers = list(HEADER_RE.finditer(text))
    if headers:
        for i, m in enumerate(headers):
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            body = text[m.end():end].strip()
            act, label, title = (m["act"].strip(), m["label"].strip(), m["title"].strip())
            chunks.append(Chunk(_cid(doc, label, body), doc, act, label, title, body))
        return chunks

    generic = list(GENERIC_RE.finditer(text))
    if len(generic) >= 3:
        for i, m in enumerate(generic):
            end = generic[i + 1].start() if i + 1 < len(generic) else len(text)
            body = text[m.start():end].strip()
            for j, part in enumerate(_window(body, 2500, 250)):
                label = f"Section {m['num']}"
                chunks.append(Chunk(_cid(doc, f"{label}#{j}", part), doc, default_act, label,
                                    m["title"].strip(), part))
        return chunks

    for j, part in enumerate(_window(text)):
        chunks.append(Chunk(_cid(doc, str(j), part), doc, default_act, f"Part {j + 1}", doc, part))
    return chunks


def load_path(path: Path, default_act: str = "") -> list[Chunk]:
    return chunk_text(read_file(path), path.name, default_act or path.stem)


def load_dir(folder: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in {".txt", ".md", ".pdf"}:
            out.extend(load_path(p))
    return out
