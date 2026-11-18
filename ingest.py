"""Command-line entry points.

  python ingest.py build              # seed + rule extraction over data/raw (no API calls)
  python ingest.py build --llm        # also run LLM extraction on every chunk (uses Groq/Ollama)
  python ingest.py add path/to.pdf    # add one more document (LLM extraction on)
  python ingest.py ask "question"     # GraphRAG answer in the terminal
  python ingest.py compare "question" # GraphRAG vs plain RAG
  python ingest.py stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag.pipeline import GraphRAG


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--llm", action="store_true")
    a = sub.add_parser("add"); a.add_argument("path"); a.add_argument("--no-llm", action="store_true")
    q = sub.add_parser("ask"); q.add_argument("question")
    c = sub.add_parser("compare"); c.add_argument("question")
    sub.add_parser("stats")
    args = ap.parse_args()

    rag = GraphRAG()
    try:
        if args.cmd == "build":
            rag.build(use_llm=args.llm)
        elif args.cmd == "add":
            print(rag.ingest_file(Path(args.path), use_llm=not args.no_llm))
        elif args.cmd == "stats":
            print(json.dumps(rag.store.stats(), indent=2))
        elif args.cmd == "ask":
            ans = rag.ask(args.question)
            print("\nEntry nodes:", ans.seeds)
            print(f"Facts used: {len(ans.facts)} | Sources: {[s.heading for s in ans.sources]}\n")
            print(ans.answer)
            print("\nGrounding:", ans.grounding)
        elif args.cmd == "compare":
            for ans in (rag.ask_plain_rag(args.question), rag.ask(args.question)):
                print("=" * 30, ans.mode.upper(), "=" * 30)
                print(ans.answer)
                print("Grounding:", ans.grounding, "\n")
    finally:
        rag.store.close()


if __name__ == "__main__":
    main()
