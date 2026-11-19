"""CLI interactive : lance le graphe et gère les points de contrôle humains.

Usage : python -m sdd.cli examples/brief.md [--constitution examples/constitution.md] [--out out]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from .graph import build_graph
from .llm import PROVIDERS, get_fallback_llm, get_llm


def _ask_review(payload: dict) -> dict:
    print("\n" + "=" * 70)
    print(f"POINT DE CONTRÔLE : {payload['stage']}")
    if "file" in payload:
        print(f"Artefact : {payload['file']}")
    if payload.get("preview"):
        for path, content in payload["preview"].items():
            print(f"\n--- {path} ---\n{content}")
    if "passed" in payload:
        print("Tests :", "✅ OK" if payload["passed"] else "❌ échec")
    for e in payload.get("errors", []):
        print(f"  ❌ {e}")
    for w in payload.get("warnings", []):
        print(f"  ⚠️  {w}")
    qs = payload.get("open_questions", [])
    for q in qs:
        print(f"  ❓ {q['id']} {q['question']}")
    choice = input("\n[a]pprouver / [r]éviser / [q]uitter > ").strip().lower()
    if choice.startswith("q"):
        return {"action": "abort"}
    if choice.startswith("a"):
        return {"action": "approve"}
    answers = {}
    for q in qs:
        ans = input(f"Réponse à {q['id']} (vide = passer) > ").strip()
        if ans:
            answers[q["id"]] = ans
    feedback = input("Retour libre (vide = aucun) > ").strip()
    return {"action": "revise", "answers": answers, "feedback": feedback}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("brief")
    ap.add_argument("--constitution", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--thread", default=None, help="Reprendre une session existante")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--require-tests", action="store_true", help="Chaque exigence fonctionnelle doit avoir une tâche de test")
    ap.add_argument("--provider", choices=PROVIDERS, default=None, help="Fournisseur (sinon SDD_PROVIDER, défaut ollama)")
    ap.add_argument("--model", default=None, help="Modèle (sinon SDD_MODEL, défaut selon le fournisseur)")
    args = ap.parse_args()
    if args.provider:
        os.environ["SDD_PROVIDER"] = args.provider
        if not args.model:  # le modèle et le secours du .env visaient peut-être un autre fournisseur
            os.environ.pop("SDD_MODEL", None)
            os.environ.pop("SDD_FALLBACK_MODEL", None)
    if args.model:
        os.environ["SDD_MODEL"] = args.model

    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(args.out).mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(Path(args.out) / "checkpoints.db", check_same_thread=False))
    graph = build_graph(get_llm(), args.out, args.max_retries, checkpointer=saver, require_tests=args.require_tests,
                        fallback_llm=get_fallback_llm())
    thread = args.thread or uuid.uuid4().hex[:8]
    cfg = {"configurable": {"thread_id": thread}}
    print(f"Session : {thread} (reprendre avec --thread {thread})")

    if args.thread:
        inp = None
    else:
        inp = {
            "brief": Path(args.brief).read_text(encoding="utf-8"),
            "constitution": Path(args.constitution).read_text(encoding="utf-8") if args.constitution else "",
        }
    while True:
        result = graph.invoke(inp, cfg)
        intr = result.get("__interrupt__")
        if not intr:
            break
        inp = Command(resume=_ask_review(intr[0].value))
    print("\nTerminé. Fichiers écrits :", json.dumps(result.get("written", []), ensure_ascii=False))


if __name__ == "__main__":
    main()
