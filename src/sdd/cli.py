"""CLI interactive : lance le graphe SDD et traite les alertes du Gardien dans le terminal.

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

from .graph import SUPERVISED, build_graph
from .llm import PROVIDERS, corrector_rounds, get_corrector_llm, get_fallback_llm, get_guardian_llm, get_llm


SEV = {"blocking": "⛔", "warning": "⚠️ ", "info": "ℹ️ "}


def _ask_gate(payload: dict) -> dict:
    """Boîte d'alertes du Gardien en mode terminal."""
    print("\n" + "=" * 70)
    title = "ALERTE DU GARDIEN" if payload["kind"] == "alert" else "POINT DE CONTRÔLE"
    print(f"{title} — étape : {payload['stage']}")
    if payload.get("file"):
        print(f"Artefact : {payload['file']}")
    if payload.get("gate_error"):
        print(f"  ❌ Décision refusée : {payload['gate_error']}")
    if payload.get("preview"):
        for path, content in payload["preview"].items():
            print(f"\n--- {path} ---\n{content}")
    if "passed" in payload:
        print("Tests :", "✅ OK" if payload["passed"] else "❌ échec")
    for a in payload["alerts"] + payload.get("info", []):
        print(f"  {SEV[a['severity']]} [{a['code']}] {a['message']}  (refs : {', '.join(a['refs']) or '—'})")
    qs = payload.get("open_questions", [])
    menu = "[r]épondre / [c]orriger / [a]ccepter le risque / [q]uitter"
    if not payload["blocking"]:
        menu = "[ENTRÉE] continuer / " + menu
    choice = input(f"\n{menu} > ").strip().lower()
    if choice.startswith("q"):
        return {"action": "abort"}
    if choice.startswith("a"):
        return {"action": "accept_risk", "justification": input("Justification (obligatoire) > ").strip()}
    if choice.startswith("r"):
        answers = {}
        for q in qs:
            ans = input(f"Réponse à {q['id']} {q['question']} (vide = passer) > ").strip()
            if ans:
                answers[q["id"]] = ans
        return {"action": "answer", "answers": answers}
    if choice.startswith("c"):
        return {"action": "correct", "feedback": input("Correction attendue > ").strip()}
    return {"action": "continue"}


def main() -> None:
    load_dotenv()
    from .llm import setup_logging

    setup_logging()  # chaque appel au modèle (étape, essai, durée) s'affiche dans le terminal
    ap = argparse.ArgumentParser()
    ap.add_argument("brief")
    ap.add_argument("--constitution", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--thread", default=None, help="Reprendre une session existante")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--require-tests", action="store_true", help="Chaque exigence fonctionnelle doit avoir une tâche de test")
    ap.add_argument("--supervised", action="store_true",
                    help="Pause à chaque étape (sinon mode usine : pause seulement sur alerte du Gardien)")
    ap.add_argument("--no-semantic", action="store_true", help="Gardien : contrôles déterministes seulement (sans LLM)")
    ap.add_argument("--no-corrector", action="store_true", help="Sans Correcteur : toute alerte va directement à l'humain")
    ap.add_argument("--provider", choices=PROVIDERS, default=None, help="Fournisseur (sinon SDD_PROVIDER, défaut gemini)")
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
                        fallback_llm=get_fallback_llm(),
                        guardian_llm=None if args.no_semantic else get_guardian_llm(),
                        checkpoints=SUPERVISED if args.supervised else (),
                        corrector_llm=None if args.no_corrector else get_corrector_llm(),
                        max_corrections=corrector_rounds())
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
        inp = Command(resume=_ask_gate(intr[0].value))
    print("\nTerminé. Fichiers écrits :", json.dumps(result.get("written", []), ensure_ascii=False))


if __name__ == "__main__":
    main()
