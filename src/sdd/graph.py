"""Graphe LangGraph du workflow SDD, surveillé par le Gardien (architecture « AI factory »).

Côté production, les étapes de GitHub spec-kit :
    constitution -> specify -> clarify -> plan -> tasks -> analyze -> implement (une tâche à la fois) -> verify ⇄ fix

Côté contrôle, après CHAQUE étape productrice, le nœud `guardian` exécute le graphe du Gardien
(compilé séparément, voir guardian.py) sur les artefacts produits :

    clarify ─┐   plan ─┐   analyze ─┐   implement ─┐   verify ─┐
             ▼         ▼            ▼              ▼           ▼
          guardian ──(alerte ?)──► gate ⏸ interrupt() ──► décision humaine ──► reprise au bon nœud
             │ (rien à signaler)
             ▼
          étape suivante

Le nœud `gate` est le seul point de pause : il met le workflow en pause avec interrupt() (état sauvegardé
par le checkpointer) et attend la décision humaine : répondre (sources A-xxx), corriger (sources F-xxx),
accepter le risque (justification obligatoire), continuer (avertissements seulement) ou arrêter.
Chaque contrôle et chaque décision sont ajoutés au journal d'audit (`audit`).
"""
from __future__ import annotations

import json
import operator
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import prompts, render
from .guardian import build_guardian, check_decision, redo_node
from .ingest import split_statements
from .llm import generate
from .models import Clarifications, GateDecision, Implementation, Plan, SourceStatement, Spec, TaskList
from .validators import (
    _norm,
    analyze,
    fixable_files,
    is_test_file,
    topological_order,
    validate_clarify,
    validate_fix,
    validate_implementation,
    validate_plan,
    validate_spec,
    validate_tasks,
)

__all__ = ["build_graph", "generate", "run_tests", "SUPERVISED"]

# Mode supervisé : pause aussi à la fin de chaque étape, même sans alerte du Gardien.
SUPERVISED = ("spec", "plan", "analyze", "impl", "verify")
# Étape suivante quand le Gardien laisse passer (ou que l'humain accepte / acquitte).
NEXT = {"spec": "plan", "plan": "tasks", "analyze": "implement", "impl": "apply_impl", "verify": END}
STAGE_FILE = {"spec": "spec.md", "plan": "plan.md", "analyze": "analyze.md", "impl": "alerts.md", "verify": "verify.md"}


class SDDState(TypedDict, total=False):
    brief: str
    constitution: str
    sources: list[dict]
    spec: dict
    spec_errors: list[str]
    clarifications: list[dict]
    analysis: dict
    plan: dict
    plan_errors: list[str]
    tasks: dict
    tasks_errors: list[str]
    task_order: list[str]
    current: int
    impl: dict
    impl_errors: list[str]
    verify: dict
    fix_attempts: int
    decision: str
    written: list[str]
    # --- Gardien ---
    stage: str                 # dernière étape produite, contrôlée par le Gardien
    alerts: list[dict]         # rapport du dernier contrôle
    accepted: list[str]        # empreintes des alertes acceptées / acquittées par l'humain
    answered: list[str]        # questions déjà répondues (texte normalisé) : jamais reposées
    redo: str                  # nœud à relancer après une réponse ou une correction
    gate_error: str | None     # décision humaine refusée (ex. accepter un risque sans justification)
    audit: Annotated[list[dict], operator.add]  # journal d'audit (réducteur : on ajoute, on n'efface jamais)


# --------------------------------------------------------------------------- #
def _sources(state: SDDState) -> list[SourceStatement]:
    return [SourceStatement.model_validate(s) for s in state["sources"]]


def _fmt_sources(state: SDDState) -> str:
    return "\n".join(f"[{s['id']}] ({s['origin']}) {s['text']}" for s in state["sources"])


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def _add_human_input(state: SDDState, answers: dict[str, str], feedback: str, open_questions: list[dict]) -> list[dict]:
    """Transforme réponses/retours humains en sources citables (A-xxx / F-xxx)."""
    sources = list(state["sources"])
    qtext = {q["id"]: q["question"] for q in open_questions}
    n_a = sum(1 for s in sources if s["origin"] == "answer") + 1
    for qid, answer in sorted(answers.items()):
        sources.append({"id": f"A-{n_a:03d}", "text": f"{qtext.get(qid, qid)} Réponse : {answer}", "origin": "answer"})
        n_a += 1
    if feedback.strip():
        n_f = sum(1 for s in sources if s["origin"] == "feedback") + 1
        sources += [s.model_dump() for s in split_statements(feedback, "F", "feedback", start=n_f)]
    return sources


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_workspace(ws: Path) -> dict[str, str]:
    if not ws.exists():
        return {}
    return {
        str(p.relative_to(ws)): p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(ws.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
    }


def run_tests(ws: Path, timeout: int = 120) -> dict:
    """Exécute pytest dans le workspace. Déterministe côté pipeline : seul le code testé varie."""
    has_tests = any(is_test_file(f) and f.rsplit("/", 1)[-1].startswith("test_") for f in _read_workspace(ws))
    if not has_tests:
        return {"passed": True, "ran": False, "output": "Aucun fichier de test dans le workspace."}
    env = {**os.environ, "PYTHONPATH": f"{ws}{os.pathsep}{ws / 'src'}", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header"],
            cwd=ws, env=env, capture_output=True, text=True, timeout=timeout,
        )
        out = (p.stdout + p.stderr)[-6000:]
        if p.returncode == 5:  # code pytest « aucun test collecté »
            return {"passed": True, "ran": False, "output": out}
        return {"passed": p.returncode == 0, "ran": True, "output": out}
    except subprocess.TimeoutExpired:
        return {"passed": False, "ran": True, "output": f"Délai dépassé ({timeout}s)."}


# --------------------------------------------------------------------------- #
def build_graph(
    llm, out_dir: str | Path = "out", max_retries: int = 3, checkpointer=None,
    max_fix_attempts: int = 3, require_tests: bool = False, fallback_llm=None,
    guardian_llm=None, checkpoints: Iterable[str] = (),
):
    """Construit le graphe SDD surveillé par le Gardien.

    guardian_llm : LLM du contrôle sémantique du Gardien (None = contrôles déterministes seuls).
    checkpoints  : étapes où l'on fait une pause même sans alerte (SUPERVISED = toutes) ;
                   vide = mode « usine » : pause uniquement quand le Gardien lève une alerte.
    """
    out = Path(out_dir)
    ws = out / "workspace"
    checkpoints = set(checkpoints)
    gen = partial(generate, fallback_llm=fallback_llm)  # tous les nœuds partagent le même modèle de secours
    guardian_graph = build_guardian(guardian_llm, max_retries=1, require_tests=require_tests)

    # ---------------- 1. constitution (déterministe, sans LLM) ----------------
    def constitution(state: SDDState):
        srcs = split_statements(state.get("constitution", ""), "C", "constitution")
        return {"sources": [s.model_dump() for s in srcs], "current": 0, "written": [], "fix_attempts": 0,
                "accepted": [], "answered": []}

    # ---------------- 2. specify ----------------
    def specify(state: SDDState):
        update: dict[str, Any] = {}
        if not any(s["origin"] == "brief" for s in state["sources"]):  # 1er passage : découpage du brief
            brief = [s.model_dump() for s in split_statements(state["brief"], "B", "brief")]
            state = {**state, "sources": state["sources"] + brief}
            update["sources"] = state["sources"]
        render.write(out, "sources.json", _dump(state["sources"]))
        sources = _sources(state)
        spec, errs = gen(
            llm, Spec, prompts.SPECIFY, "SOURCES :\n" + _fmt_sources(state),
            lambda s: validate_spec(s, sources), max_retries,
        )
        return {**update, "spec": spec.model_dump(), "spec_errors": errs}

    # ---------------- 3. clarify (LLM : questions seulement ; les réponses viennent de l'humain) ----------------
    def clarify(state: SDDState):
        sources = _sources(state)
        payload = f"SOURCES :\n{_fmt_sources(state)}\n\nSPEC :\n{_dump(state['spec'])}"
        c, _errs = gen(llm, Clarifications, prompts.CLARIFY, payload, lambda x: validate_clarify(x, sources), max_retries)
        # questions encore ouvertes du tour précédent + questions de la spec + questions de clarification,
        # renumérotées de façon stable. Règles déterministes : une question déjà répondue n'est jamais
        # reposée, et une question sans réponse ne disparaît jamais d'elle-même (seule une réponse ou une
        # acceptation de risque explicite la fait sortir du chemin).
        previous = state.get("clarifications") or []
        merged, seen = [], set(state.get("answered", []))
        for q in previous + state["spec"].get("open_questions", []) + [q.model_dump() for q in c.questions]:
            key = _norm(q["question"])
            if key not in seen:
                seen.add(key)
                merged.append({**q, "id": f"Q-{len(merged) + 1:03d}"})
        view = {**state, "clarifications": merged}
        render.write(out, "spec.md", render.spec_md(view, merged))
        render.write(out, "spec.json", _dump(state["spec"]))
        return {"clarifications": merged, "stage": "spec"}

    # ---------------- 4. plan ----------------
    def plan(state: SDDState):
        sources, spec = _sources(state), Spec.model_validate(state["spec"])
        payload = f"SOURCES :\n{_fmt_sources(state)}\n\nSPEC :\n{_dump(state['spec'])}"
        p, errs = gen(llm, Plan, prompts.PLAN, payload, lambda x: validate_plan(x, spec, sources), max_retries)
        p = p.model_dump()
        answered = set(state.get("answered", []))
        # questions du plan : préfixe QP- pour ne pas se confondre avec celles de la spec
        p["open_questions"] = [{**q, "id": f"QP-{i:03d}"} for i, q in enumerate(
            (q for q in p["open_questions"] if _norm(q["question"]) not in answered), 1)]
        view = {**state, "plan": p, "plan_errors": errs}
        render.write(out, "plan.md", render.plan_md(view))
        render.write(out, "plan.json", _dump(p))
        return {"plan": p, "plan_errors": errs, "stage": "plan"}

    # ---------------- 5. tasks ----------------
    def tasks(state: SDDState):
        spec, p = Spec.model_validate(state["spec"]), Plan.model_validate(state["plan"])
        payload = f"SPEC :\n{_dump(state['spec'])}\n\nPLAN :\n{_dump(state['plan'])}"
        t, errs = gen(llm, TaskList, prompts.TASKS, payload, lambda x: validate_tasks(x, p, spec, require_tests), max_retries)
        order = [x.id for x in t.tasks] if any("cycle" in e for e in errs) else topological_order(t)
        return {"tasks": t.model_dump(), "tasks_errors": errs, "task_order": order, "current": 0}

    # ---------------- 6. analyze (déterministe, sans LLM) ----------------
    def analyze_node(state: SDDState):
        res = analyze(
            Spec.model_validate(state["spec"]), Plan.model_validate(state["plan"]),
            TaskList.model_validate(state["tasks"]), _sources(state), require_tests,
        )
        render.write(out, "tasks.md", render.tasks_md(state))
        render.write(out, "tasks.json", _dump(state["tasks"]))
        render.write(out, "analyze.md", render.analyze_md(state, res))
        return {"analysis": res, "stage": "analyze"}

    # ---------------- 7. implement (une tâche à la fois) ----------------
    def implement(state: SDDState):
        tl = TaskList.model_validate(state["tasks"])
        tid = state["task_order"][state["current"]]
        task = next(t for t in tl.tasks if t.id == tid)
        reqs = [r for r in state["spec"]["requirements"] if r["id"] in task.requirement_ids]
        items = [i for i in state["plan"]["items"] if i["id"] in task.plan_item_ids]
        human = [s for s in state["sources"] if s["origin"] in ("feedback", "answer")]
        payload = (
            f"TÂCHE :\n{task.model_dump_json(indent=1)}\n\nEXIGENCES :\n{_dump(reqs)}\n\n"
            f"ÉLÉMENTS DU PLAN :\n{_dump(items)}\n\n"
            f"CHOIX TECHNIQUES :\n{_dump(state['plan'].get('tech_choices', []))}\n\n"
            f"FICHIERS DU PROJET (déjà écrits) :\n{_dump(_read_workspace(ws))}\n\n"
            f"RÉPONSES ET RETOURS HUMAINS :\n{_dump(human)}"
        )
        impl, errs = gen(
            llm, Implementation, prompts.IMPLEMENT, payload,
            lambda x: validate_implementation(x, tl, tid), max_retries,
        )
        return {"impl": impl.model_dump(), "impl_errors": errs, "stage": "impl"}

    # Les fichiers ne sont écrits qu'APRÈS le feu vert du Gardien (ou de l'humain).
    def apply_impl(state: SDDState):
        written = list(state.get("written", []))
        for f in state["impl"]["files"]:
            render.write(ws, f["path"], f["content"])
            if f["path"] not in written:
                written.append(f["path"])
        return {"written": written, "current": state["current"] + 1}

    # ---------------- 7b. verify (déterministe) + fix (LLM) ----------------
    def verify(state: SDDState):
        res = run_tests(ws)
        render.write(out, "verify.md", f"# Vérification\n\nRésultat : {'✅ OK' if res['passed'] else '❌ échec'}\n\n```\n{res['output']}\n```\n")
        return {"verify": res, "stage": "verify"}

    def fix(state: SDDState):
        tl = TaskList.model_validate(state["tasks"])
        feedback = [s for s in state["sources"] if s["origin"] == "feedback"]
        payload = (
            f"SORTIE DES TESTS :\n{state['verify']['output']}\n\n"
            f"ALLOWED_FILES :\n{_dump(fixable_files(tl))}\n\n"
            f"FICHIERS DU PROJET :\n{_dump(_read_workspace(ws))}\n\n"
            f"EXIGENCES :\n{_dump(state['spec']['requirements'])}\n\n"
            f"RETOURS HUMAINS :\n{_dump(feedback)}"
        )
        patch, errs = gen(llm, Implementation, prompts.FIX, payload, lambda x: validate_fix(x, tl), max_retries)
        written = list(state.get("written", []))
        if not errs:  # on n'écrit jamais un correctif invalide
            for f in patch.files:
                render.write(ws, f.path, f.content)
                if f.path not in written:
                    written.append(f.path)
        return {"fix_attempts": state.get("fix_attempts", 0) + 1, "written": written}

    # ---------------- Le Gardien : contrôle indépendant après chaque étape ----------------
    ARTIFACTS = ("sources", "spec", "clarifications", "plan", "tasks", "impl", "verify", "written")

    def guardian(state: SDDState):
        stage = state["stage"]
        artifacts = {k: state.get(k) for k in ARTIFACTS}
        if stage == "impl":
            artifacts["task_id"] = state["task_order"][state["current"]]
        # Le Gardien est un graphe à part : il ne reçoit qu'un instantané des artefacts, jamais l'état du producteur.
        report = guardian_graph.invoke(
            {"stage": stage, "artifacts": artifacts, "accepted": state.get("accepted", [])}
        )["report"]
        entry = {
            "time": _now(), "type": "contrôle", "stage": stage,
            "task_id": artifacts.get("task_id"),
            "alerts": [{k: a[k] for k in ("id", "code", "severity", "message")} for a in report],
        }
        render.write(out, "alerts.md", render.alerts_md(stage, report))
        render.write(out, "audit.md", render.audit_md(state.get("audit", []) + [entry]))
        return {"alerts": report, "audit": [entry], "gate_error": None}

    def gate(state: SDDState):
        stage, alerts = state["stage"], state.get("alerts", [])
        active = [a for a in alerts if a["severity"] != "info"]
        payload: dict[str, Any] = {
            "stage": stage,
            "kind": "alert" if active else "checkpoint",
            "blocking": any(a["severity"] == "blocking" for a in active),
            "alerts": active,
            "info": [a for a in alerts if a["severity"] == "info"],
            "open_questions": [a["question"] for a in active if a.get("question")],
            "file": str(out / STAGE_FILE[stage]),
            "gate_error": state.get("gate_error"),
            "errors": [a["message"] for a in active if a["severity"] == "blocking"],
            "warnings": [a["message"] for a in active if a["severity"] == "warning"],
        }
        if stage == "impl":
            impl = state["impl"]
            payload |= {"task_id": impl["task_id"], "files": [f["path"] for f in impl["files"]],
                        "preview": {f["path"]: f["content"] for f in impl["files"]}}
        if stage == "verify":
            payload["passed"] = state["verify"]["passed"]
        raw = interrupt(payload)  # ⏸ pause : l'état est sauvegardé par le checkpointer jusqu'à la décision

        d = GateDecision.model_validate(raw)
        answers = {k: v.strip() for k, v in d.answers.items() if v.strip()}
        action = d.action
        if answers and action != "abort":
            action = "answer"  # des réponses saisies ne sont JAMAIS perdues (bug corrigé)
        error = check_decision(action, answers, d.feedback, d.justification, alerts)
        entry = {
            "time": _now(), "type": "décision", "stage": stage, "action": action,
            "alerts": [a["id"] for a in active], "answers": answers, "feedback": d.feedback.strip(),
            "justification": d.justification.strip(), "refused": error,
        }
        render.write(out, "audit.md", render.audit_md(state.get("audit", []) + [entry]))
        if error:
            return {"decision": "invalid", "gate_error": error, "audit": [entry]}
        update: dict[str, Any] = {"decision": action, "gate_error": None, "audit": [entry]}
        if action in ("answer", "correct"):
            qs = [a["question"] for a in active if a.get("question")]
            update["sources"] = _add_human_input(state, answers, d.feedback, qs)
            update["answered"] = state.get("answered", []) + [_norm(q["question"]) for q in qs if q["id"] in answers]
            update["redo"] = redo_node(active, stage, only_questions=(action == "answer" and not d.feedback.strip()))
            if update["redo"] == "fix":
                update["fix_attempts"] = 0
        elif action in ("accept_risk", "continue"):
            update["accepted"] = state.get("accepted", []) + [a["id"] for a in active]
        return update

    # ---------------- routage ----------------
    def route_guardian(state: SDDState):
        if any(a["severity"] != "info" for a in state.get("alerts", [])) or state["stage"] in checkpoints:
            return "gate"
        return NEXT[state["stage"]]

    def route_gate(state: SDDState):
        d = state["decision"]
        if d == "invalid":
            return "gate"  # décision refusée : on redemande, avec le motif
        if d == "abort":
            return END
        if d in ("continue", "accept_risk"):
            return NEXT[state["stage"]]
        return state["redo"]

    def route_apply(state: SDDState):
        return "implement" if state["current"] < len(state["task_order"]) else "verify"

    def route_verify(state: SDDState):
        if not state["verify"]["passed"] and state.get("fix_attempts", 0) < max_fix_attempts:
            return "fix"
        return "guardian"

    g = StateGraph(SDDState)
    # Production (étapes spec-kit)
    g.add_node("constitution", constitution)
    g.add_node("specify", specify)
    g.add_node("clarify", clarify)
    g.add_node("plan", plan)
    g.add_node("tasks", tasks)
    g.add_node("analyze", analyze_node)
    g.add_node("implement", implement)
    g.add_node("apply_impl", apply_impl)
    g.add_node("verify", verify)
    g.add_node("fix", fix)
    # Contrôle
    g.add_node("guardian", guardian)
    g.add_node("gate", gate)

    g.add_edge(START, "constitution")
    g.add_edge("constitution", "specify")
    g.add_edge("specify", "clarify")
    for producer in ("clarify", "plan", "analyze", "implement"):
        g.add_edge(producer, "guardian")
    g.add_edge("tasks", "analyze")
    g.add_conditional_edges("verify", route_verify, ["fix", "guardian"])
    g.add_edge("fix", "verify")
    g.add_conditional_edges("guardian", route_guardian, ["gate", "plan", "tasks", "implement", "apply_impl", END])
    g.add_conditional_edges("gate", route_gate,
                            ["gate", "specify", "plan", "tasks", "implement", "apply_impl", "fix", END])
    g.add_conditional_edges("apply_impl", route_apply, ["implement", "verify"])
    return g.compile(checkpointer=checkpointer)
