"""Graphe LangGraph du workflow SDD.

Étapes alignées sur GitHub spec-kit :
constitution -> specify -> clarify -> review_spec* -> plan -> review_plan* -> tasks -> analyze*
             -> implement -> review_impl* -> (implement ...) -> verify -> (fix -> verify)... -> review_verify* -> END
(* = point de contrôle humain via interrupt())

Chaque nœud de génération = appel LLM (température 0, sortie structurée par JSON Schema)
+ validation déterministe + boucle de correction bornée.
Chaque nœud review_* = interrupt() : l'humain approuve, corrige ou répond aux questions.
Tout ce que dit l'humain devient une nouvelle source citable (A-xxx / F-xxx).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import prompts, render
from .ingest import split_statements
from .llm import structured_method
from .models import Clarifications, Implementation, Plan, Review, SourceStatement, Spec, TaskList
from .validators import (
    analyze,
    validate_clarify,
    fixable_files,
    is_test_file,
    topological_order,
    validate_fix,
    validate_implementation,
    validate_plan,
    validate_spec,
    validate_tasks,
)


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


# --------------------------------------------------------------------------- #
def _sources(state: SDDState) -> list[SourceStatement]:
    return [SourceStatement.model_validate(s) for s in state["sources"]]


def _fmt_sources(state: SDDState) -> str:
    return "\n".join(f"[{s['id']}] ({s['origin']}) {s['text']}" for s in state["sources"])


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def generate(llm, schema, system: str, payload: str, validate: Callable[[Any], list[str]], max_retries: int, fallback_llm=None):
    """Appel LLM structuré + validation + correction bornée.

    Ne lève pas si l'objet reste invalide : renvoie le dernier objet et ses erreurs
    pour que l'humain tranche au point de contrôle suivant.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{input}"), MessagesPlaceholder("corrections", optional=True)]
    )
    model = llm.with_structured_output(schema, method=structured_method())
    if fallback_llm is not None:  # mécanisme standard LangChain : bascule si le modèle principal échoue
        model = model.with_fallbacks([fallback_llm.with_structured_output(schema, method=structured_method())])
    chain = prompt | model  # LCEL
    corrections: list = []
    obj, errors = None, ["Aucune sortie."]
    for _ in range(max_retries + 1):
        try:
            out = chain.invoke({"input": payload, "corrections": corrections})
            obj = out if isinstance(out, schema) else schema.model_validate(out)
            errors = validate(obj)
        except Exception as exc:  # sortie non conforme au schéma / refus de l'API
            errors = [f"Sortie non conforme au schéma : {str(exc)[:500]}"]
        if not errors:
            return obj, []
        if obj is not None:
            corrections.append(AIMessage(obj.model_dump_json()))
        corrections.append(
            HumanMessage(
                "Ta réponse est rejetée par le validateur :\n- "
                + "\n- ".join(errors)
                + "\nCorrige UNIQUEMENT ces points, sans rien ajouter d'autre."
            )
        )
    if obj is None:
        raise RuntimeError("Le modèle n'a produit aucune sortie valide : " + "; ".join(errors))
    return obj, errors


def _add_human_input(state: SDDState, review: Review, open_questions: list[dict]) -> list[dict]:
    """Transforme réponses/feedback humains en sources citables."""
    sources = list(state["sources"])
    qtext = {q["id"]: q["question"] for q in open_questions}
    n_a = sum(1 for s in sources if s["origin"] == "answer") + 1
    for qid, answer in sorted(review.answers.items()):
        sources.append({"id": f"A-{n_a:03d}", "text": f"{qtext.get(qid, qid)} Réponse : {answer}", "origin": "answer"})
        n_a += 1
    if review.feedback.strip():
        n_f = sum(1 for s in sources if s["origin"] == "feedback") + 1
        sources += [s.model_dump() for s in split_statements(review.feedback, "F", "feedback", start=n_f)]
    return sources


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
):
    out = Path(out_dir)
    ws = out / "workspace"
    gen = partial(generate, fallback_llm=fallback_llm)  # tous les nœuds partagent le même modèle de secours

    # ---------------- 1. constitution (déterministe, sans LLM) ----------------
    def constitution(state: SDDState):
        srcs = split_statements(state.get("constitution", ""), "C", "constitution")
        return {"sources": [s.model_dump() for s in srcs], "current": 0, "written": [], "fix_attempts": 0}

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

    # ---------------- 3. clarify (LLM : questions seulement, puis humain) ----------------
    def clarify(state: SDDState):
        sources = _sources(state)
        payload = f"SOURCES :\n{_fmt_sources(state)}\n\nSPEC :\n{_dump(state['spec'])}"
        c, _errs = gen(llm, Clarifications, prompts.CLARIFY, payload, lambda x: validate_clarify(x, sources), max_retries)
        # questions de la spec + questions de clarification, renumérotées de façon stable
        merged, seen = [], set()
        for q in state["spec"].get("open_questions", []) + [q.model_dump() for q in c.questions]:
            key = q["question"].strip().lower()
            if key not in seen:
                seen.add(key)
                merged.append({**q, "id": f"Q-{len(merged) + 1:03d}"})
        return {"clarifications": merged}

    # interrupt() rejoue le nœud entier à la reprise : on le garde dans un nœud sans appel LLM.
    def review_spec(state: SDDState):
        merged = state["clarifications"]
        render.write(out, "spec.md", render.spec_md(state, merged))
        render.write(out, "spec.json", _dump(state["spec"]))
        raw = interrupt(
            {"stage": "clarify", "file": str(out / "spec.md"), "errors": state.get("spec_errors", []), "open_questions": merged}
        )
        rv = Review.model_validate(raw)
        update: dict[str, Any] = {"decision": rv.action}
        if rv.action == "revise":
            update["sources"] = _add_human_input(state, rv, merged)
        return update

    # ---------------- 4. plan ----------------
    def plan(state: SDDState):
        sources, spec = _sources(state), Spec.model_validate(state["spec"])
        payload = f"SOURCES :\n{_fmt_sources(state)}\n\nSPEC :\n{_dump(state['spec'])}"
        p, errs = gen(llm, Plan, prompts.PLAN, payload, lambda x: validate_plan(x, spec, sources), max_retries)
        return {"plan": p.model_dump(), "plan_errors": errs}

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
        raw = interrupt({"stage": "analyze", "file": str(out / "analyze.md"), "errors": res["errors"], "warnings": res["warnings"]})
        rv = Review.model_validate(raw)
        update: dict[str, Any] = {"decision": rv.action, "analysis": res}
        if rv.action == "revise":
            update["sources"] = _add_human_input(state, rv, [])
        return update

    # ---------------- 7. implement (une tâche à la fois) ----------------
    def implement(state: SDDState):
        tl = TaskList.model_validate(state["tasks"])
        tid = state["task_order"][state["current"]]
        task = next(t for t in tl.tasks if t.id == tid)
        reqs = [r for r in state["spec"]["requirements"] if r["id"] in task.requirement_ids]
        items = [i for i in state["plan"]["items"] if i["id"] in task.plan_item_ids]
        feedback = [s for s in state["sources"] if s["origin"] == "feedback"]
        payload = (
            f"TÂCHE :\n{task.model_dump_json(indent=1)}\n\nEXIGENCES :\n{_dump(reqs)}\n\n"
            f"ÉLÉMENTS DU PLAN :\n{_dump(items)}\n\n"
            f"CHOIX TECHNIQUES :\n{_dump(state['plan'].get('tech_choices', []))}\n\n"
            f"FICHIERS DU PROJET (déjà écrits) :\n{_dump(_read_workspace(ws))}\n\n"
            f"RETOURS HUMAINS :\n{_dump(feedback)}"
        )
        impl, errs = gen(
            llm, Implementation, prompts.IMPLEMENT, payload,
            lambda x: validate_implementation(x, tl, tid), max_retries,
        )
        return {"impl": impl.model_dump(), "impl_errors": errs}

    # ---------------- 7b. verify (déterministe) + fix (LLM) ----------------
    def verify(state: SDDState):
        res = run_tests(ws)
        render.write(out, "verify.md", f"# Vérification\n\nRésultat : {'✅ OK' if res['passed'] else '❌ échec'}\n\n```\n{res['output']}\n```\n")
        return {"verify": res}

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

    # ---------------- revues humaines ----------------
    def make_review(stage: str, renderer: Callable[[SDDState], str]):
        def review(state: SDDState):
            render.write(out, f"{stage}.md", renderer(state))
            render.write(out, f"{stage}.json", _dump(state[stage]))
            raw = interrupt(
                {
                    "stage": stage,
                    "file": str(out / f"{stage}.md"),
                    "errors": state.get(f"{stage}_errors", []),
                    "open_questions": state[stage].get("open_questions", []),
                }
            )
            rv = Review.model_validate(raw)
            update: dict[str, Any] = {"decision": rv.action}
            if rv.action == "revise":
                update["sources"] = _add_human_input(state, rv, state[stage].get("open_questions", []))
            return update

        return review

    def review_impl(state: SDDState):
        impl = state["impl"]
        raw = interrupt(
            {
                "stage": "impl",
                "task_id": impl["task_id"],
                "files": [f["path"] for f in impl["files"]],
                "preview": {f["path"]: f["content"] for f in impl["files"]},
                "errors": state.get("impl_errors", []),
            }
        )
        rv = Review.model_validate(raw)
        if rv.action == "approve":
            written = list(state.get("written", []))
            for f in impl["files"]:
                render.write(ws, f["path"], f["content"])
                if f["path"] not in written:
                    written.append(f["path"])
            return {"decision": "approve", "written": written, "current": state["current"] + 1}
        if rv.action == "revise":
            return {"decision": "revise", "sources": _add_human_input(state, rv, [])}
        return {"decision": "abort"}

    def review_verify(state: SDDState):
        raw = interrupt(
            {
                "stage": "verify",
                "file": str(out / "verify.md"),
                "passed": state["verify"]["passed"],
                "errors": [] if state["verify"]["passed"] else [state["verify"]["output"][-1500:]],
            }
        )
        rv = Review.model_validate(raw)
        if rv.action == "revise":
            return {"decision": "revise", "fix_attempts": 0, "sources": _add_human_input(state, rv, [])}
        return {"decision": rv.action}

    # ---------------- routage ----------------
    def route(next_ok: str, redo: str):
        return lambda state: {"approve": next_ok, "revise": redo}.get(state["decision"], END)

    def route_impl(state: SDDState):
        if state["decision"] == "revise":
            return "implement"
        if state["decision"] == "approve":
            return "implement" if state["current"] < len(state["task_order"]) else "verify"
        return END

    def route_verify(state: SDDState):
        if not state["verify"]["passed"] and state.get("fix_attempts", 0) < max_fix_attempts:
            return "fix"
        return "review_verify"

    g = StateGraph(SDDState)
    # Étapes alignées sur spec-kit : constitution, specify, clarify, plan, tasks, analyze, implement
    g.add_node("constitution", constitution)
    g.add_node("specify", specify)
    g.add_node("clarify", clarify)
    g.add_node("review_spec", review_spec)
    g.add_node("plan", plan)
    g.add_node("review_plan", make_review("plan", render.plan_md))
    g.add_node("tasks", tasks)
    g.add_node("analyze", analyze_node)
    g.add_node("implement", implement)
    g.add_node("review_impl", review_impl)
    g.add_node("verify", verify)
    g.add_node("fix", fix)
    g.add_node("review_verify", review_verify)

    g.add_edge(START, "constitution")
    g.add_edge("constitution", "specify")
    g.add_edge("specify", "clarify")
    g.add_edge("clarify", "review_spec")
    g.add_conditional_edges("review_spec", route("plan", "specify"), ["plan", "specify", END])
    g.add_edge("plan", "review_plan")
    g.add_conditional_edges("review_plan", route("tasks", "plan"), ["tasks", "plan", END])
    g.add_edge("tasks", "analyze")
    g.add_conditional_edges("analyze", route("implement", "tasks"), ["implement", "tasks", END])
    g.add_edge("implement", "review_impl")
    g.add_conditional_edges("review_impl", route_impl, ["implement", "verify", END])
    g.add_conditional_edges("verify", route_verify, ["fix", "review_verify"])
    g.add_edge("fix", "verify")
    g.add_conditional_edges("review_verify", route("__end__", "fix"), ["fix", END])
    return g.compile(checkpointer=checkpointer)
