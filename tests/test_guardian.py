"""Le Gardien : contrôles déterministes, contrôle sémantique et briefs piégés (tous hors ligne).

Chaque « brief piégé » reproduit une façon dont l'IA pourrait inventer ou dériver.
Tous doivent mettre le workflow en pause.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from sdd.graph import build_graph
from sdd.guardian import build_guardian, check_impl, consolidate_alerts, make_alert
from sdd.ingest import split_statements
from sdd.models import Clarifications, Implementation, SemanticReview, Spec, TaskList
from test_pipeline import BRIEF, CONSTITUTION, GOOD_PLAN, GOOD_SPEC, GOOD_TASKS, FakeLLM, cite

# Reproduction du dernier test réel : un brief de dépenses qui ne dit rien du stockage ni des identifiants.
EXPENSES_BRIEF = """# Dépenses
- L'utilisateur peut ajouter une dépense avec un montant.
- L'utilisateur peut lister les dépenses.
"""
EXPENSES_SPEC = Spec.model_validate({"requirements": [
    {"id": "REQ-001", "kind": "functional", "statement": "L'utilisateur peut ajouter une dépense avec un montant.",
     "citations": [cite("B-001", "ajouter une dépense avec un montant")]},
    {"id": "REQ-002", "kind": "functional", "statement": "L'utilisateur peut lister les dépenses.",
     "citations": [cite("B-002", "lister les dépenses")]},
]})
EXPENSES_PLAN = {"items": [{"id": "PLN-001", "kind": "component", "title": "Dépenses", "description": "x",
                            "requirement_ids": ["REQ-001", "REQ-002"]}], "tech_choices": [], "open_questions": []}
EXPENSES_TASKS = {"tasks": [{"id": "T-001", "title": "Dépenses", "description": "x", "requirement_ids": ["REQ-001", "REQ-002"],
                             "plan_item_ids": ["PLN-001"], "files": ["expenses/cli.py"], "depends_on": [],
                             "acceptance_criteria": []}]}
INVENTIVE_CODE = """import argparse
import json

DB = "expenses.json"

def next_id(items):
    return max([i["id"] for i in items], default=0) + 1
"""


def expenses_artifacts(code=INVENTIVE_CODE):
    sources = split_statements(EXPENSES_BRIEF, "B", "brief")
    return {
        "sources": [s.model_dump() for s in sources], "spec": EXPENSES_SPEC.model_dump(), "clarifications": [],
        "plan": EXPENSES_PLAN, "tasks": EXPENSES_TASKS, "written": [], "task_id": "T-001",
        "impl": {"task_id": "T-001", "files": [{"path": "expenses/cli.py", "content": code}]},
    }


# ---------------------------- contrôles déterministes ---------------------------- #
def test_trap_invented_data_file_is_blocking():
    alerts = check_impl(expenses_artifacts())
    codes = {a["code"]: a for a in alerts}
    assert codes["FICHIER_NON_SOURCE"]["severity"] == "blocking"
    assert "expenses.json" in codes["FICHIER_NON_SOURCE"]["message"]
    # la bibliothèque standard est tolérée mais tracée (argparse, json)
    assert codes["BIBLIOTHEQUE_STANDARD"]["severity"] == "info"
    assert "argparse" in codes["BIBLIOTHEQUE_STANDARD"]["message"]


def test_data_file_whose_format_is_sourced_is_only_a_warning():
    a = expenses_artifacts()
    a["sources"].append({"id": "B-003", "text": "Les dépenses sont sauvegardées dans un fichier JSON.", "origin": "brief"})
    codes = {x["code"]: x["severity"] for x in check_impl(a)}
    assert codes["NOM_FICHIER_CHOISI"] == "warning" and "FICHIER_NON_SOURCE" not in codes
    a["sources"].append({"id": "A-001", "text": "Nom du fichier ? Réponse : expenses.json", "origin": "answer"})
    assert "NOM_FICHIER_CHOISI" not in {x["code"] for x in check_impl(a)}  # nommé par l'humain : OK


def test_trap_unsourced_external_dependency_is_blocking():
    alerts = check_impl(expenses_artifacts("import click\nimport pytest\n"))
    blocking = [a for a in alerts if a["severity"] == "blocking"]
    assert [a["code"] for a in blocking] == ["DEPENDANCE_NON_SOURCEE", "DEPENDANCE_NON_SOURCEE"]
    a = expenses_artifacts("import pytest\n")
    a["sources"].append({"id": "C-001", "text": "Les tests utilisent pytest.", "origin": "constitution"})
    assert not [x for x in check_impl(a) if x["severity"] == "blocking"]  # pytest est autorisé par une source


def test_trap_duplicate_requirements_are_blocking(tmp_path):
    dup = Spec.model_validate(GOOD_SPEC.model_dump() | {"requirements": [GOOD_SPEC.model_dump()["requirements"][0]] * 2
                                                        + [GOOD_SPEC.model_dump()["requirements"][1]]})
    llm = FakeLLM({"Spec": [dup], "Clarifications": [Clarifications()]})
    g = build_graph(llm, tmp_path, max_retries=0, checkpointer=MemorySaver())
    res = g.invoke({"brief": BRIEF, "constitution": CONSTITUTION}, {"configurable": {"thread_id": "d"}})
    pause = res["__interrupt__"][0].value
    assert pause["blocking"] and any("dupliqués" in a["message"] for a in pause["alerts"])


def test_alert_fingerprint_is_stable_and_acceptance_downgrades_to_info():
    a1 = make_alert("QUESTION_OUVERTE", "blocking", "spec", "Q-001 x", ["Q-001"], key="Quel format ?")
    a2 = make_alert("QUESTION_OUVERTE", "blocking", "analyze", "Q-003 y", ["Q-003"], key="Quel  format ?")
    assert a1["id"] == a2["id"]  # même question renumérotée, même empreinte
    out = consolidate_alerts([a2, a2], [a1["id"]])
    assert len(out) == 1 and out[0]["severity"] == "info" and out[0]["accepted"]


# ---------------------------- contrôle sémantique (LLM) ---------------------------- #
def guardian_artifacts():
    sources = split_statements(CONSTITUTION, "C", "constitution") + split_statements(BRIEF, "B", "brief")
    return {"sources": [s.model_dump() for s in sources], "spec": GOOD_SPEC.model_dump(), "clarifications": [],
            "plan": GOOD_PLAN.model_dump(), "tasks": GOOD_TASKS.model_dump()}


def test_trap_contradiction_found_by_semantic_guardian():
    auditor = FakeLLM({"SemanticReview": [SemanticReview(findings=[
        {"kind": "contradiction", "severity": "blocking", "refs": ["B-002", "REQ-002"],
         "message": "REQ-002 stocke en base SQL alors que B-002 demande un fichier JSON."}])]})
    report = build_guardian(auditor).invoke({"stage": "plan", "artifacts": guardian_artifacts(), "accepted": []})["report"]
    assert [(a["code"], a["severity"], a["origin"], a["redo"]) for a in report] == [
        ("SEMANTIQUE_CONTRADICTION", "blocking", "sémantique", "specify")]


def test_semantic_guardian_cannot_invent_ids():
    auditor = FakeLLM({"SemanticReview": [SemanticReview(findings=[
        {"kind": "incoherence", "severity": "blocking", "refs": ["REQ-099"], "message": "inventé"}])] * 2})
    report = build_guardian(auditor).invoke({"stage": "plan", "artifacts": guardian_artifacts(), "accepted": []})["report"]
    assert report == []  # constat qui cite un id inexistant : écarté (après une demande de correction)
    assert auditor.calls["SemanticReview"] == 2


def test_semantic_guardian_failure_is_reported_not_hidden():
    class Down:
        def with_structured_output(self, *_a, **_k):
            from langchain_core.runnables import RunnableLambda

            def boom(_):
                raise RuntimeError("quota")
            return RunnableLambda(boom)

    report = build_guardian(Down()).invoke({"stage": "spec", "artifacts": guardian_artifacts(), "accepted": []})["report"]
    assert [(a["code"], a["severity"]) for a in report] == [("GARDIEN_SEMANTIQUE_INDISPONIBLE", "warning")]


def test_semantic_check_is_skipped_when_deterministic_checks_already_block():
    auditor = FakeLLM({"SemanticReview": []})
    arts = guardian_artifacts() | {"clarifications": [{"id": "Q-001", "question": "?", "source_ids": []}]}
    report = build_guardian(auditor).invoke({"stage": "spec", "artifacts": arts, "accepted": []})["report"]
    assert report[0]["code"] == "QUESTION_OUVERTE" and auditor.calls["SemanticReview"] == 0  # quota préservé


def test_semantic_warning_can_be_acknowledged_in_the_full_graph(tmp_path):
    llm = FakeLLM({"Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [GOOD_PLAN],
                   "TaskList": [GOOD_TASKS], "Implementation": [
                       Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "# store\n"}]),
                       Implementation(task_id="T-002", files=[{"path": "todo/cli.py", "content": "# cli\n"}])]})
    warn = SemanticReview(findings=[{"kind": "hors_perimetre", "severity": "warning", "refs": ["PLN-002"],
                                     "message": "PLN-002 décrit une CLI alors que le brief ne parle pas d'interface."}])
    auditor = FakeLLM({"SemanticReview": [SemanticReview(), warn, warn]})
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver(), guardian_llm=auditor)
    cfg = {"configurable": {"thread_id": "s"}}
    res = g.invoke({"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    pause = res["__interrupt__"][0].value
    assert pause["stage"] == "plan" and not pause["blocking"] and pause["alerts"][0]["origin"] == "sémantique"
    res = g.invoke(Command(resume={"action": "continue"}), cfg)  # acquitté : ne repause pas à l'analyse
    assert "__interrupt__" not in res and (tmp_path / "workspace/todo/cli.py").exists()


def test_semantic_guardian_uses_backup_model_when_its_model_is_out_of_quota():
    """Quota du Gardien épuisé : il bascule sur le modèle de secours au lieu de sauter le contrôle sémantique."""
    from langchain_core.runnables import RunnableLambda

    from sdd.guardian import semantic_alerts
    from sdd.models import SemanticReview

    class Quota(Exception):
        status_code = 429

    class M:
        def __init__(self, name, out):
            self.model, self.out = name, out

        def with_structured_output(self, schema, **_kw):
            def run(_i):
                if isinstance(self.out, Exception):
                    raise self.out
                return self.out
            return RunnableLambda(run)

    a = {"sources": [], "spec": {"requirements": []}}
    dead = M("gemini-3.8-flash", Quota("429 RESOURCE_EXHAUSTED quota PerDay"))
    backup = M("gemini-3.5-flash", SemanticReview(findings=[]))
    alerts = semantic_alerts(dead, "spec", a, 0, fallback_llm=backup)
    assert not any(x["code"] == "GARDIEN_SEMANTIQUE_INDISPONIBLE" for x in alerts)
