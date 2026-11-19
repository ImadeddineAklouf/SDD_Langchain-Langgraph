"""Tests sans réseau : un faux LLM renvoie des sorties scriptées."""
from collections import defaultdict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from sdd.graph import build_graph
from sdd.ingest import split_statements
from sdd.models import Clarifications, Implementation, Plan, Spec, TaskList
from sdd.validators import validate_plan, validate_spec, validate_tasks

BRIEF = """# Todo
- L'utilisateur peut ajouter une tâche avec un titre.
- Les tâches sont sauvegardées dans un fichier JSON local.
"""
CONSTITUTION = "- Le code est écrit en Python 3.11."


class FakeLLM:
    def __init__(self, scripted: dict):
        self.queues = defaultdict(list, {k: list(v) for k, v in scripted.items()})
        self.calls = defaultdict(int)

    def with_structured_output(self, schema, **_kw):
        from langchain_core.runnables import RunnableLambda

        def answer(_prompt_value):
            self.calls[schema.__name__] += 1
            return self.queues[schema.__name__].pop(0)

        return RunnableLambda(answer)


def cite(sid, quote):
    return {"source_id": sid, "quote": quote}


GOOD_SPEC = Spec.model_validate({
    "requirements": [
        {"id": "REQ-001", "kind": "functional", "statement": "L'utilisateur peut ajouter une tâche avec un titre.",
         "citations": [cite("B-001", "ajouter une tâche avec un titre")],
         "acceptance_criteria": [{"text": "ajouter une tâche avec un titre", "citations": [cite("B-001", "ajouter une tâche avec un titre")]}]},
        {"id": "REQ-002", "kind": "constraint", "statement": "Les tâches sont sauvegardées dans un fichier JSON local.",
         "citations": [cite("B-002", "fichier JSON local")]},
    ],
})
HALLUCINATED_SPEC = Spec.model_validate({
    "requirements": [
        {"id": "REQ-001", "kind": "functional", "statement": "Ajouter une tâche avec titre, priorité et date d'échéance via OAuth.",
         "citations": [cite("B-001", "ajouter une tâche avec un titre")]},
    ],
})
GOOD_PLAN = Plan.model_validate({
    "items": [
        {"id": "PLN-001", "kind": "component", "title": "Stockage", "description": "JSON", "requirement_ids": ["REQ-002"]},
        {"id": "PLN-002", "kind": "interface", "title": "CLI add", "description": "add", "requirement_ids": ["REQ-001"]},
    ],
    "tech_choices": [{"name": "Python 3.11", "source_ids": ["C-001"]}],
})
GOOD_TASKS = TaskList.model_validate({
    "tasks": [
        {"id": "T-001", "title": "Stockage", "description": "store", "requirement_ids": ["REQ-002"],
         "plan_item_ids": ["PLN-001"], "files": ["todo/store.py"]},
        {"id": "T-002", "title": "Commande add", "description": "add", "requirement_ids": ["REQ-001"],
         "plan_item_ids": ["PLN-002"], "depends_on": ["T-001"], "files": ["todo/cli.py"],
         "acceptance_criteria": ["ajouter une tâche avec un titre"]},
    ]
})


def sources():
    return split_statements(CONSTITUTION, "C", "constitution") + split_statements(BRIEF, "B", "brief")


# ---------------------------- validateurs ---------------------------------- #
def test_ingest_is_deterministic():
    assert [s.id for s in sources()] == ["C-001", "B-001", "B-002"]
    assert sources() == sources()


def test_good_spec_passes():
    assert validate_spec(GOOD_SPEC, sources()) == []


def test_hallucinated_spec_rejected():
    errs = validate_spec(HALLUCINATED_SPEC, sources())
    assert any("termes absents" in e for e in errs)
    assert any("B-002" in e and "couverte" in e for e in errs)


def test_fake_quote_rejected():
    spec = GOOD_SPEC.model_copy(deep=True)
    spec.requirements[0].citations[0].quote = "supprimer une tâche"
    assert any("mot pour mot" in e for e in validate_spec(spec, sources()))


def test_invented_tech_rejected():
    plan = GOOD_PLAN.model_copy(deep=True)
    plan.tech_choices[0].name = "FastAPI"
    assert any("FastAPI" in e for e in validate_plan(plan, GOOD_SPEC, sources()))


def test_task_coverage_and_invented_criteria():
    tl = TaskList.model_validate({"tasks": [GOOD_TASKS.tasks[1].model_dump() | {"depends_on": [],
                                  "acceptance_criteria": ["exporter en CSV"]}]})
    errs = validate_tasks(tl, GOOD_PLAN, GOOD_SPEC)
    assert any("REQ-002" in e for e in errs)
    assert any("exporter en CSV" in e for e in errs)


def test_out_of_scope_file_rejected():
    from sdd.validators import validate_implementation
    impl = Implementation(task_id="T-001", files=[{"path": "setup.py", "content": ""}])
    assert any("hors du périmètre" in e for e in validate_implementation(impl, GOOD_TASKS, "T-001"))


# ---------------------------- graphe complet -------------------------------- #
def run_until_pause(g, inp, cfg):
    res = g.invoke(inp, cfg)
    return res, (res["__interrupt__"][0].value if "__interrupt__" in res else None)


def test_factory_mode_runs_end_to_end_without_pause_when_guardian_is_happy(tmp_path):
    llm = FakeLLM({
        "Spec": [HALLUCINATED_SPEC, GOOD_SPEC],  # 1er essai rejeté par le validateur -> correction
        "Clarifications": [Clarifications()],
        "Plan": [GOOD_PLAN],
        "TaskList": [GOOD_TASKS],
        "Implementation": [
            Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "# store\n"}]),
            Implementation(task_id="T-002", files=[{"path": "todo/cli.py", "content": "# cli\n"}]),
        ],
    })
    g = build_graph(llm, tmp_path, max_retries=2, checkpointer=MemorySaver())
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, {"configurable": {"thread_id": "t"}})
    assert pause is None  # mode usine : aucune alerte => aucune pause
    assert llm.calls["Spec"] == 2
    assert res["task_order"] == ["T-001", "T-002"]
    assert (tmp_path / "workspace/todo/cli.py").read_text() == "# cli\n"
    assert "| REQ-001 | B-001 | T-002 |" in (tmp_path / "tasks.md").read_text()
    controls = [e["stage"] for e in res["audit"] if e["type"] == "contrôle"]
    assert controls == ["spec", "plan", "analyze", "impl", "impl", "verify"]  # le Gardien contrôle chaque étape


def test_supervised_mode_pauses_at_each_stage(tmp_path):
    from sdd.graph import SUPERVISED

    llm = FakeLLM({
        "Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [GOOD_PLAN], "TaskList": [GOOD_TASKS],
        "Implementation": [
            Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "# store\n"}]),
            Implementation(task_id="T-002", files=[{"path": "todo/cli.py", "content": "# cli\n"}]),
        ],
    })
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver(), checkpoints=SUPERVISED)
    cfg = {"configurable": {"thread_id": "t"}}
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    stages = []
    while pause:
        stages.append(pause["stage"])
        assert pause["kind"] == "checkpoint" and pause["alerts"] == []
        if pause["stage"] == "impl":  # rien n'est écrit avant la décision
            assert not (tmp_path / "workspace" / pause["files"][0]).exists()
        res, pause = run_until_pause(g, Command(resume={"action": "continue"}), cfg)
    assert stages == ["spec", "plan", "analyze", "impl", "impl", "verify"]


SPEC_Q = Spec.model_validate(GOOD_SPEC.model_dump() | {
    "open_questions": [{"id": "Q-001", "question": "Quel format de titre ?", "source_ids": ["B-001"]}]})


def spec_citing_answer(aid="A-001", quote="Réponse : texte libre"):
    d = GOOD_SPEC.model_dump()
    d["requirements"][0]["citations"].append(cite(aid, quote))
    return Spec.model_validate(d)


def test_human_answers_become_citable_sources_and_unanswered_questions_persist(tmp_path):
    llm = FakeLLM({"Spec": [SPEC_Q, spec_citing_answer()], "Clarifications": [
        Clarifications(questions=[{"id": "Q-001", "question": "Combien de tâches au maximum ?", "source_ids": ["B-001"]}]),
        Clarifications()]})
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    assert pause["kind"] == "alert" and pause["blocking"]
    assert [q["id"] for q in pause["open_questions"]] == ["Q-001", "Q-002"]  # spec + clarify, renumérotées
    assert {a["code"] for a in pause["alerts"]} == {"QUESTION_OUVERTE"}
    res, pause = run_until_pause(g, Command(resume={"action": "answer", "answers": {"Q-001": "texte libre"}}), cfg)
    assert any(s["id"] == "A-001" and "texte libre" in s["text"] for s in res["sources"])
    # la question sans réponse ne disparaît pas, même si le LLM ne la repose pas
    assert [q["question"] for q in pause["open_questions"]] == ["Combien de tâches au maximum ?"]
    assert llm.calls["Implementation"] == 0
    res, pause = run_until_pause(g, Command(resume={"action": "abort"}), cfg)
    assert pause is None


def test_typed_answers_are_never_lost_even_with_continue(tmp_path):
    """Bug du dernier test : réponses saisies puis « Approuver » => réponses perdues. Désormais impossible."""
    llm = FakeLLM({"Spec": [SPEC_Q, spec_citing_answer()], "Clarifications": [Clarifications(), Clarifications()]})
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver(), checkpoints=("spec",))
    cfg = {"configurable": {"thread_id": "t"}}
    run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    res, pause = run_until_pause(g, Command(resume={"action": "continue", "answers": {"Q-001": "texte libre"}}), cfg)
    assert pause["kind"] == "checkpoint"  # spec complète : plus d'alerte
    assert any(s["id"] == "A-001" for s in res["sources"])
    assert llm.calls["Spec"] == 2  # specify relancé avec la réponse
    assert res["audit"][-2]["action"] == "answer"


def test_ignored_answer_is_blocking(tmp_path):
    """Piège : la spec régénérée ne reprend pas la réponse humaine."""
    llm = FakeLLM({"Spec": [SPEC_Q, GOOD_SPEC, GOOD_SPEC], "Clarifications": [Clarifications(), Clarifications()]})
    g = build_graph(llm, tmp_path, max_retries=1, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    res, pause = run_until_pause(g, Command(resume={"action": "answer", "answers": {"Q-001": "texte libre"}}), cfg)
    assert pause["blocking"] and [a["code"] for a in pause["alerts"]] == ["REPONSE_IGNOREE"]
    assert "A-001" in pause["alerts"][0]["refs"] and pause["alerts"][0]["redo"] == "specify"


def test_implementation_cannot_start_while_a_question_is_open(tmp_path):
    llm = FakeLLM({"Spec": [SPEC_Q], "Clarifications": [Clarifications()], "Plan": [GOOD_PLAN],
                   "TaskList": [GOOD_TASKS], "Implementation": []})
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    for bad in ({"action": "continue"}, {"action": "accept_risk"}, {"action": "answer", "answers": {"Q-001": "   "}},
                {"action": "correct", "feedback": ""}):
        res, pause = run_until_pause(g, Command(resume=bad), cfg)
        assert pause and pause["stage"] == "spec" and pause["gate_error"], bad  # décision refusée, toujours en pause
    assert llm.calls["Plan"] == 0 and llm.calls["Implementation"] == 0
    refused = [e for e in res["audit"] if e["type"] == "décision" and e["refused"]]
    assert len(refused) == 4  # chaque refus est tracé


def test_accepted_risk_is_audited_and_not_raised_again(tmp_path):
    llm = FakeLLM({"Spec": [SPEC_Q], "Clarifications": [Clarifications()], "Plan": [GOOD_PLAN],
                   "TaskList": [GOOD_TASKS], "Implementation": [
                       Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "# store\n"}]),
                       Implementation(task_id="T-002", files=[{"path": "todo/cli.py", "content": "# cli\n"}])]})
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    res, pause = run_until_pause(
        g, Command(resume={"action": "accept_risk", "justification": "Titre libre, sans contrainte de format."}), cfg)
    assert pause is None  # la même question n'est plus bloquante aux étapes suivantes
    accepted = [a for e in res["audit"] if e["type"] == "contrôle" and e["stage"] == "analyze" for a in e["alerts"]]
    assert [(a["code"], a["severity"]) for a in accepted] == [("QUESTION_OUVERTE", "info")]
    decision = next(e for e in res["audit"] if e["type"] == "décision")
    assert decision["action"] == "accept_risk" and "Titre libre" in decision["justification"]
    assert "accept_risk" in (tmp_path / "audit.md").read_text()


def test_plan_question_blocks_and_answer_resumes_at_plan(tmp_path):
    plan_q = Plan.model_validate(GOOD_PLAN.model_dump() | {
        "open_questions": [{"id": "Q-001", "question": "Quel nom pour le fichier JSON ?", "source_ids": ["B-002"]}]})
    llm = FakeLLM({"Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [plan_q, GOOD_PLAN]})
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver(), checkpoints=("plan",))
    cfg = {"configurable": {"thread_id": "t"}}
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    assert pause["stage"] == "plan" and pause["open_questions"][0]["id"] == "QP-001"
    res, pause = run_until_pause(g, Command(resume={"action": "answer", "answers": {"QP-001": "todo.json"}}), cfg)
    assert llm.calls["Spec"] == 1 and llm.calls["Plan"] == 2  # reprise au bon nœud : plan, pas specify
    assert pause["kind"] == "checkpoint" and any(s["id"] == "A-001" for s in res["sources"])


def test_correction_resumes_the_faulty_task_with_feedback(tmp_path):
    llm = FakeLLM({
        "Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [GOOD_PLAN], "TaskList": [GOOD_TASKS],
        "Implementation": [
            Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "import requests\n"}]),
            Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "import json\n"}]),
            Implementation(task_id="T-002", files=[{"path": "todo/cli.py", "content": "# cli\n"}]),
        ],
    })
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    assert pause["stage"] == "impl" and pause["alerts"][0]["code"] == "DEPENDANCE_NON_SOURCEE"
    assert not (tmp_path / "workspace/todo/store.py").exists()  # rien d'écrit pendant la pause
    res, pause = run_until_pause(g, Command(resume={"action": "correct", "feedback": "Utilise uniquement la bibliothèque standard."}), cfg)
    assert pause is None
    assert any(s["id"] == "F-001" for s in res["sources"])
    assert (tmp_path / "workspace/todo/store.py").read_text() == "import json\n"


def test_verify_runs_tests_and_fix_cannot_touch_tests(tmp_path):
    tasks = TaskList.model_validate({"tasks": [
        {"id": "T-001", "title": "Stockage", "description": "store", "requirement_ids": ["REQ-002"],
         "plan_item_ids": ["PLN-001"], "files": ["store.py", "tests/test_store.py"]},
        {"id": "T-002", "title": "add", "description": "add", "requirement_ids": ["REQ-001"],
         "plan_item_ids": ["PLN-002"], "depends_on": ["T-001"], "files": ["store.py"],
         "acceptance_criteria": ["ajouter une tâche avec un titre"]},
    ]})
    test_code = "from store import add\n\ndef test_add():\n    assert add([], 'x') == ['x']\n"
    llm = FakeLLM({
        "Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [GOOD_PLAN], "TaskList": [tasks],
        "Implementation": [
            Implementation(task_id="T-001", files=[{"path": "store.py", "content": "def add(l, t):\n    return l\n"},
                                                   {"path": "tests/test_store.py", "content": test_code}]),
            Implementation(task_id="T-002", files=[{"path": "store.py", "content": "def add(l, t):\n    return l\n"}]),
            # 1er correctif : tente de modifier le test -> rejeté par le validateur
            Implementation(task_id="FIX", files=[{"path": "tests/test_store.py", "content": "def test_add():\n    pass\n"}]),
            # 2e : correctif légitime
            Implementation(task_id="FIX", files=[{"path": "store.py", "content": "def add(l, t):\n    return l + [t]\n"}]),
        ],
    })
    g = build_graph(llm, tmp_path, max_retries=2, checkpointer=MemorySaver())
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, {"configurable": {"thread_id": "t"}})
    assert pause is None and res["verify"]["passed"] is True
    assert res["fix_attempts"] == 1
    assert (tmp_path / "workspace/tests/test_store.py").read_text() == test_code


def test_failing_tests_after_fixes_raise_a_blocking_alert(tmp_path):
    tasks = TaskList.model_validate({"tasks": [
        {"id": "T-001", "title": "Stockage", "description": "store", "requirement_ids": ["REQ-002", "REQ-001"],
         "plan_item_ids": ["PLN-001", "PLN-002"], "files": ["store.py", "tests/test_store.py"],
         "acceptance_criteria": ["ajouter une tâche avec un titre"]}]})
    test_code = "def test_x():\n    assert False\n"
    llm = FakeLLM({"Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [GOOD_PLAN], "TaskList": [tasks],
                   "Implementation": [Implementation(task_id="T-001", files=[
                       {"path": "store.py", "content": "X = 1\n"}, {"path": "tests/test_store.py", "content": test_code}]),
                       Implementation(task_id="FIX", files=[{"path": "store.py", "content": "X = 2\n"}])]})
    g = build_graph(llm, tmp_path, max_retries=0, max_fix_attempts=1, checkpointer=MemorySaver())
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, {"configurable": {"thread_id": "t"}})
    assert pause["stage"] == "verify" and pause["passed"] is False
    assert [a["code"] for a in pause["alerts"]] == ["TESTS_EN_ECHEC"]


def test_escaped_newlines_rejected():
    from sdd.validators import validate_implementation
    impl = Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "import os\\nprint(1)\\n"}])
    assert any("sauts de ligne" in e for e in validate_implementation(impl, GOOD_TASKS, "T-001"))


def test_require_tests_option():
    errs = validate_tasks(GOOD_TASKS, GOOD_PLAN, GOOD_SPEC, require_tests=True)
    assert any("REQ-001" in e and "test" in e for e in errs)
    assert validate_tasks(GOOD_TASKS, GOOD_PLAN, GOOD_SPEC) == []


def test_fallback_model_used_when_primary_fails():
    from sdd.llm import generate

    class Broken:
        def with_structured_output(self, schema, **_kw):
            from langchain_core.runnables import RunnableLambda

            def boom(_):
                raise RuntimeError("429 quota atteint")

            return RunnableLambda(boom)

    backup = FakeLLM({"Spec": [GOOD_SPEC]})
    spec, errs = generate(Broken(), Spec, "sys", "x", lambda s: validate_spec(s, sources()), 0, fallback_llm=backup)
    assert errs == [] and backup.calls["Spec"] == 1


def test_constitution_rule_cannot_be_functional():
    spec = Spec.model_validate(GOOD_SPEC.model_dump() | {"requirements": GOOD_SPEC.model_dump()["requirements"] + [
        {"id": "REQ-003", "kind": "functional", "statement": "Le code est écrit en Python 3.11.",
         "citations": [cite("C-001", "Le code est écrit en Python 3.11.")]}]})
    assert any("REQ-003" in e and "constitution" in e for e in validate_spec(spec, sources()))
