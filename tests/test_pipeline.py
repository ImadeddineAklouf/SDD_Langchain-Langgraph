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
def test_full_pipeline_with_retry_and_human_checkpoints(tmp_path):
    llm = FakeLLM({
        "Spec": [HALLUCINATED_SPEC, GOOD_SPEC],  # 1er essai rejeté -> correction
        "Clarifications": [Clarifications()],
        "Plan": [GOOD_PLAN],
        "TaskList": [GOOD_TASKS],
        "Implementation": [
            Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "# store\n"}]),
            Implementation(task_id="T-002", files=[{"path": "todo/cli.py", "content": "# cli\n"}]),
        ],
    })
    g = build_graph(llm, tmp_path, max_retries=2, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    res = g.invoke({"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    stages = []
    while "__interrupt__" in res:
        payload = res["__interrupt__"][0].value
        stages.append(payload["stage"])
        assert payload["errors"] == []
        res = g.invoke(Command(resume={"action": "approve"}), cfg)

    assert stages == ["clarify", "plan", "analyze", "impl", "impl", "verify"]
    assert llm.calls["Spec"] == 2
    assert res["task_order"] == ["T-001", "T-002"]
    assert (tmp_path / "workspace/todo/cli.py").read_text() == "# cli\n"
    assert "| REQ-001 | B-001 | T-002 |" in (tmp_path / "tasks.md").read_text()


def test_human_answers_become_citable_sources(tmp_path):
    spec_q = Spec.model_validate(GOOD_SPEC.model_dump() | {
        "open_questions": [{"id": "Q-001", "question": "Quel format de titre ?", "source_ids": ["B-001"]}]})
    llm = FakeLLM({"Spec": [spec_q, GOOD_SPEC], "Clarifications": [
        Clarifications(questions=[{"id": "Q-001", "question": "Combien de tâches au maximum ?", "source_ids": ["B-001"]}]),
        Clarifications()]})
    g = build_graph(llm, tmp_path, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    res = g.invoke({"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    qs = res["__interrupt__"][0].value["open_questions"]
    assert [q["id"] for q in qs] == ["Q-001", "Q-002"]  # spec + clarify, renumérotées
    res = g.invoke(Command(resume={"action": "revise", "answers": {"Q-001": "texte libre"}}), cfg)
    assert any(s["id"] == "A-001" and "texte libre" in s["text"] for s in res["sources"])
    res = g.invoke(Command(resume={"action": "abort"}), cfg)
    assert "__interrupt__" not in res


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
    cfg = {"configurable": {"thread_id": "t"}}
    res = g.invoke({"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    while res["__interrupt__"][0].value["stage"] != "verify":
        res = g.invoke(Command(resume={"action": "approve"}), cfg)
    assert res["__interrupt__"][0].value["passed"] is True
    assert res["fix_attempts"] == 1
    assert (tmp_path / "workspace/tests/test_store.py").read_text() == test_code


def test_escaped_newlines_rejected():
    from sdd.validators import validate_implementation
    impl = Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "import os\\nprint(1)\\n"}])
    assert any("sauts de ligne" in e for e in validate_implementation(impl, GOOD_TASKS, "T-001"))


def test_require_tests_option():
    errs = validate_tasks(GOOD_TASKS, GOOD_PLAN, GOOD_SPEC, require_tests=True)
    assert any("REQ-001" in e and "test" in e for e in errs)
    assert validate_tasks(GOOD_TASKS, GOOD_PLAN, GOOD_SPEC) == []


def test_fallback_model_used_when_primary_fails():
    from sdd.graph import generate

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
