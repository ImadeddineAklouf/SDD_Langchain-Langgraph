"""Le Correcteur (3e agent) : patchs ciblés, revalidés par le code, jamais d'invention."""
import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from sdd.corrector import apply_corrections, apply_patch, correctable
from sdd.graph import build_graph
from sdd.ingest import split_statements
from sdd.models import Clarifications, Correction, PatchOp, Plan, Spec
from sdd.validators import validate_plan, validate_spec
from test_pipeline import BRIEF, CONSTITUTION, GOOD_PLAN, GOOD_SPEC, FakeLLM, cite, run_until_pause

SOURCES = split_statements(CONSTITUTION, "C", "constitution") + split_statements(BRIEF, "B", "brief")
BAD_PLAN = Plan.model_validate(GOOD_PLAN.model_dump() | {"items": [
    GOOD_PLAN.items[0].model_dump(),
    GOOD_PLAN.items[1].model_dump() | {"requirement_ids": ["REQ-002"]},  # REQ-001 n'est plus couverte
]})
FIX = PatchOp(op="append", path="items/PLN-002/requirement_ids", value='"REQ-001"',
              reason="PLAN_INVALIDE : REQ-001 non couverte, PLN-002 (CLI add) la réalise")
INVENTED = PatchOp(op="append", path="tech_choices", value=json.dumps({"name": "Django", "source_ids": ["C-001"]}),
                   reason="ajout d'un framework")


def op(o, path, value="", reason="r"):
    return PatchOp(op=o, path=path, value=value, reason=reason)


def test_apply_patch_targets_elements_by_id_and_rejects_bad_paths():
    doc = GOOD_SPEC.model_dump()
    d = apply_patch(doc, op("set", "requirements/REQ-002/kind", '"non_functional"'))
    assert d["requirements"][1]["kind"] == "non_functional" and doc["requirements"][1]["kind"] == "constraint"
    d = apply_patch(doc, op("append", "requirements/REQ-001/citations", json.dumps(cite("B-002", "fichier JSON local"))))
    assert len(d["requirements"][0]["citations"]) == 2
    d = apply_patch(doc, op("remove", "requirements/REQ-002"))
    assert [r["id"] for r in d["requirements"]] == ["REQ-001"]
    for bad in (op("set", "requirements/REQ-999/kind", '"x"'), op("append", "requirements/REQ-001/kind", '"x"'),
                op("set", "requirements/REQ-001/kind", "{pas du json")):
        try:
            apply_patch(doc, bad)
            raise AssertionError("aurait dû être rejeté")
        except ValueError:
            pass


def test_only_patches_that_reduce_errors_are_kept_and_nothing_can_be_invented():
    spec = GOOD_SPEC
    validate = lambda p: validate_plan(p, spec, SOURCES)  # noqa: E731
    assert validate(BAD_PLAN)
    fake_quote = op("append", "items/PLN-001/requirement_ids", '"REQ-777"')
    doc, errs, log = apply_corrections(BAD_PLAN.model_dump(), [INVENTED, fake_quote, FIX], Plan, validate)
    assert errs == [] and [kept for _, kept, _ in log] == [False, False, True]
    assert all(t["name"] != "Django" for t in doc["tech_choices"])
    # une citation inventée (pas mot pour mot dans la source) ne passe pas non plus
    bad_cite = op("append", "requirements/REQ-001/citations", json.dumps(cite("B-001", "avec une priorité")))
    d2 = GOOD_SPEC.model_dump()
    d2["requirements"][0]["citations"] = []
    d2["requirements"][0]["citations"].append(cite("B-001", "ajouter une tâche avec un titre"))
    _, _, log = apply_corrections(d2, [bad_cite], Spec, lambda s: validate_spec(s, SOURCES))
    assert log[0][1] is False


def test_questions_are_never_given_to_the_corrector():
    alerts = [{"severity": "blocking", "code": "QUESTION_OUVERTE"}, {"severity": "blocking", "code": "PLAN_INVALIDE"},
              {"severity": "warning", "code": "SPEC_INVALIDE"}]
    assert {k: len(v) for k, v in correctable(alerts).items()} == {"plan": 1}


def test_graph_corrector_fixes_plan_then_guardian_rechecks(tmp_path):
    llm = FakeLLM({"Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [BAD_PLAN] * 4,
                   "Correction": [Correction(ops=[INVENTED, FIX])]})
    g = build_graph(llm, tmp_path, max_retries=0, checkpointer=MemorySaver(), checkpoints=("plan",),
                    corrector_llm=llm)
    cfg = {"configurable": {"thread_id": "t"}}
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    assert pause["stage"] == "plan" and pause["kind"] == "checkpoint" and not pause["blocking"]
    assert "REQ-001" in res["plan"]["items"][1]["requirement_ids"] and llm.calls["Correction"] == 1
    entry = next(e for e in res["audit"] if e["type"] == "correcteur")
    assert "✔ append items/PLN-002/requirement_ids" in entry["justification"]
    assert "✘ append tech_choices" in entry["justification"]
    assert "🩹" in (tmp_path / "audit.md").read_text()


def test_corrector_is_bounded_then_human_decides(tmp_path):
    llm = FakeLLM({"Spec": [GOOD_SPEC], "Clarifications": [Clarifications()], "Plan": [BAD_PLAN] * 4,
                   "Correction": [Correction(ops=[INVENTED]), Correction()]})
    g = build_graph(llm, tmp_path, max_retries=0, checkpointer=MemorySaver(), corrector_llm=llm, max_corrections=2)
    cfg = {"configurable": {"thread_id": "t"}}
    res, pause = run_until_pause(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg)
    assert llm.calls["Correction"] == 2  # 2 tours, puis pause humaine
    assert pause["blocking"] and any(a["code"] == "PLAN_INVALIDE" for a in pause["alerts"])
    res, pause = run_until_pause(g, Command(resume={"action": "abort"}), cfg)
    assert pause is None
