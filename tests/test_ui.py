"""Tests hors ligne de l'interface web : pilotage pas à pas (runner) et application Streamlit (AppTest)."""
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from sdd.graph import build_graph
from sdd.models import Clarifications, Implementation
from sdd.runner import NODE_TO_STEP, STEPS, advance, resume, summarize
from test_pipeline import BRIEF, CONSTITUTION, GOOD_PLAN, GOOD_SPEC, GOOD_TASKS, FakeLLM


def scripted_llm():
    return FakeLLM({
        "Spec": [GOOD_SPEC],
        "Clarifications": [Clarifications()],
        "Plan": [GOOD_PLAN],
        "TaskList": [GOOD_TASKS],
        "Implementation": [
            Implementation(task_id="T-001", files=[{"path": "todo/store.py", "content": "# store\n"}]),
            Implementation(task_id="T-002", files=[{"path": "todo/cli.py", "content": "# cli\n"}]),
        ],
    })


def test_every_graph_node_maps_to_a_speckit_step(tmp_path):
    g = build_graph(scripted_llm(), tmp_path)
    nodes = set(g.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == set(NODE_TO_STEP)
    assert set(NODE_TO_STEP.values()) == {k for k, _ in STEPS}


def test_runner_stops_at_each_checkpoint_and_reports_nodes(tmp_path):
    g = build_graph(scripted_llm(), tmp_path, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "ui"}}
    seen = []
    pending = advance(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg, lambda n, u: seen.append((n, summarize(n, u))))
    assert [n for n, _ in seen] == ["constitution", "specify", "clarify"]
    assert seen[1][1].startswith("2 exigence(s)")
    stages = [pending["stage"]]
    while pending:
        pending = resume(g, cfg, {"action": "approve"}, lambda n, u: seen.append((n, summarize(n, u))))
        stages.append(pending and pending["stage"])
    assert stages == ["clarify", "plan", "analyze", "impl", "impl", "verify", None]
    assert ("verify", "aucun test à lancer") in seen
    assert (tmp_path / "workspace/todo/cli.py").exists()


def test_streamlit_app_runs_full_workflow(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import sdd.llm
    import sdd.ui

    monkeypatch.setenv("SDD_UI_OUT", str(tmp_path))
    monkeypatch.setattr(sdd.llm, "get_llm", lambda *a, **k: scripted_llm())
    monkeypatch.setattr(sdd.llm, "get_fallback_llm", lambda: None)
    monkeypatch.setattr(sdd.llm, "ollama_models", lambda: ["qwen3:14b"])
    monkeypatch.setattr(sdd.llm, "gemini_models", lambda: ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.8-flash-lite"])
    monkeypatch.delenv("SDD_PROVIDER", raising=False)
    monkeypatch.delenv("SDD_MODEL", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    # from_file : ui.py est exécuté comme script principal, exactement comme avec `streamlit run`
    at = AppTest.from_file(sdd.ui.__file__, default_timeout=30).run()
    assert not at.exception
    # modèle choisi dans la liste réelle des modèles de la clé
    assert at.selectbox[0].value == "gemini" and at.selectbox[1].value == "gemini-3.8-flash"
    assert at.selectbox[2].value == "(aucun)"
    assert not at.sidebar.warning  # clé présente, modèle disponible : rien à signaler
    at.text_area[0].set_value(BRIEF)
    at.text_area[1].set_value(CONSTITUTION)
    next(b for b in at.button if "Lancer" in b.label).click().run()
    stages = []
    for _ in range(10):
        assert not at.exception, at.exception
        if not at.subheader or not at.subheader[0].value.startswith("⏸"):
            break
        stages.append(at.subheader[0].value.split(" ")[1])
        next(b for b in at.button if "Approuver" in b.label).click().run()
    assert stages == ["Clarify", "Plan", "Analyze", "Implement", "Implement", "Verify"]
    assert any("Workflow terminé" in s.value for s in at.success)
    assert not at.get("help_info"), "un objet Streamlit est affiché par la « magic » (doc DeltaGenerator)"
    thread = next(Path(tmp_path).iterdir())
    assert (thread / "workspace/todo/cli.py").read_text() == "# cli\n"
