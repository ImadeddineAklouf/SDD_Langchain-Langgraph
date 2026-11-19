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


def test_runner_reports_nodes_and_guardian_pauses(tmp_path):
    from sdd.graph import SUPERVISED

    g = build_graph(scripted_llm(), tmp_path, checkpointer=MemorySaver(), checkpoints=SUPERVISED)
    cfg = {"configurable": {"thread_id": "ui"}}
    seen = []
    pending = advance(g, {"brief": BRIEF, "constitution": CONSTITUTION}, cfg, lambda n, u: seen.append((n, summarize(n, u))))
    assert [n for n, _ in seen] == ["constitution", "specify", "clarify", "guardian"]
    assert seen[1][1].startswith("2 exigence(s)")
    assert seen[3][1].startswith("aucune incohérence")
    stages = [pending["stage"]]
    while pending:
        pending = resume(g, cfg, {"action": "continue"}, lambda n, u: seen.append((n, summarize(n, u))))
        stages.append(pending and pending["stage"])
    assert stages == ["spec", "plan", "analyze", "impl", "impl", "verify", None]
    assert ("verify", "aucun test à lancer") in seen
    assert ("gate", "décision humaine : continuer") in seen
    assert (tmp_path / "workspace/todo/cli.py").exists()


def _app(tmp_path, monkeypatch, llm):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import sdd.llm
    import sdd.ui

    monkeypatch.setenv("SDD_UI_OUT", str(tmp_path))
    monkeypatch.setattr(sdd.llm, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sdd.llm, "get_fallback_llm", lambda: None)
    monkeypatch.setattr(sdd.llm, "get_guardian_llm", lambda: None)
    monkeypatch.setattr(sdd.llm, "ollama_models", lambda: ["qwen3:14b"])
    monkeypatch.setattr(sdd.llm, "gemini_models", lambda: ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.8-flash-lite"])
    for v in ("SDD_PROVIDER", "SDD_MODEL", "SDD_GUARDIAN_MODEL", "SDD_GUARDIAN_SEMANTIC"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    # from_file : ui.py est exécuté comme script principal, exactement comme avec `streamlit run`
    return AppTest.from_file(sdd.ui.__file__, default_timeout=30).run()


def _start(at):
    # le scénario scripté n'a pas de tâche de test : avec l'option cochée, le Gardien bloquerait (à juste titre)
    next(c for c in at.checkbox if c.label.startswith("Exiger une tâche de test")).uncheck()
    at.text_area[0].set_value(BRIEF)
    at.text_area[1].set_value(CONSTITUTION)
    next(b for b in at.button if "Lancer" in b.label).click().run()


def test_streamlit_factory_mode_runs_without_pause(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, scripted_llm())
    assert not at.exception
    # modèle choisi dans la liste réelle des modèles de la clé
    assert at.selectbox[0].value == "gemini" and at.selectbox[1].value == "gemini-3.8-flash"
    assert at.selectbox[2].value == "(aucun)"
    assert at.selectbox[3].value == "(même modèle)"  # modèle du Gardien
    assert not at.sidebar.warning  # clé présente, modèle disponible : rien à signaler
    _start(at)
    assert not at.exception, at.exception
    assert any("Workflow terminé" in s.value for s in at.success)
    assert not at.get("help_info"), "un objet Streamlit est affiché par la « magic » (doc DeltaGenerator)"
    thread = next(Path(tmp_path).iterdir())
    assert (thread / "workspace/todo/cli.py").read_text() == "# cli\n"
    assert (thread / "audit.md").exists()


def test_streamlit_alert_inbox_blocks_then_resumes_with_answer(tmp_path, monkeypatch):
    from test_pipeline import SPEC_Q, spec_citing_answer

    llm = scripted_llm()
    llm.queues["Spec"] = [SPEC_Q, spec_citing_answer()]
    llm.queues["Clarifications"] = [Clarifications(), Clarifications()]
    at = _app(tmp_path, monkeypatch, llm)
    _start(at)
    assert not at.exception, at.exception
    assert at.subheader[0].value.startswith("🚨 Boîte d'alertes du Gardien")
    assert any("QUESTION_OUVERTE" in e.value for e in at.error)
    assert next(b for b in at.button if "Continuer" in b.label).disabled  # alerte bloquante : pas de « Continuer »
    # accepter le risque sans justification : refusé par le graphe, toujours en pause
    next(b for b in at.button if "Accepter" in b.label).click().run()
    assert any("Décision refusée" in e.value for e in at.error)
    # répondre : la réponse devient A-001 et le workflow reprend jusqu'au bout
    next(t for t in at.text_input if t.label.startswith("Q-001")).set_value("texte libre")
    next(b for b in at.button if "Répondre" in b.label).click().run()
    assert not at.exception, at.exception
    assert any("Workflow terminé" in s.value for s in at.success)
    thread = next(Path(tmp_path).iterdir())
    assert "A-001" in (thread / "sources.json").read_text()
    audit = (thread / "audit.md").read_text()
    assert "accept_risk" in audit and "refusée" in audit and "answer" in audit
