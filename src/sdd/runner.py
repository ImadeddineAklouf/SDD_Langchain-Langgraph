"""Pilotage pas à pas du graphe, partagé par l'interface web (sans dépendance à Streamlit).

`advance()` exécute le graphe jusqu'à la prochaine pause (alerte du Gardien, interrupt) ou jusqu'à la fin,
en signalant chaque nœud terminé. Elle n'ajoute aucune donnée : elle ne fait que transmettre
l'entrée initiale (brief + constitution) ou la décision humaine (`Command(resume=...)`).
"""
from __future__ import annotations

from typing import Any, Callable

from langgraph.types import Command

# Les 7 étapes de spec-kit et les nœuds du graphe qui les réalisent.
STEPS: list[tuple[str, str]] = [
    ("constitution", "1. Constitution"),
    ("specify", "2. Specify"),
    ("clarify", "3. Clarify"),
    ("plan", "4. Plan"),
    ("tasks", "5. Tasks"),
    ("analyze", "6. Analyze"),
    ("implement", "7. Implement"),
    ("guardian", "🛡️ Gardien"),
]
NODE_TO_STEP = {
    "constitution": "constitution",
    "specify": "specify",
    "clarify": "clarify",
    "plan": "plan",
    "tasks": "tasks",
    "analyze": "analyze",
    "implement": "implement",
    "apply_impl": "implement",
    "verify": "implement",
    "fix": "implement",
    "guardian": "guardian",
    "gate": "guardian",
}
STAGE_TITLES = {
    "spec": "Spécification (specify + clarify)",
    "plan": "Plan technique",
    "analyze": "Tâches et analyse croisée spec ↔ plan ↔ tâches",
    "impl": "Implémentation de la tâche",
    "verify": "Vérification (tests)",
}
ACTIONS = {
    "continue": "continuer", "answer": "répondre", "correct": "corriger",
    "accept_risk": "accepter le risque", "abort": "arrêter", "invalid": "décision refusée",
}


def summarize(node: str, update: dict[str, Any] | None) -> str:
    """Résumé lisible d'une mise à jour d'état (purement descriptif)."""
    u = update or {}
    if node == "constitution":
        return f"{len(u.get('sources', []))} règle(s) de constitution découpée(s), sans LLM"
    if node == "specify":
        n = len(u.get("spec", {}).get("requirements", []))
        return f"{n} exigence(s), {len(u.get('spec_errors', []))} erreur(s) de validation restante(s)"
    if node == "clarify":
        return f"{len(u.get('clarifications', []))} question(s) ouverte(s)"
    if node == "plan":
        p = u.get("plan", {})
        return f"{len(p.get('items', []))} élément(s) de plan, {len(u.get('plan_errors', []))} erreur(s)"
    if node == "tasks":
        return f"{len(u.get('tasks', {}).get('tasks', []))} tâche(s), {len(u.get('tasks_errors', []))} erreur(s)"
    if node == "analyze":
        a = u.get("analysis", {})
        return f"{len(a.get('errors', []))} erreur(s), {len(a.get('warnings', []))} avertissement(s)"
    if node == "implement":
        impl = u.get("impl", {})
        files = ", ".join(f["path"] for f in impl.get("files", []))
        return f"{impl.get('task_id', '?')} → {files or 'aucun fichier'}"
    if node == "verify":
        v = u.get("verify", {})
        if not v.get("ran", True):
            return "aucun test à lancer"
        return "tests OK" if v.get("passed") else "tests en échec"
    if node == "fix":
        return f"tentative de correction n°{u.get('fix_attempts', '?')}"
    if node == "apply_impl":
        return "fichiers écrits dans le workspace"
    if node == "guardian":
        al = u.get("alerts", [])
        n = {s: sum(1 for a in al if a["severity"] == s) for s in ("blocking", "warning", "info")}
        if not n["blocking"] and not n["warning"]:
            return f"aucune incohérence ({n['info']} info)"
        return f"⛔ {n['blocking']} bloquante(s), ⚠️ {n['warning']} avertissement(s) : pause"
    if node == "gate":
        return f"décision humaine : {ACTIONS.get(u.get('decision', ''), u.get('decision', '?'))}"
    return ""


def advance(graph, inp, cfg: dict, on_update: Callable[[str, dict], None] | None = None) -> dict | None:
    """Exécute jusqu'au prochain interrupt. Renvoie sa charge utile, ou None si le graphe est terminé.

    `inp` vaut l'entrée initiale ({brief, constitution}), un `Command(resume=...)`, ou None pour
    reprendre une session existante depuis son checkpoint.
    """
    for chunk in graph.stream(inp, cfg, stream_mode="updates"):
        for node, update in chunk.items():
            if node != "__interrupt__" and on_update:
                on_update(node, update or {})
    snap = graph.get_state(cfg)
    return snap.interrupts[0].value if snap.interrupts else None


def resume(graph, cfg: dict, review: dict, on_update=None) -> dict | None:
    """Transmet la décision humaine (continue / answer / correct / accept_risk / abort) à la porte du Gardien."""
    return advance(graph, Command(resume=review), cfg, on_update)
