"""Rendu Markdown des artefacts (déterministe) + écriture disque."""
from __future__ import annotations

from pathlib import Path


def write(base: Path, rel: str, content: str) -> Path:
    p = Path(base) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _errors(state, key) -> str:
    errs = state.get(key) or []
    if not errs:
        return "\n> ✅ Validation déterministe : OK\n"
    return "\n> ⚠️ Erreurs non résolues après correction automatique :\n" + "".join(f"> - {e}\n" for e in errs)


def _questions(qs) -> str:
    if not qs:
        return ""
    lines = ["\n## Questions ouvertes\n"]
    for q in qs:
        src = ", ".join(q.get("source_ids", []))
        lines.append(f"- **{q['id']}** {q['question']}" + (f" _(sources : {src})_" if src else ""))
    return "\n".join(lines) + "\n"


def spec_md(state, questions=None) -> str:
    spec = state["spec"]
    out = ["# Spécification\n", _errors(state, "spec_errors"), "\n## Exigences\n"]
    for r in spec["requirements"]:
        cites = "; ".join(f"{c['source_id']} « {c['quote']} »" for c in r["citations"])
        out.append(f"### {r['id']} ({r['kind']})\n{r['statement']}\n\n_Sources : {cites}_\n")
        for i, ac in enumerate(r["acceptance_criteria"], 1):
            out.append(f"- CA{i} : {ac['text']}")
        out.append("")
    out.append(_questions(questions if questions is not None else spec.get("open_questions")))
    out.append("\n## Sources\n")
    out += [f"- `{s['id']}` ({s['origin']}) {s['text']}" for s in state["sources"]]
    return "\n".join(out) + "\n"


def plan_md(state) -> str:
    p = state["plan"]
    out = ["# Plan technique\n", _errors(state, "plan_errors"), "\n## Éléments\n"]
    for it in p["items"]:
        out.append(f"### {it['id']} [{it['kind']}] {it['title']}\n{it['description']}\n\n_Exigences : {', '.join(it['requirement_ids'])}_\n")
    if p["tech_choices"]:
        out.append("## Choix techniques\n")
        out += [f"- {t['name']} _(sources : {', '.join(t['source_ids'])})_" for t in p["tech_choices"]]
    out.append(_questions(p.get("open_questions")))
    return "\n".join(out) + "\n"


def tasks_md(state) -> str:
    tl = state["tasks"]["tasks"]
    out = ["# Tâches\n", _errors(state, "tasks_errors"), f"\nOrdre d'exécution : {' → '.join(state.get('task_order', []))}\n"]
    for t in tl:
        out.append(f"### {t['id']} {t['title']}\n{t['description']}\n")
        out.append(f"- Exigences : {', '.join(t['requirement_ids'])}")
        out.append(f"- Plan : {', '.join(t['plan_item_ids'])}")
        out.append(f"- Dépend de : {', '.join(t['depends_on']) or '—'}")
        out.append(f"- Fichiers : {', '.join(t['files'])}")
        for ac in t["acceptance_criteria"]:
            out.append(f"- [ ] {ac}")
        out.append("")
    # Matrice de traçabilité source -> exigence -> tâche
    out.append("## Matrice de traçabilité\n\n| Exigence | Sources | Tâches |\n|---|---|---|")
    for r in state["spec"]["requirements"]:
        srcs = sorted({c["source_id"] for c in r["citations"]})
        tids = [t["id"] for t in tl if r["id"] in t["requirement_ids"]]
        out.append(f"| {r['id']} | {', '.join(srcs)} | {', '.join(tids) or '❌'} |")
    return "\n".join(out) + "\n"


def analyze_md(state, res) -> str:
    out = ["# Analyse de cohérence (spec ↔ plan ↔ tâches)\n"]
    out.append("## Erreurs\n" + ("\n".join(f"- ❌ {e}" for e in res["errors"]) or "Aucune ✅") + "\n")
    out.append("## Avertissements\n" + ("\n".join(f"- ⚠️ {w}" for w in res["warnings"]) or "Aucun") + "\n")
    req = state["spec"]["requirements"]
    tl = state["tasks"]["tasks"]
    covered = sum(1 for r in req if any(r["id"] in t["requirement_ids"] for t in tl))
    out.append(f"## Couverture\n{covered}/{len(req)} exigences couvertes par au moins une tâche.\n")
    return "\n".join(out)
