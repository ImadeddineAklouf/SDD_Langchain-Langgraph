"""Validateurs 100 % déterministes (aucun LLM).

Chaque fonction retourne une liste d'erreurs lisibles. Liste vide = artefact valide.
Ces erreurs sont renvoyées au LLM pour correction, puis montrées à l'humain.
"""
from __future__ import annotations

import re
import unicodedata

from .models import Implementation, Plan, SourceStatement, Spec, TaskList

# Mots-outils FR/EN ignorés pour le contrôle d'ancrage lexical.
STOPWORDS = set(
    """
    le la les un une des du de d l au aux et ou en dans par pour sur avec sans ne pas plus
    que qui quoi dont où ce cet cette ces son sa ses leur leurs il elle ils elles on nous vous
    je tu se s est sont être doit doivent peut peuvent fait faire a ont avoir tout tous toute
    toutes chaque lorsque quand si alors aussi comme afin ainsi y n c qu j m t
    the a an and or of to in on for with without by is are be must should can may it its this
    that these those each when if then as at from into not no all any
    système systeme utilisateur utilisateurs user users system
    """.split()
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[’`]", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9àâäçéèêëîïôöùûüÿœæ_-]+", _norm(text))
    return {_strip_accents(w) for w in words if len(w) > 2 and w not in STOPWORDS}


def grounding_ratio(text: str, sources: list[str]) -> tuple[float, set[str]]:
    """Part des mots porteurs de sens de `text` présents dans `sources`.

    Retourne (ratio, mots_inconnus). Un ratio bas = le LLM a ajouté du contenu.
    """
    toks = _tokens(text)
    if not toks:
        return 1.0, set()
    pool: set[str] = set()
    for s in sources:
        pool |= _tokens(s)
    # tolérance morphologique simple : préfixe commun de 5 lettres
    def known(w: str) -> bool:
        return w in pool or any(len(w) >= 5 and p.startswith(w[:5]) for p in pool)

    unknown = {w for w in toks if not known(w)}
    return 1 - len(unknown) / len(toks), unknown


def _check_citations(where: str, citations, src: dict[str, SourceStatement]) -> list[str]:
    errs = []
    for c in citations:
        if c.source_id not in src:
            errs.append(f"{where}: source inconnue '{c.source_id}'.")
        elif _norm(c.quote) not in _norm(src[c.source_id].text):
            errs.append(
                f"{where}: la citation \"{c.quote}\" n'est pas un extrait mot pour mot de {c.source_id}."
            )
    return errs


# --------------------------------------------------------------------------- #
def validate_spec(spec: Spec, sources: list[SourceStatement], min_grounding: float = 0.7) -> list[str]:
    src = {s.id: s for s in sources}
    errs: list[str] = []
    ids = [r.id for r in spec.requirements]
    if len(ids) != len(set(ids)):
        errs.append("Identifiants d'exigences dupliqués.")
    if not spec.requirements:
        errs.append("Aucune exigence extraite.")

    cited: set[str] = set()
    for r in spec.requirements:
        errs += _check_citations(r.id, r.citations, src)
        cited |= {c.source_id for c in r.citations}
        origins = {src[c.source_id].origin for c in r.citations if c.source_id in src}
        if r.kind == "functional" and origins == {"constitution"}:
            errs.append(f"{r.id}: une règle issue uniquement de la constitution ne peut pas être 'functional' "
                        "(utilise 'constraint' ou 'non_functional').")
        quotes = [src[c.source_id].text for c in r.citations if c.source_id in src]
        ratio, unknown = grounding_ratio(r.statement, quotes)
        if ratio < min_grounding:
            errs.append(
                f"{r.id}: l'énoncé contient des termes absents des sources citées {sorted(unknown)} "
                f"(ancrage {ratio:.0%} < {min_grounding:.0%}). N'ajoute aucune information."
            )
        for i, ac in enumerate(r.acceptance_criteria, 1):
            errs += _check_citations(f"{r.id}/CA{i}", ac.citations, src)
            ac_src = [src[c.source_id].text for c in ac.citations if c.source_id in src]
            ratio, unknown = grounding_ratio(ac.text, ac_src)
            if ratio < min_grounding:
                errs.append(f"{r.id}/CA{i}: termes non sourcés {sorted(unknown)}.")
        cited |= {c.source_id for ac in r.acceptance_criteria for c in ac.citations}

    # Chaque réponse humaine (A-xxx) doit être reprise par une exigence : sinon elle est perdue.
    for s in sources:
        if s.origin == "answer" and s.id not in cited:
            errs.append(f"La réponse humaine {s.id} (\"{s.text}\") n'est reprise par aucune exigence : "
                        "cite-la dans l'exigence qu'elle précise (ou dans une contrainte).")

    for q in spec.open_questions:
        for sid in q.source_ids:
            if sid not in src:
                errs.append(f"{q.id}: source inconnue '{sid}'.")
        cited |= set(q.source_ids)

    # Couverture : chaque phrase du brief doit être exploitée (exigence ou question).
    for s in sources:
        if s.origin == "brief" and s.id not in cited:
            errs.append(f"La source {s.id} (\"{s.text}\") n'est couverte par aucune exigence ni question.")
    return errs


# --------------------------------------------------------------------------- #
def validate_plan(plan: Plan, spec: Spec, sources: list[SourceStatement]) -> list[str]:
    src = {s.id: s for s in sources}
    req_ids = {r.id for r in spec.requirements}
    errs: list[str] = []
    covered: set[str] = set()
    for it in plan.items:
        for rid in it.requirement_ids:
            if rid not in req_ids:
                errs.append(f"{it.id}: exigence inconnue '{rid}'.")
        covered |= set(it.requirement_ids)
    for rid in sorted(req_ids - covered):
        errs.append(f"L'exigence {rid} n'est couverte par aucun élément du plan.")

    # Aucun choix techno inventé : le nom doit apparaître dans une source citée.
    for tc in plan.tech_choices:
        texts = [src[s].text for s in tc.source_ids if s in src]
        unknown = [s for s in tc.source_ids if s not in src]
        if unknown:
            errs.append(f"Choix techno '{tc.name}': sources inconnues {unknown}.")
        if not any(_norm(tc.name) in _norm(t) for t in texts):
            errs.append(
                f"Choix techno '{tc.name}' non mentionné dans les sources citées. "
                "Si la techno n'est pas imposée, pose une question ouverte au lieu de choisir."
            )
    return errs


# --------------------------------------------------------------------------- #
def validate_tasks(tasks: TaskList, plan: Plan, spec: Spec, require_tests: bool = False) -> list[str]:
    req = {r.id: r for r in spec.requirements}
    plan_ids = {p.id for p in plan.items}
    task_ids = [t.id for t in tasks.tasks]
    errs: list[str] = []
    if len(task_ids) != len(set(task_ids)):
        errs.append("Identifiants de tâches dupliqués.")
    covered: set[str] = set()
    for t in tasks.tasks:
        for rid in t.requirement_ids:
            if rid not in req:
                errs.append(f"{t.id}: exigence inconnue '{rid}'.")
        for pid in t.plan_item_ids:
            if pid not in plan_ids:
                errs.append(f"{t.id}: élément de plan inconnu '{pid}'.")
        for d in t.depends_on:
            if d not in task_ids:
                errs.append(f"{t.id}: dépendance inconnue '{d}'.")
        for f in t.files:
            if f.startswith("/") or ".." in f.split("/"):
                errs.append(f"{t.id}: chemin interdit '{f}' (doit être relatif, sans '..').")
        allowed_ac = {_norm(ac.text) for rid in t.requirement_ids if rid in req for ac in req[rid].acceptance_criteria}
        for ac in t.acceptance_criteria:
            if _norm(ac) not in allowed_ac:
                errs.append(f"{t.id}: critère d'acceptation \"{ac}\" absent des exigences référencées.")
        covered |= set(t.requirement_ids)
    for rid in sorted(set(req) - covered):
        errs.append(f"L'exigence {rid} n'est couverte par aucune tâche.")
    if require_tests:  # option explicite : chaque exigence fonctionnelle a une tâche qui écrit un test
        for r in spec.requirements:
            if r.kind == "functional" and not any(
                r.id in t.requirement_ids and any(is_test_file(f) and f.rsplit("/", 1)[-1].startswith("test_") for f in t.files)
                for t in tasks.tasks
            ):
                errs.append(f"L'exigence fonctionnelle {r.id} n'a aucune tâche qui écrit un fichier de test (test_*.py).")
    if _has_cycle({t.id: t.depends_on for t in tasks.tasks}):
        errs.append("Les dépendances entre tâches forment un cycle.")
    return errs


def _has_cycle(graph: dict[str, list[str]]) -> bool:
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        if any(visit(m) for m in graph.get(n, []) if m in graph):
            return True
        state[n] = 2
        return False

    return any(visit(n) for n in graph)


def topological_order(tasks: TaskList) -> list[str]:
    """Ordre stable (tri par id à égalité) => exécution reproductible."""
    deps = {t.id: set(t.depends_on) for t in tasks.tasks}
    order: list[str] = []
    while deps:
        ready = sorted(t for t, d in deps.items() if not d - set(order))
        if not ready:
            raise ValueError("cycle")
        order.append(ready[0])
        del deps[ready[0]]
    return order


# --------------------------------------------------------------------------- #
def syntax_errors(impl: Implementation) -> list[str]:
    """Contrôle syntaxique déterministe des fichiers Python/JSON/TOML produits."""
    import ast
    import json
    import tomllib

    errs = []
    for fc in impl.files:
        try:
            if fc.path.endswith(".py"):
                ast.parse(fc.content, filename=fc.path)
            elif fc.path.endswith(".json") and fc.content.strip():
                json.loads(fc.content)
            elif fc.path.endswith(".toml"):
                tomllib.loads(fc.content)
        except (SyntaxError, ValueError) as exc:
            hint = " (as-tu échappé les retours à la ligne en \\n ? écris de vrais sauts de ligne)" if "\\n" in fc.content else ""
            errs.append(f"{fc.path} : contenu invalide : {exc}{hint}")
    return errs


def validate_implementation(impl: Implementation, tasks: TaskList, task_id: str) -> list[str]:
    task = next(t for t in tasks.tasks if t.id == task_id)
    errs: list[str] = syntax_errors(impl)
    if impl.task_id != task_id:
        errs.append(f"task_id attendu {task_id}, reçu {impl.task_id}.")
    for fc in impl.files:
        if fc.path not in task.files:
            errs.append(f"Fichier '{fc.path}' hors du périmètre de {task_id} (autorisés : {task.files}).")
    return errs


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") or "/tests/" in path or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def fixable_files(tasks: TaskList) -> list[str]:
    """Fichiers qu'une correction automatique peut toucher : ceux déclarés par les tâches, hors tests."""
    return sorted({f for t in tasks.tasks for f in t.files if not is_test_file(f)})


def validate_fix(impl: Implementation, tasks: TaskList) -> list[str]:
    allowed = set(fixable_files(tasks))
    errs = syntax_errors(impl)
    for fc in impl.files:
        if is_test_file(fc.path):
            errs.append(f"Interdit de modifier le test '{fc.path}' : les tests sont l'oracle.")
        elif fc.path not in allowed:
            errs.append(f"Fichier '{fc.path}' non déclaré par une tâche (autorisés : {sorted(allowed)}).")
    return errs


def validate_clarify(c, sources: list[SourceStatement]) -> list[str]:
    src = {s.id for s in sources}
    errs = [f"{q.id}: source inconnue '{sid}'." for q in c.questions for sid in q.source_ids if sid not in src]
    ids = [q.id for q in c.questions]
    if len(ids) != len(set(ids)):
        errs.append("Identifiants de questions dupliqués.")
    return errs


def analyze(spec: Spec, plan: Plan, tasks: TaskList, sources: list[SourceStatement], require_tests: bool = False) -> dict:
    """Étape analyze : contrôle croisé 100 % déterministe des trois artefacts."""
    errors = validate_spec(spec, sources) + validate_plan(plan, spec, sources) + validate_tasks(tasks, plan, spec, require_tests)
    used_plan = {p for t in tasks.tasks for p in t.plan_item_ids}
    warnings = [f"Élément de plan {p.id} sans aucune tâche." for p in plan.items if p.id not in used_plan]
    files: dict[str, list[str]] = {}
    for t in tasks.tasks:
        for f in t.files:
            files.setdefault(f, []).append(t.id)
    order = {tid: i for i, tid in enumerate(topological_order(tasks))} if not any("cycle" in e for e in errors) else {}
    deps = {t.id: set(t.depends_on) for t in tasks.tasks}
    for f, tids in files.items():
        if len(tids) > 1:
            warnings.append(f"Le fichier {f} est modifié par plusieurs tâches ({', '.join(tids)}) : elles s'exécutent dans l'ordre {sorted(tids, key=lambda x: order.get(x, 0))}.")
    unanswered = [q.id for q in spec.open_questions]
    if unanswered:
        warnings.append(f"Questions ouvertes non résolues dans la spec : {', '.join(unanswered)}.")
    return {"errors": errors, "warnings": warnings}


# --------------------------------------------------------------------------- #
def repair_spec(spec: Spec, sources: list[SourceStatement], min_grounding: float = 0.7) -> tuple[Spec, list[str]]:
    """Réparation DÉTERMINISTE d'une spec que le modèle n'a pas su corriger (rien n'est inventé) :

    1. un énoncé mal ancré reçoit la citation des sources qui contiennent ses termes manquants
       (souvent la réponse humaine A-xxx qu'il reprend sans la citer) ;
    2. une réponse humaine jamais citée devient une contrainte dont l'énoncé est la source mot pour mot ;
    3. une phrase du brief non couverte devient une exigence dont l'énoncé est la phrase mot pour mot.

    Renvoie la spec réparée et la liste des réparations (tracées dans l'audit)."""
    from .models import Citation, Requirement

    src = {s.id: s for s in sources}
    data = spec.model_copy(deep=True)
    notes: list[str] = []

    def anchor(where: str, text: str, citations: list) -> None:
        for _ in range(4):
            quotes = [src[c.source_id].text for c in citations if c.source_id in src]
            ratio, unknown = grounding_ratio(text, quotes)
            if ratio >= min_grounding or not unknown:
                return
            have = {c.source_id for c in citations}
            best, gain = None, 0
            for s in sources:
                if s.id in have:
                    continue
                g = len(unknown - grounding_ratio(" ".join(unknown), [s.text])[1])
                if g > gain:
                    best, gain = s, g
            if best is None:
                return
            citations.append(Citation(source_id=best.id, quote=best.text))
            notes.append(f"{where} : citation de {best.id} ajoutée (elle contient les termes {sorted(unknown)[:6]})")

    for r in data.requirements:
        anchor(r.id, r.statement, r.citations)
        for i, ac in enumerate(r.acceptance_criteria, 1):
            anchor(f"{r.id}/CA{i}", ac.text, ac.citations)

    def cited() -> set[str]:
        ids = {c.source_id for r in data.requirements for c in r.citations}
        ids |= {c.source_id for r in data.requirements for ac in r.acceptance_criteria for c in ac.citations}
        return ids

    def next_id() -> str:
        nums = [int(m.group(1)) for r in data.requirements if (m := re.match(r"REQ-(\d+)$", r.id))]
        return f"REQ-{max(nums, default=0) + 1:03d}"

    for s in sources:
        if s.origin == "answer" and s.id not in cited():
            rid = next_id()
            data.requirements.append(Requirement(id=rid, kind="constraint", statement=s.text,
                                                 citations=[Citation(source_id=s.id, quote=s.text)]))
            notes.append(f"{rid} : contrainte ajoutée, reprenant mot pour mot la réponse humaine {s.id}")
    questioned = {sid for q in data.open_questions for sid in q.source_ids}
    for s in sources:
        if s.origin == "brief" and s.id not in cited() and s.id not in questioned:
            rid = next_id()
            kind = "functional" if re.search(r"\bpeu(t|vent)\b", s.text, re.I) else "constraint"
            data.requirements.append(Requirement(id=rid, kind=kind, statement=s.text,
                                                 citations=[Citation(source_id=s.id, quote=s.text)]))
            notes.append(f"{rid} : exigence ajoutée, reprenant mot pour mot la phrase du brief {s.id}")
    return data, notes
