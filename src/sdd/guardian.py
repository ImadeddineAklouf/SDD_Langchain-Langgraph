"""Le Gardien : vérificateur INDÉPENDANT de la cohérence SDD (architecture « AI factory »).

D'un côté, le graphe SDD produit (spec, plan, tâches, code). De l'autre, ce graphe-ci, compilé
séparément, contrôle chaque artefact juste après sa production et lève des alertes :

- ``blocking`` : pause obligatoire ; l'humain répond, corrige, accepte le risque (avec justification) ou arrête ;
- ``warning``  : pause ; l'humain acquitte (ou corrige) ;
- ``info``     : tracé dans le journal, sans pause.

1. Contrôles déterministes (code pur, reproductibles, sans LLM) : ils forment l'essentiel du Gardien.
2. Contrôle sémantique (LLM, facultatif, idéalement un autre modèle que le producteur) : il ne peut
   qu'AJOUTER des alertes, jamais en retirer ni valider ; un constat qui cite un id inexistant est rejeté.

Le Gardien ne fait pas confiance aux erreurs rapportées par le producteur : il revalide tout à partir
des artefacts eux-mêmes. Chaque alerte a une empreinte stable (``id``) : un risque accepté par
l'humain n'est plus jamais bloquant, mais reste visible (rétrogradé en ``info``).
"""
from __future__ import annotations

import ast
import hashlib
import json
import operator
import re
import sys
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .llm import generate
from .models import Alert, Implementation, Plan, SemanticReview, SourceStatement, Spec, TaskList
from .validators import _norm, validate_implementation, validate_plan, validate_spec, validate_tasks

STAGES = ["spec", "plan", "analyze", "impl", "verify"]
# Ordre des nœuds producteurs : à la reprise, on relance le plus en amont des nœuds concernés.
REDO_ORDER = ["specify", "plan", "tasks", "implement", "fix"]
DEFAULT_REDO = {"spec": "specify", "plan": "plan", "analyze": "tasks", "impl": "implement", "verify": "fix"}
SEMANTIC_STAGES = {"spec", "plan", "analyze"}
SEVERITY_RANK = {"blocking": 0, "warning": 1, "info": 2}

# Fichiers de données : leur nom et leur format sont des choix fonctionnels, ils doivent venir d'une source.
FORMAT_WORDS = {
    "json": ["json"], "csv": ["csv"], "tsv": ["tsv"], "db": ["sqlite", "base de données", "database"],
    "sqlite": ["sqlite"], "sqlite3": ["sqlite"], "yaml": ["yaml"], "yml": ["yaml", "yml"], "xml": ["xml"],
    "txt": ["texte", "txt", "text"], "ini": ["ini"], "cfg": ["cfg"], "toml": ["toml"],
    "parquet": ["parquet"], "xlsx": ["excel", "xlsx"],
}
_DATAFILE = re.compile(r"^[\w./-]*\w\.(" + "|".join(FORMAT_WORDS) + r")$", re.I)
_IDS = re.compile(r"\b(?:[BCAF]-\d{3}|REQ-\d{3}|PLN-\d{3}|T-\d{3}|QP?-\d{3})\b")


# --------------------------------------------------------------------------- #
# Fabrique d'alertes
# --------------------------------------------------------------------------- #
def make_alert(code: str, severity: str, stage: str, message: str, refs=(), redo: str | None = None,
               key: str | None = None, origin: str = "déterministe", question: dict | None = None) -> dict:
    """Crée une alerte avec une empreinte stable : même problème => même id (pour l'acceptation du risque)."""
    fp = hashlib.sha1(f"{code}|{_norm(key or message)}".encode()).hexdigest()[:8]
    return Alert(
        id=f"AL-{fp}", code=code, severity=severity, stage=stage, message=message,
        refs=sorted(set(refs)), redo=redo or DEFAULT_REDO[stage], origin=origin, question=question,
    ).model_dump()


def _from_errors(errors: list[str], stage: str, code: str, redo: str) -> list[dict]:
    out = []
    for e in errors:
        c = "REPONSE_IGNOREE" if "réponse humaine" in e else code
        out.append(make_alert(c, "blocking", stage, e, _IDS.findall(e), redo))
    return out


def _sources(a: dict) -> list[SourceStatement]:
    return [SourceStatement.model_validate(s) for s in a.get("sources", [])]


def _source_text(a: dict) -> str:
    return " ".join(_norm(s["text"]) for s in a.get("sources", []))


# --------------------------------------------------------------------------- #
# Contrôles déterministes réutilisables
# --------------------------------------------------------------------------- #
def open_question_alerts(stage: str, questions: list[dict], redo: str) -> list[dict]:
    """Toute question ouverte est BLOQUANTE : sans réponse, l'IA devrait deviner."""
    return [
        make_alert(
            "QUESTION_OUVERTE", "blocking", stage,
            f"{q['id']} sans réponse : « {q['question']} ». Continuer obligerait l'IA à deviner.",
            [q["id"], *q.get("source_ids", [])], redo, key=q["question"], question=q,
        )
        for q in questions
    ]


def all_open_questions(a: dict) -> list[tuple[dict, str]]:
    """Questions encore ouvertes (spec/clarify puis plan), avec le nœud qui doit les intégrer."""
    qs = [(q, "specify") for q in a.get("clarifications") or []]
    qs += [(q, "plan") for q in (a.get("plan") or {}).get("open_questions", [])]
    return qs


def _all_open_question_alerts(stage: str, a: dict) -> list[dict]:
    out = []
    for q, redo in all_open_questions(a):
        out += open_question_alerts(stage, [q], redo)
    return out


def data_file_alerts(stage: str, names: dict[str, list[str]], a: dict, allowed: set[str], redo: str) -> list[dict]:
    """Un fichier de données (x.json, x.csv…) doit être nommé, ou au moins son format, dans une source."""
    text = _source_text(a)
    out = []
    for name, refs in sorted(names.items()):
        base = name.rsplit("/", 1)[-1].lower()
        if name in allowed or base in allowed or base in text:
            continue
        ext = base.rsplit(".", 1)[-1]
        if any(w in text for w in FORMAT_WORDS.get(ext, [ext])):
            out.append(make_alert(
                "NOM_FICHIER_CHOISI", "warning", stage,
                f"Le format .{ext} est demandé par une source, mais le nom « {name} » est un choix de l'IA : à valider.",
                refs, redo, key=f"{name}"))
        else:
            out.append(make_alert(
                "FICHIER_NON_SOURCE", "blocking", stage,
                f"Le fichier de données « {name} » (format .{ext}) n'est demandé par aucune source : invention.",
                refs, redo, key=f"{name}"))
    return out


def _imports(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {n.name.split(".")[0] for n in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def _string_literals(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {n.value.strip() for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _local_modules(paths) -> set[str]:
    mods = set()
    for p in paths:
        for part in p.split("/"):
            mods.add(part.removesuffix(".py"))
    return mods


# --------------------------------------------------------------------------- #
# Contrôles par étape (entrée : instantané des artefacts du graphe SDD)
# --------------------------------------------------------------------------- #
def check_spec(a: dict, require_tests: bool = False) -> list[dict]:
    spec = Spec.model_validate(a["spec"])
    alerts = _from_errors(validate_spec(spec, _sources(a)), "spec", "SPEC_INVALIDE", "specify")
    return alerts + open_question_alerts("spec", a.get("clarifications") or [], "specify")


def check_plan(a: dict, require_tests: bool = False) -> list[dict]:
    spec, plan = Spec.model_validate(a["spec"]), Plan.model_validate(a["plan"])
    alerts = _from_errors(validate_plan(plan, spec, _sources(a)), "plan", "PLAN_INVALIDE", "plan")
    return alerts + _all_open_question_alerts("plan", a)


def check_analyze(a: dict, require_tests: bool = False) -> list[dict]:
    """Contrôle croisé spec ↔ plan ↔ tâches, chaque erreur renvoyée au nœud qui l'a produite."""
    sources = _sources(a)
    spec, plan, tl = Spec.model_validate(a["spec"]), Plan.model_validate(a["plan"]), TaskList.model_validate(a["tasks"])
    alerts = _from_errors(validate_spec(spec, sources), "analyze", "SPEC_INVALIDE", "specify")
    alerts += _from_errors(validate_plan(plan, spec, sources), "analyze", "PLAN_INVALIDE", "plan")
    alerts += _from_errors(validate_tasks(tl, plan, spec, require_tests), "analyze", "TACHES_INVALIDES", "tasks")
    used = {p for t in tl.tasks for p in t.plan_item_ids}
    for p in plan.items:
        if p.id not in used:
            alerts.append(make_alert("PLAN_SANS_TACHE", "warning", "analyze",
                                     f"L'élément de plan {p.id} ({p.title}) n'est réalisé par aucune tâche.", [p.id], "tasks"))
    files: dict[str, list[str]] = {}
    for t in tl.tasks:
        for f in t.files:
            files.setdefault(f, []).append(t.id)
    for f, tids in sorted(files.items()):
        if len(tids) > 1:
            alerts.append(make_alert("FICHIER_PARTAGE", "info", "analyze",
                                     f"Le fichier {f} est modifié par plusieurs tâches ({', '.join(tids)}).", tids, "tasks"))
    data = {f: tids for f, tids in files.items() if _DATAFILE.match(f)}
    alerts += data_file_alerts("analyze", data, a, set(), "tasks")
    # Règle centrale de l'AI factory : pas d'implémentation tant qu'une question reste ouverte.
    return alerts + _all_open_question_alerts("analyze", a)


def check_impl(a: dict, require_tests: bool = False) -> list[dict]:
    tl = TaskList.model_validate(a["tasks"])
    impl = Implementation.model_validate(a["impl"])
    tid = impl.task_id
    alerts = _from_errors(validate_implementation(impl, tl, a["task_id"]), "impl", "IMPL_INVALIDE", "implement")
    text = _source_text(a)
    tech = " ".join(_norm(t["name"]) for t in (a.get("plan") or {}).get("tech_choices", []))
    declared = {f for t in tl.tasks for f in t.files}
    local = _local_modules(declared | set(a.get("written") or []))
    stdlib_used: set[str] = set()
    literals: dict[str, list[str]] = {}
    for fc in impl.files:
        if not fc.path.endswith(".py"):
            continue
        for mod in sorted(_imports(fc.content)):
            if mod in local or mod == "__future__":
                continue
            if mod in sys.stdlib_module_names:
                stdlib_used.add(mod)
            elif not re.search(rf"\b{re.escape(mod.lower())}\b", text + " " + tech):
                alerts.append(make_alert(
                    "DEPENDANCE_NON_SOURCEE", "blocking", "impl",
                    f"{fc.path} importe « {mod} », une bibliothèque externe qu'aucune source n'autorise.",
                    [tid], "implement", key=f"{fc.path}|{mod}"))
        for lit in _string_literals(fc.content):
            if _DATAFILE.match(lit):
                literals.setdefault(lit, []).append(tid)
    allowed = declared | {f.rsplit("/", 1)[-1].lower() for f in declared}
    alerts += data_file_alerts("impl", literals, a, allowed, "implement")
    if stdlib_used:
        alerts.append(make_alert("BIBLIOTHEQUE_STANDARD", "info", "impl",
                                 f"{tid} utilise la bibliothèque standard : {', '.join(sorted(stdlib_used))}.", [tid], "implement"))
    return alerts + _all_open_question_alerts("impl", a)


def check_verify(a: dict, require_tests: bool = False) -> list[dict]:
    v = a.get("verify") or {}
    alerts = []
    if not v.get("passed", False):
        tail = "\n".join((v.get("output") or "").strip().splitlines()[-6:])
        alerts.append(make_alert("TESTS_EN_ECHEC", "blocking", "verify",
                                 f"Les tests échouent malgré les corrections automatiques :\n{tail}", [], "fix",
                                 key="tests"))
    elif require_tests and not v.get("ran", False):
        alerts.append(make_alert("AUCUN_TEST", "warning", "verify",
                                 "La constitution exige des tests, mais aucun test n'a été exécuté.", [], "tasks"))
    return alerts


CHECKS: dict[str, Callable[[dict, bool], list[dict]]] = {
    "spec": check_spec, "plan": check_plan, "analyze": check_analyze, "impl": check_impl, "verify": check_verify,
}


# --------------------------------------------------------------------------- #
# Contrôle sémantique (LLM) : il ne peut qu'ajouter des alertes
# --------------------------------------------------------------------------- #
def _known_ids(a: dict) -> set[str]:
    ids = {s["id"] for s in a.get("sources", [])}
    ids |= {r["id"] for r in (a.get("spec") or {}).get("requirements", [])}
    ids |= {p["id"] for p in (a.get("plan") or {}).get("items", [])}
    ids |= {t["id"] for t in (a.get("tasks") or {}).get("tasks", [])}
    return ids


def _redo_for_refs(refs: list[str], stage: str) -> str:
    """Le constat est renvoyé au nœud le plus en amont parmi les artefacts cités."""
    kinds = []
    for r in refs:
        if r.startswith("T-"):
            kinds.append("tasks")
        elif r.startswith("PLN-"):
            kinds.append("plan")
        else:
            kinds.append("specify")
    return min(kinds, key=REDO_ORDER.index) if kinds else DEFAULT_REDO[stage]


def semantic_alerts(llm, stage: str, a: dict, max_retries: int = 1, fallback_llm=None) -> list[dict]:
    known = _known_ids(a)
    parts = ["SOURCES :\n" + "\n".join(f"[{s['id']}] ({s['origin']}) {s['text']}" for s in a.get("sources", []))]
    parts.append("SPEC :\n" + json.dumps(a["spec"]["requirements"], ensure_ascii=False, separators=(",", ":")))
    if stage in ("plan", "analyze"):
        parts.append("PLAN :\n" + json.dumps(a["plan"]["items"], ensure_ascii=False, separators=(",", ":")))
    if stage == "analyze":
        parts.append("TÂCHES :\n" + json.dumps(
            [{k: t[k] for k in ("id", "title", "description", "requirement_ids", "plan_item_ids", "files")}
             for t in a["tasks"]["tasks"]], ensure_ascii=False, separators=(",", ":")))

    def validate(r: SemanticReview) -> list[str]:
        return [f"Constat {i}: id inexistant '{x}' (ids valides uniquement)."
                for i, f in enumerate(r.findings, 1) for x in f.refs if x not in known]

    try:
        review, _errs = generate(llm, SemanticReview, prompts.GUARDIAN, "\n\n".join(parts), validate, max_retries,
                                 fallback_llm=fallback_llm)
    except Exception as exc:  # quota, réseau… : le Gardien le signale au lieu de se taire
        return [make_alert("GARDIEN_SEMANTIQUE_INDISPONIBLE", "warning", stage,
                           f"Le contrôle sémantique n'a pas pu s'exécuter ({type(exc).__name__}: {str(exc)[:200]}). "
                           "Seuls les contrôles déterministes ont été faits.", [], key=f"indispo|{stage}")]
    out = []
    for f in review.findings:
        if not f.refs or any(x not in known for x in f.refs):
            continue  # un constat qui cite un id inventé est écarté : le Gardien n'invente pas non plus
        out.append(make_alert(f"SEMANTIQUE_{f.kind.upper()}", f.severity, stage, f.message, f.refs,
                              _redo_for_refs(f.refs, stage), origin="sémantique"))
    return out


# --------------------------------------------------------------------------- #
# Le graphe du Gardien (compilé séparément du graphe SDD)
# --------------------------------------------------------------------------- #
class GuardianState(TypedDict, total=False):
    stage: str
    artifacts: dict[str, Any]
    accepted: list[str]
    alerts: Annotated[list[dict], operator.add]  # réducteur : contrôles déterministes + sémantiques
    report: list[dict]


def consolidate_alerts(alerts: list[dict], accepted: list[str]) -> list[dict]:
    """Dédoublonne, rétrograde en info les risques déjà acceptés, trie par gravité."""
    seen, out = set(), []
    for al in alerts:
        if al["id"] in seen:
            continue
        seen.add(al["id"])
        if al["id"] in accepted and al["severity"] != "info":
            al = {**al, "severity": "info", "accepted": True}
        out.append(al)
    return sorted(out, key=lambda x: (SEVERITY_RANK[x["severity"]], x["code"],
                                      (x.get("question") or {}).get("id", ""), x["id"]))


def build_guardian(llm=None, max_retries: int = 1, require_tests: bool = False, fallback_llm=None):
    """Graphe du Gardien : deterministic -> (semantic) -> consolidate.

    Le contrôle sémantique n'est lancé que si un LLM est fourni, à une étape où il a du sens, et
    seulement si les contrôles déterministes n'ont rien trouvé de bloquant (inutile de payer un appel
    LLM pour un artefact déjà rejeté, et cela ménage les quotas gratuits).
    """
    def deterministic(s: GuardianState):
        return {"alerts": CHECKS[s["stage"]](s["artifacts"], require_tests)}

    def semantic(s: GuardianState):
        # même modèle de secours que le producteur : un quota épuisé ne fait plus sauter le contrôle sémantique
        return {"alerts": semantic_alerts(llm, s["stage"], s["artifacts"], max_retries, fallback_llm)}

    def consolidate(s: GuardianState):
        return {"report": consolidate_alerts(s.get("alerts", []), s.get("accepted", []))}

    def route(s: GuardianState):
        blocking = any(a["severity"] == "blocking" and a["id"] not in s.get("accepted", []) for a in s.get("alerts", []))
        if llm is not None and s["stage"] in SEMANTIC_STAGES and not blocking:
            return "semantic"
        return "consolidate"

    g = StateGraph(GuardianState)
    g.add_node("deterministic", deterministic)
    g.add_node("semantic", semantic)
    g.add_node("consolidate", consolidate)
    g.add_edge(START, "deterministic")
    g.add_conditional_edges("deterministic", route, ["semantic", "consolidate"])
    g.add_edge("semantic", "consolidate")
    g.add_edge("consolidate", END)
    return g.compile(name="gardien")


# --------------------------------------------------------------------------- #
# Décision humaine : validation déterministe
# --------------------------------------------------------------------------- #
def check_decision(action: str, answers: dict, feedback: str, justification: str, alerts: list[dict]) -> str | None:
    """Renvoie un message d'erreur si la décision ne traite pas correctement les alertes, sinon None."""
    active = [a for a in alerts if a["severity"] != "info"]
    blocking = [a for a in active if a["severity"] == "blocking"]
    qids = {a["question"]["id"] for a in active if a.get("question")}
    if action == "continue" and blocking:
        return ("Impossible de continuer : des alertes bloquantes doivent être traitées (répondre, corriger, "
                "accepter le risque avec une justification, ou arrêter).")
    if action == "accept_risk" and not justification.strip():
        return "Accepter un risque exige une justification écrite (elle est tracée dans le journal d'audit)."
    if action == "answer":
        if not answers:
            return "Aucune réponse saisie."
        unknown = sorted(set(answers) - qids)
        if unknown:
            return f"Réponse à une question inconnue : {', '.join(unknown)}."
    if action == "correct" and not feedback.strip():
        return "Décrivez la correction attendue."
    return None


def redo_node(alerts: list[dict], stage: str, only_questions: bool = False) -> str:
    """Nœud de reprise : le plus en amont parmi les alertes traitées."""
    targets = [a["redo"] for a in alerts if a["severity"] != "info" and (a.get("question") or not only_questions)]
    if not targets:
        return DEFAULT_REDO[stage]
    return min(targets, key=REDO_ORDER.index)
