"""Le Correcteur : troisième agent de l'AI factory.

Producteur (étapes SDD) → Gardien (vérifie) → Correcteur (corrige ce que le Gardien refuse) → Gardien (revérifie).

- Il ne reçoit que les alertes corrigeables, l'artefact fautif et les sources.
- Il propose des PATCHS ciblés (`Correction`), jamais une réécriture complète.
- Chaque patch est appliqué puis revalidé par les mêmes validateurs déterministes que le Gardien : il n'est
  gardé que s'il fait baisser le nombre d'erreurs. Le modèle ne peut donc rien valider lui-même, ni rien
  inventer (une citation doit être copiée mot pour mot d'une source, un énoncé doit être ancré, etc.).
- Le nombre de tours est borné ; au-delà, c'est la pause humaine habituelle.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable

from .models import Correction, PatchOp

# alertes qu'un correcteur peut traiter ; les questions ouvertes restent à l'humain
CORRECTABLE = {"SPEC_INVALIDE": "spec", "REPONSE_IGNOREE": "spec", "PLAN_INVALIDE": "plan",
               "TACHES_INVALIDES": "tasks"}


def correctable(alerts: list[dict]) -> dict[str, list[dict]]:
    """Alertes bloquantes corrigeables, groupées par artefact (spec, plan, tasks)."""
    out: dict[str, list[dict]] = {}
    for a in alerts:
        if a["severity"] == "blocking" and a["code"] in CORRECTABLE:
            out.setdefault(CORRECTABLE[a["code"]], []).append(a)
    return out


def _find(container: Any, key: str):
    if isinstance(container, dict):
        if key not in container:
            raise KeyError(f"champ inconnu « {key} »")
        return container[key]
    if isinstance(container, list):
        for el in container:
            if isinstance(el, dict) and key in (el.get("id"), el.get("name")):
                return el
        if key.isdigit() and int(key) < len(container):
            return container[int(key)]
        raise KeyError(f"élément « {key} » introuvable")
    raise KeyError(f"impossible de descendre dans « {key} »")


def apply_patch(doc: dict, op: PatchOp) -> dict:
    """Applique un patch sur une COPIE du document. Lève ValueError si le chemin ou la valeur est invalide."""
    new = copy.deepcopy(doc)
    parts = [p for p in op.path.strip("/").split("/") if p]
    if not parts:
        raise ValueError("chemin vide")
    try:
        value = json.loads(op.value) if op.value.strip() else None
    except json.JSONDecodeError as exc:
        raise ValueError(f"valeur JSON invalide : {exc}") from None
    try:
        parent = new
        for p in parts[:-1]:
            parent = _find(parent, p)
        last = parts[-1]
        if op.op == "append":
            target = _find(parent, last)
            if not isinstance(target, list):
                raise ValueError(f"« {op.path} » n'est pas une liste")
            target.append(value)
        elif op.op == "set":
            if isinstance(parent, dict):
                parent[last] = value
            else:
                parent[parent.index(_find(parent, last))] = value
        else:  # remove
            if isinstance(parent, dict):
                del parent[last]
            else:
                parent.remove(_find(parent, last))
    except KeyError as exc:
        raise ValueError(str(exc)) from None
    return new


def apply_corrections(doc: dict, ops: list[PatchOp], model_cls, validate: Callable[[Any], list[str]]):
    """Applique les patchs un par un ; chacun n'est gardé que s'il fait baisser le nombre d'erreurs.

    Renvoie (document corrigé, erreurs restantes, journal [(patch, gardé ?, motif)])."""
    errors = validate(model_cls.model_validate(doc))
    log: list[tuple[PatchOp, bool, str]] = []
    for op in ops:
        try:
            cand = apply_patch(doc, op)
            cand_errors = validate(model_cls.model_validate(cand))
        except Exception as exc:  # chemin inconnu, JSON invalide, schéma non respecté
            log.append((op, False, f"rejeté : {str(exc)[:160]}"))
            continue
        if len(cand_errors) < len(errors):
            doc, errors = cand, cand_errors
            log.append((op, True, f"gardé ({len(errors)} erreur(s) restante(s))"))
        else:
            log.append((op, False, "rejeté : ne réduit pas les erreurs"))
    return doc, errors, log


def correction_payload(kind: str, alerts: list[dict], doc: dict, sources_txt: str, context: str) -> str:
    lines = "\n".join(f"- [{a['code']}] {a['message']}" for a in alerts)
    return (f"ALERTES DU GARDIEN SUR L'ARTEFACT « {kind} » :\n{lines}\n\nARTEFACT ({kind}) :\n"
            f"{json.dumps(doc, ensure_ascii=False, separators=(",", ":"))}\n\nSOURCES :\n{sources_txt}{context}")


__all__ = ["CORRECTABLE", "Correction", "apply_corrections", "apply_patch", "correctable", "correction_payload"]
