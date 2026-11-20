"""Schémas Pydantic stricts pour chaque artefact SDD.

Tous les modèles interdisent les champs supplémentaires (extra="forbid") :
le LLM ne peut pas glisser d'information hors schéma.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Sources : seule « vérité » autorisée. Produites sans LLM.
# --------------------------------------------------------------------------- #
class SourceStatement(Strict):
    id: str = Field(description="Identifiant stable, ex. B-001 (brief), A-001 (réponse humaine)")
    text: str
    origin: Literal["brief", "constitution", "answer", "feedback"]


class Citation(Strict):
    source_id: str = Field(description="Id d'une SourceStatement existante")
    quote: str = Field(description="Extrait COPIÉ MOT POUR MOT depuis la source citée")


# --------------------------------------------------------------------------- #
# 1. Spécification
# --------------------------------------------------------------------------- #
class AcceptanceCriterion(Strict):
    text: str
    citations: list[Citation] = Field(min_length=1)


class Requirement(Strict):
    id: str = Field(description="REQ-001, REQ-002, ...")
    kind: Literal["functional", "non_functional", "constraint"]
    statement: str = Field(description="Reformulation minimale, sans ajout d'information")
    citations: list[Citation] = Field(min_length=1)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)


class OpenQuestion(Strict):
    id: str = Field(description="Q-001, Q-002, ...")
    question: str
    source_ids: list[str] = Field(default_factory=list, description="Sources ambiguës concernées")


class Spec(Strict):
    requirements: list[Requirement]
    open_questions: list[OpenQuestion] = Field(default_factory=list)


class Clarifications(Strict):
    """Sortie de l'étape clarify : uniquement des questions, jamais de réponses."""
    questions: list[OpenQuestion] = Field(default_factory=list, max_length=5)


# --------------------------------------------------------------------------- #
# 2. Plan technique
# --------------------------------------------------------------------------- #
class PlanItem(Strict):
    id: str = Field(description="PLN-001, ...")
    kind: Literal["component", "data_model", "interface", "decision"]
    title: str
    description: str
    requirement_ids: list[str] = Field(min_length=1)


class TechChoice(Strict):
    name: str = Field(description="Techno/lib/langage, doit apparaître dans une source citée")
    source_ids: list[str] = Field(min_length=1)


class Plan(Strict):
    items: list[PlanItem]
    tech_choices: list[TechChoice] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 3. Tâches
# --------------------------------------------------------------------------- #
class Task(Strict):
    id: str = Field(description="T-001, ...")
    title: str
    description: str
    requirement_ids: list[str] = Field(min_length=1)
    plan_item_ids: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    files: list[str] = Field(min_length=1, description="Chemins relatifs que la tâche a le droit de modifier")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Copie EXACTE de critères d'acceptation des exigences référencées",
    )


class TaskList(Strict):
    tasks: list[Task]


# --------------------------------------------------------------------------- #
# Le Correcteur : corrections ciblées (patchs), jamais une réécriture complète
# --------------------------------------------------------------------------- #
class PatchOp(Strict):
    """Une correction ciblée d'un artefact JSON (spec, plan ou tâches)."""
    op: Literal["set", "append", "remove"] = Field(
        description="set : remplacer la valeur au chemin ; append : ajouter à la liste au chemin ; "
                    "remove : supprimer l'élément au chemin")
    path: str = Field(description="Chemin : champs séparés par '/', éléments de liste désignés par leur id "
                                  "(ou name), ex. requirements/REQ-003/kind, requirements/REQ-003/citations, "
                                  "requirements, items/PLN-002/requirement_ids, tasks/T-004")
    value: str = Field(default="", description="Valeur en JSON (vide pour remove)")
    reason: str = Field(description="Alerte corrigée et source qui justifie la correction")


class Correction(Strict):
    ops: list[PatchOp] = Field(default_factory=list, max_length=25)


# --------------------------------------------------------------------------- #
# 4. Implémentation
# --------------------------------------------------------------------------- #
class FileChange(Strict):
    path: str
    content: str


class Implementation(Strict):
    task_id: str
    files: list[FileChange] = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Le Gardien : alertes, constats sémantiques et décision humaine
# --------------------------------------------------------------------------- #
Severity = Literal["blocking", "warning", "info"]


class Alert(Strict):
    """Alerte levée par le Gardien. `id` est une empreinte stable (même problème = même id)."""
    id: str
    code: str = Field(description="Type de contrôle, ex. QUESTION_OUVERTE, REPONSE_IGNOREE")
    severity: Severity
    stage: str = Field(description="Étape contrôlée : spec, plan, analyze, impl, verify")
    message: str
    refs: list[str] = Field(default_factory=list, description="Ids concernés : B-001, REQ-002, T-003...")
    redo: str = Field(description="Nœud SDD à relancer pour corriger")
    origin: Literal["déterministe", "sémantique"] = "déterministe"
    question: dict | None = Field(default=None, description="Question ouverte à laquelle l'humain peut répondre")
    accepted: bool = Field(default=False, description="Risque déjà accepté/acquitté par l'humain (rétrogradé en info)")


class Finding(Strict):
    """Constat du contrôle sémantique (LLM). Il ne peut que signaler, jamais valider."""
    kind: Literal["contradiction", "hors_perimetre", "deformation", "incoherence"]
    severity: Literal["blocking", "warning"]
    refs: list[str] = Field(min_length=1, description="Ids EXISTANTS concernés (B-001, A-001, REQ-001, PLN-001, T-001)")
    message: str = Field(description="Explication courte et factuelle, en français")


class SemanticReview(Strict):
    findings: list[Finding] = Field(default_factory=list, max_length=8)


class GateDecision(Strict):
    """Décision humaine à la porte du Gardien (charge utile de reprise après interrupt)."""
    action: Literal["continue", "answer", "correct", "accept_risk", "abort"]
    answers: dict[str, str] = Field(default_factory=dict, description="Réponses aux questions ouvertes: {Q-001: '...'}")
    feedback: str = Field(default="", description="Correction demandée (devient une source F-xxx)")
    justification: str = Field(default="", description="Obligatoire pour accepter un risque")
