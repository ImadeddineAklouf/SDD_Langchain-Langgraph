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
# 4. Implémentation
# --------------------------------------------------------------------------- #
class FileChange(Strict):
    path: str
    content: str


class Implementation(Strict):
    task_id: str
    files: list[FileChange] = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Revue humaine (payload de reprise après interrupt)
# --------------------------------------------------------------------------- #
class Review(Strict):
    action: Literal["approve", "revise", "abort"]
    answers: dict[str, str] = Field(default_factory=dict, description="Réponses aux questions ouvertes: {Q-001: '...'}")
    feedback: str = ""
