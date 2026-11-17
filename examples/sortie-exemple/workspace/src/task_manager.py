"""Composant principal qui gère les tâches, l'ajout, la liste, la mise à jour et la persistance.

Ce module expose la classe :

- :class:`TaskManager`
    Gestionnaire de tâches qui utilise le modèle :class:`~src.task.Task` et la classe
    :class:`~src.persistence.TaskPersistence` pour sauvegarder et charger les tâches
    depuis un fichier JSON.

Les fonctionnalités implémentées correspondent aux exigences REQ-005, REQ-006, REQ-007 et REQ-009.
"""

from __future__ import annotations

from typing import List

from src.task import Task
from src.persistence import TaskPersistence


class TaskManager:
    """Gestionnaire de tâches.

    Le gestionnaire charge les tâches existantes depuis un fichier JSON au
    moment de son initialisation. Toutes les modifications (ajout, mise à jour
    du statut) sont immédiatement sauvegardées.
    """

    def __init__(self, file_path: str = "tasks.json") -> None:
        """Crée un nouveau gestionnaire de tâches.

        Args:
            file_path: Chemin vers le fichier JSON de persistance.
        """
        self._file_path = file_path
        self._tasks: List[Task] = TaskPersistence.load_tasks(self._file_path)
        # Reset the id counter if no tasks are loaded to ensure deterministic
        # id assignment for fresh managers.
        if not self._tasks:
            Task._id_counter = 1

    # ---------------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------------
    def add_task(self, title: str) -> Task:
        """Ajoute une nouvelle tâche.

        L'utilisateur peut ajouter une tâche avec un titre.

        Args:
            title: Le titre de la tâche.

        Returns:
            La tâche nouvellement créée.
        """
        task = Task(title, "pending")
        self._tasks.append(task)
        TaskPersistence.save_tasks(self._tasks, self._file_path)
        return task

    def list_tasks(self) -> List[Task]:
        """Renvoie la liste des tâches.

        L'utilisateur peut lister les tâches.

        Returns:
            Une liste de :class:`~src.task.Task`.
        """
        return list(self._tasks)

    def mark_done(self, task_id: int) -> Task:
        """Marque une tâche comme terminée.

        L'utilisateur peut marquer une tâche comme terminée.

        Args:
            task_id: L'identifiant de la tâche à mettre à jour.

        Returns:
            La tâche mise à jour.

        Raises:
            ValueError: Si aucune tâche ne possède l'identifiant fourni.
        """
        for task in self._tasks:
            if task.id == task_id:
                task.status = "done"
                TaskPersistence.save_tasks(self._tasks, self._file_path)
                return task
        raise ValueError(f"Task with id {task_id} not found")

    # ---------------------------------------------------------------------
    # Utilitaires
    # ---------------------------------------------------------------------
    def get_task_by_id(self, task_id: int) -> Task | None:
        """Retourne la tâche correspondant à l'identifiant, ou ``None``.

        Args:
            task_id: L'identifiant de la tâche.

        Returns:
            La tâche ou ``None`` si elle n'existe pas.
        """
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def clear_tasks(self) -> None:
        """Supprime toutes les tâches et réinitialise le fichier de persistance.

        Cette méthode est utile pour les tests ou pour réinitialiser l'état.
        """
        self._tasks.clear()
        TaskPersistence.save_tasks(self._tasks, self._file_path)
