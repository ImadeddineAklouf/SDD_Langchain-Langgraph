import json
import os
from typing import List
from src.task import Task

class TaskPersistence:
    """Classe responsable de la sauvegarde et du chargement des tâches depuis un fichier JSON local."""

    @staticmethod
    def save_tasks(tasks: List[Task], file_path: str) -> None:
        """Sauvegarde la liste de tâches dans le fichier JSON spécifié."""
        data = [
            {"id": task.id, "title": task.title, "status": task.status}
            for task in tasks
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_tasks(file_path: str) -> List[Task]:
        """Charge les tâches depuis le fichier JSON spécifié."""
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tasks: List[Task] = []
        max_id = 0
        for item in data:
            task = Task(item["title"], item["status"])
            task.id = item["id"]
            tasks.append(task)
            if task.id > max_id:
                max_id = task.id

        Task._id_counter = max_id + 1
        return tasks