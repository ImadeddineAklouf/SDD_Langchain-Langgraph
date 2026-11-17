class Task:
    """Modèle de données représentant une tâche.

    Chaque instance possède un identifiant entier unique attribué à la création,
    en commençant à 1, ainsi qu'un titre et un statut.
    """

    _id_counter: int = 1

    def __init__(self, title: str, status: str) -> None:
        """Crée une nouvelle tâche.

        Args:
            title: Le titre de la tâche.
            status: Le statut de la tâche (par ex. "pending", "done", ...).
        """
        self.id: int = Task._id_counter
        Task._id_counter += 1
        self.title: str = title
        self.status: str = status

    def __repr__(self) -> str:
        return f"Task(id={self.id}, title={self.title!r}, status={self.status!r})"
