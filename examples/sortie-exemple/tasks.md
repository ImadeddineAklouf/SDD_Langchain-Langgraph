# Tâches


> ✅ Validation déterministe : OK


Ordre d'exécution : T-001 → T-002 → T-003 → T-004 → T-005 → T-006 → T-007 → T-008 → T-009 → T-010 → T-011 → T-012 → T-013

### T-001 Créer le modèle de données Task
Définir la classe Task avec id, titre et statut, en suivant le modèle de données.

- Exigences : REQ-009
- Plan : PLN-002
- Dépend de : —
- Fichiers : src/task.py

### T-002 Implémenter l'interface de persistance
Créer la classe TaskPersistence pour sauvegarder et charger les tâches depuis un fichier JSON local.

- Exigences : REQ-008
- Plan : PLN-003
- Dépend de : —
- Fichiers : src/persistence.py

### T-003 Développer le composant Task Manager
Mettre en place la logique d'ajout, de listing et de mise à jour des tâches, en utilisant le modèle Task et la persistance.

- Exigences : REQ-005, REQ-006, REQ-007, REQ-009
- Plan : PLN-001, PLN-002, PLN-003
- Dépend de : —
- Fichiers : src/task_manager.py

### T-004 Créer l'interface en ligne de commande
Fournir des commandes pour ajouter, lister et terminer des tâches via la CLI.

- Exigences : REQ-005, REQ-006, REQ-007
- Plan : PLN-004, PLN-001
- Dépend de : —
- Fichiers : src/cli.py

### T-005 Écrire le test d'ajout de tâche
Test pytest pour vérifier que l'utilisateur peut ajouter une tâche avec un titre.

- Exigences : REQ-005
- Plan : PLN-005
- Dépend de : —
- Fichiers : tests/test_add_task.py

### T-006 Écrire le test de listing des tâches
Test pytest pour vérifier que l'utilisateur peut lister les tâches.

- Exigences : REQ-006
- Plan : PLN-005
- Dépend de : —
- Fichiers : tests/test_list_tasks.py

### T-007 Écrire le test de marquage d'une tâche terminée
Test pytest pour vérifier que l'utilisateur peut marquer une tâche comme terminée.

- Exigences : REQ-007
- Plan : PLN-005
- Dépend de : —
- Fichiers : tests/test_mark_task_done.py

### T-008 Écrire le test de persistance des tâches
Test pytest pour vérifier que les tâches sont sauvegardées dans un fichier JSON local.

- Exigences : REQ-008
- Plan : PLN-005
- Dépend de : —
- Fichiers : tests/test_persistence.py

### T-009 Écrire le test de l'attribution d'identifiant de tâche
Test pytest pour vérifier que chaque tâche possède un identifiant entier attribué à la création, en commençant à 1.

- Exigences : REQ-009
- Plan : PLN-005
- Dépend de : —
- Fichiers : tests/test_task_id_assignment.py

### T-010 Écrire le test de stratégie de couverture
Test pytest pour vérifier que chaque fonctionnalité est couverte par au moins un test.

- Exigences : REQ-003
- Plan : PLN-005
- Dépend de : —
- Fichiers : tests/test_coverage_strategy.py

### T-011 Spécifier la version Python 3.11
Créer un fichier .python-version indiquant la version Python 3.11 utilisée pour le projet.

- Exigences : REQ-001
- Plan : PLN-001
- Dépend de : —
- Fichiers : .python-version

### T-012 Définir l'utilisation de pytest
Créer un fichier requirements.txt contenant uniquement la dépendance pytest.

- Exigences : REQ-002
- Plan : PLN-001
- Dépend de : —
- Fichiers : requirements.txt

### T-013 Vérifier l'absence de dépendances externes
Créer un fichier constraints.txt listant les seules dépendances autorisées (standard library + pytest).

- Exigences : REQ-004
- Plan : PLN-001
- Dépend de : —
- Fichiers : constraints.txt

## Matrice de traçabilité

| Exigence | Sources | Tâches |
|---|---|---|
| REQ-001 | C-001 | T-011 |
| REQ-002 | C-002 | T-012 |
| REQ-003 | C-003 | T-010 |
| REQ-004 | C-004 | T-013 |
| REQ-005 | B-001 | T-003, T-004, T-005 |
| REQ-006 | B-002 | T-003, T-004, T-006 |
| REQ-007 | B-003 | T-003, T-004, T-007 |
| REQ-008 | B-004 | T-002, T-008 |
| REQ-009 | A-001 | T-001, T-003, T-009 |
