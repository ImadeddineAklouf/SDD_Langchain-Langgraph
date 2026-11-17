# Spécification


> ✅ Validation déterministe : OK


## Exigences

### REQ-001 (constraint)
Le code est écrit en Python 3.11.

_Sources : C-001 « Le code est écrit en Python 3.11. »_


### REQ-002 (constraint)
Les tests utilisent pytest.

_Sources : C-002 « Les tests utilisent pytest. »_


### REQ-003 (functional)
Chaque fonctionnalité est couverte par au moins un test pytest.

_Sources : C-003 « Chaque fonctionnalité est couverte par au moins un test pytest. »_


### REQ-004 (constraint)
Aucune dépendance externe en dehors de la bibliothèque standard et de pytest.

_Sources : C-004 « Aucune dépendance externe en dehors de la bibliothèque standard et de pytest. »_


### REQ-005 (functional)
L'utilisateur peut ajouter une tâche avec un titre.

_Sources : B-001 « L'utilisateur peut ajouter une tâche avec un titre. »_


### REQ-006 (functional)
L'utilisateur peut lister les tâches.

_Sources : B-002 « L'utilisateur peut lister les tâches. »_


### REQ-007 (functional)
L'utilisateur peut marquer une tâche comme terminée.

_Sources : B-003 « L'utilisateur peut marquer une tâche comme terminée. »_


### REQ-008 (functional)
Les tâches sont sauvegardées dans un fichier JSON local.

_Sources : B-004 « Les tâches sont sauvegardées dans un fichier JSON local. »_


### REQ-009 (functional)
Chaque tâche possède un identifiant entier attribué à la création, en commençant à 1.

_Sources : A-001 « Chaque tâche a un identifiant entier attribué à la création, en commençant à 1. »_



## Questions ouvertes

- **Q-001** Où le fichier JSON contenant les tâches est-il stocké ? _(sources : B-004)_
- **Q-002** Quel est le format exact du fichier JSON (structure des objets, champs, etc.) ? _(sources : B-004)_
- **Q-003** Comment l’utilisateur interagit-il avec le système ? S’agit‑il d’une interface en ligne de commande, d’une API, ou d’une autre forme d’interaction ? _(sources : B-001, B-002, B-003)_
- **Q-004** Les tâches sont-elles chargées automatiquement au démarrage du programme ? _(sources : B-004)_
- **Q-005** Les titres de tâches doivent‑ils être uniques ou peut‑on ajouter plusieurs tâches portant le même titre ? _(sources : B-001)_


## Sources

- `C-001` (constitution) Le code est écrit en Python 3.11.
- `C-002` (constitution) Les tests utilisent pytest.
- `C-003` (constitution) Chaque fonctionnalité est couverte par au moins un test pytest.
- `C-004` (constitution) Aucune dépendance externe en dehors de la bibliothèque standard et de pytest.
- `B-001` (brief) L'utilisateur peut ajouter une tâche avec un titre.
- `B-002` (brief) L'utilisateur peut lister les tâches.
- `B-003` (brief) L'utilisateur peut marquer une tâche comme terminée.
- `B-004` (brief) Les tâches sont sauvegardées dans un fichier JSON local.
- `A-001` (answer) Comment l'utilisateur identifie-t-il une tâche lorsqu'il la marque comme terminée ? Utilise-t-on un identifiant unique, le titre, ou un autre mécanisme ? Réponse : Chaque tâche a un identifiant entier attribué à la création, en commençant à 1.
