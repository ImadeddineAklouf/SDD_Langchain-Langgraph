## 12.1 Résultats de l'exécution finale

L'exécution finale a utilisé la constitution d'exemple, le brief du gestionnaire de tâches et l'option
`--require-tests`. Un script a simulé l'humain : au premier passage de `clarify`, il a répondu à une question,
puis il a tout approuvé. L'intégralité des fichiers produits est dans `examples/sortie-exemple/`.

| Étape | Temps cumulé | Résultat |
|---|---|---|
| `clarify` (1er passage) | 8 s | 1 question : comment identifier une tâche pour la marquer terminée ? |
| réponse humaine | — | « Chaque tâche a un identifiant entier attribué à la création, en commençant à 1. » → source `A-001` |
| `clarify` (2e passage) | 30 s | 9 exigences, dont `REQ-009` qui cite `A-001` ; 5 nouvelles questions (emplacement et format du JSON, mode d'interaction, chargement au démarrage, unicité des titres), approuvées sans réponse |
| `plan` | 58 s | 0 erreur ; choix techniques : uniquement Python 3.11 (`C-001`) et pytest (`C-002`) |
| `analyze` | 192 s | 0 erreur, 0 avertissement, 9/9 exigences couvertes |
| `implement` | 194 s à 612 s | 13 tâches, toutes acceptées par les validateurs |
| `verify` | 732 s | ✅ 10 tests pytest réussis |

**Ce qui fonctionne comme prévu**

- **Aucune technologie inventée** : le plan ne retient que Python 3.11 et pytest, les deux seules cités par la constitution.
- **La réponse humaine est tracée** : l'identifiant entier n'apparaît dans la spec que parce que l'humain l'a donné ;
  l'exigence `REQ-009` le cite (`A-001`) et trois tâches en découlent (modèle `Task`, gestionnaire, test d'attribution
  des identifiants).
- **Le LLM pose des questions au lieu d'inventer** : format du JSON, mode d'interaction, unicité des titres… autant
  de points qu'un agent non contrôlé aurait tranchés seul.
- **Traçabilité complète** :

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

**Ce que l'exécution a révélé (et les leçons)**

- **Les questions approuvées sans réponse laissent le choix au modèle.** Comme les 5 questions du second passage
  ont été approuvées sans réponse, le code a dû trancher : statuts `"pending"`/`"done"`, fichier `tasks.json` par
  défaut, erreur `ValueError` pour un identifiant inconnu… Ce sont des choix simples et cohérents, mais ce sont
  des choix. Le pipeline les signale (questions ouvertes visibles dans `spec.md`, avertissement d'`analyze`) ; c'est
  à l'humain d'y répondre s'il veut garder la main.
- **Les titres Markdown du brief sont ignorés.** Le titre « Gestionnaire de tâches en ligne de commande » n'est pas
  une source : `clarify` a donc demandé, à juste titre, comment l'utilisateur interagit avec le système. Leçon :
  toute information importante doit être écrite dans une ligne du brief, pas dans un titre.
- **Une règle de constitution classée « fonctionnelle ».** « Chaque fonctionnalité est couverte par au moins un
  test » a été classée `functional`. Avec `--require-tests`, il a donc fallu une tâche de test pour elle, et le
  modèle a écrit un test « bouche-trou » (`assert True`) dont le commentaire évoque un plugin `pytest-cov` et une CI
  qui n'existent pas. Deux corrections en découlent : une nouvelle règle déterministe (une exigence qui ne cite
  **que** la constitution ne peut pas être `functional`) et une limite documentée (les commentaires du code ne sont
  pas contrôlés, voir chapitre 14).
