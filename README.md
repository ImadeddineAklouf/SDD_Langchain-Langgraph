# SDD Pipeline déterministe (LangChain + LangGraph)

Workflow **Spec-Driven Development** complet, avec les mêmes étapes que GitHub spec-kit :
**constitution → specify → clarify → plan → tasks → analyze → implement**.
Le LLM **transforme** les entrées mais n'a jamais le droit d'**inventer**.

📘 Le rapport complet (concepts, architecture, code, choix, exécution réelle) est dans
[`docs/rapport-sdd-pipeline.pdf`](docs/rapport-sdd-pipeline.pdf).

## Principe : le LLM propose, du code déterministe dispose

| Garde-fou | Où | Ce qu'il empêche |
|---|---|---|
| Découpage constitution/brief en énoncés `C-001`, `B-001`… **sans LLM** | `ingest.py` | sources non identifiables |
| Sortie structurée `json_schema` + Pydantic `extra="forbid"`, température 0 | `llm.py`, `models.py` | champs inventés, texte libre |
| Citation **mot pour mot** obligatoire pour chaque exigence | `validate_spec` | exigences sans origine |
| **Ancrage lexical** (≥ 70 % des mots porteurs présents dans les sources) | `grounding_ratio` | ajouts (« priorité », « OAuth »…) |
| Chaque phrase du brief → exigence **ou** question ouverte | `validate_spec` | oublis silencieux |
| `clarify` ne produit que des questions ; les réponses humaines deviennent des sources `A-xxx` | `clarify`, `_add_human_input` | suppositions |
| Choix techno uniquement s'ils sont **nommés** dans une source | `validate_plan` | stack inventée |
| Tâches liées à exigence + plan, critères d'acceptation **copiés** | `validate_tasks` | tâches « bonus » |
| `analyze` : contrôle croisé spec ↔ plan ↔ tâches **sans LLM** | `validators.analyze` | incohérences |
| L'implémentation ne touche **que** les fichiers déclarés ; contrôle syntaxique | `validate_implementation` | effets de bord, code cassé |
| `verify` lance pytest ; `fix` n'a pas le droit de modifier les tests | `run_tests`, `validate_fix` | « triche » sur les tests |
| Boucles de correction bornées, points de contrôle humains (`interrupt`) | `graph.py` | décisions sans l'humain |

## Graphe LangGraph

```
START → constitution → specify → clarify → review_spec ⏸ → plan → review_plan ⏸ → tasks → analyze ⏸
      → implement → review_impl ⏸ (une tâche à la fois) → verify ⇄ fix → review_verify ⏸ → END
```

Usage standard de LangGraph : `StateGraph`, arêtes conditionnelles, `interrupt()` + `Command(resume=…)`,
checkpointer SQLite (reprise avec `--thread`). Côté LangChain : `ChatPromptTemplate | llm.with_structured_output(...)`
(LCEL) et `with_fallbacks` pour le modèle de secours.

## Modèle : Gemini (gratuit) par défaut

Par défaut **Google Gemini `gemini-2.5-flash`** via l'API gratuite (clé sur https://aistudio.google.com/apikey),
avec bascule automatique sur **`gemini-2.5-flash-lite`** (même clé, quota séparé) si le quota est atteint.
Rien ne tourne sur votre machine. La sortie structurée passe le JSON Schema à Gemini (`response_json_schema`),
qui contraint la génération.

Toujours disponibles, via la barre latérale de l'UI, `--provider`/`--model` en CLI, ou le `.env` :
`groq` (`openai/gpt-oss-120b`, rapide mais quota journalier) et `ollama` (LLM local, optionnel :
`pip install -e ".[ollama]"`).

## Démarrage

Python **3.11 à 3.13** (sur macOS, si `python3` est en 3.14+ : `brew install python@3.13`).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"
cp .env.example .env              # puis coller la clé Gemini après GOOGLE_API_KEY=
```

### Interface web locale (recommandé)

```bash
sdd-ui                            # ouvre http://localhost:8501
```

Collez un brief (et une constitution facultative), lancez : chaque étape s'affiche au fil de l'eau, et à chaque
point de contrôle (clarify, plan, analyze, implement, verify) vous **approuvez**, **révisez** (réponses aux
questions + retour libre, qui deviennent des sources citables) ou **arrêtez**. Les artefacts et le code générés
sont visibles dans l'onglet « Fichiers générés », téléchargeables en zip, et écrits dans `out/ui/<session>/`.
L'interface ne parle jamais au modèle elle-même : elle passe par le même graphe et les mêmes validateurs que la CLI.

### Ligne de commande

```bash
sdd examples/brief.md --constitution examples/constitution.md --require-tests
sdd examples/brief.md --provider groq   # changer de fournisseur ponctuellement
pytest                            # 22 tests hors ligne avec un faux LLM
```

Le code généré est exécuté par pytest : à lancer de préférence dans un environnement isolé.
