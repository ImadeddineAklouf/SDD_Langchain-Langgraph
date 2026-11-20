# SDD Pipeline déterministe (LangChain + LangGraph)

Workflow **Spec-Driven Development** complet, avec les mêmes étapes que GitHub spec-kit :
**constitution → specify → clarify → plan → tasks → analyze → implement**.
Le LLM **transforme** les entrées mais n'a jamais le droit d'**inventer**.

📘 Le rapport complet (concepts, architecture, code, choix, exécution réelle) est dans
[`docs/rapport-sdd-pipeline.pdf`](docs/rapport-sdd-pipeline.pdf).

## Architecture « AI factory » : un producteur, un Gardien

D'un côté, le **graphe SDD** produit (spec, plan, tâches, code). De l'autre, le **Gardien**, un graphe LangGraph
compilé à part, vérifie après **chaque** étape la cohérence brief ↔ spec ↔ plan ↔ tâches ↔ code. Au moindre
problème, il lève une alerte et **met le workflow en pause** (`interrupt()` + checkpointer) jusqu'à la décision
humaine : **répondre** (sources `A-xxx`), **corriger** (sources `F-xxx`), **accepter le risque** (justification
obligatoire), **continuer** (avertissements seulement) ou **arrêter**. Le workflow reprend ensuite **au bon nœud**
(celui qui a produit l'erreur), et chaque contrôle et chaque décision sont écrits dans un **journal d'audit**.

| Gravité | Effet | Exemples |
|---|---|---|
| ⛔ bloquante | pause obligatoire | question ouverte, réponse humaine ignorée, citation inventée, exigence sans tâche, fichier de données ou dépendance qu'aucune source ne demande, tests en échec |
| ⚠️ avertissement | pause, l'humain acquitte | nom de fichier choisi par l'IA (format demandé, nom non), élément de plan sans tâche, constat sémantique mineur |
| ℹ️ info | tracé, sans pause | modules de la bibliothèque standard utilisés, fichier partagé entre tâches, risque déjà accepté |

- **Déterministe d'abord** (`guardian.py`) : le gros des contrôles est du code pur. Le Gardien ne fait pas confiance
  aux erreurs rapportées par le producteur : il revalide tout à partir des artefacts.
- **Sémantique ensuite** (LLM, facultatif, idéalement un autre modèle) : contradiction, déformation, hors périmètre.
  Il ne peut **qu'ajouter** des alertes ; un constat qui cite un id inexistant est écarté.
- **Plus d'implémentation tant qu'une question reste ouverte**, et une question sans réponse ne disparaît jamais d'elle-même.
- **Mode usine** (défaut) : pause uniquement sur alerte. **Mode supervisé** : pause aussi à chaque étape.

## Garde-fous du producteur : le LLM propose, du code déterministe dispose

| Garde-fou | Où | Ce qu'il empêche |
|---|---|---|
| Découpage constitution/brief en énoncés `C-001`, `B-001`… **sans LLM** | `ingest.py` | sources non identifiables |
| Sortie structurée `json_schema` + Pydantic `extra="forbid"`, température 0 | `llm.py`, `models.py` | champs inventés, texte libre |
| Citation **mot pour mot** obligatoire pour chaque exigence | `validate_spec` | exigences sans origine |
| **Ancrage lexical** (≥ 70 % des mots porteurs présents dans les sources) | `grounding_ratio` | ajouts (« priorité », « OAuth »…) |
| Chaque phrase du brief → exigence **ou** question ; chaque réponse `A-xxx` → citée par une exigence | `validate_spec` | oublis silencieux, réponses perdues |
| `clarify` ne produit que des questions ; les réponses humaines deviennent des sources `A-xxx` | `clarify`, `gate` | suppositions |
| Choix techno uniquement s'ils sont **nommés** dans une source | `validate_plan` | stack inventée |
| Tâches liées à exigence + plan, critères d'acceptation **copiés** | `validate_tasks` | tâches « bonus » |
| L'implémentation ne touche **que** les fichiers déclarés ; écrite seulement après le feu vert du Gardien | `validate_implementation`, `apply_impl` | effets de bord |
| `verify` lance pytest ; `fix` n'a pas le droit de modifier les tests | `run_tests`, `validate_fix` | « triche » sur les tests |

## Graphe LangGraph

```
START → constitution → specify → clarify ─► guardian ─(alerte)─► gate ⏸ ─► reprise au bon nœud
                                   plan ─► guardian ─(ok)─► tasks → analyze ─► guardian ─► implement ─► guardian
                                   ─► apply_impl ─► (tâche suivante | verify ⇄ fix) ─► guardian ─► END
```

Usage standard de LangGraph : `StateGraph`, arêtes conditionnelles, sous-graphe (le Gardien), réducteur
(`Annotated[list, operator.add]` pour le journal d'audit), `interrupt()` + `Command(resume=…)`, checkpointer SQLite
(reprise avec `--thread`). Côté LangChain : `ChatPromptTemplate | llm.with_structured_output(...)` (LCEL) et
`with_fallbacks` pour le modèle de secours.

## Modèle : Gemini (gratuit) par défaut

Par défaut **Google Gemini `gemini-3.8-flash`** via l'API gratuite (clé sur https://aistudio.google.com/apikey).
Rien ne tourne sur votre machine. Les modèles ouverts dépendent du compte (`gemini-2.5-flash` est fermé aux
nouveaux comptes) : la barre latérale de l'UI liste ceux **réellement disponibles pour votre clé** et permet de
choisir le modèle principal et un modèle de secours (même clé, quota séparé). Si `GROQ_API_KEY` est aussi
renseignée, le secours par défaut est Groq : une panne, un délai dépassé ou un quota épuisé chez Gemini bascule
automatiquement l'appel chez Groq (`SDD_FALLBACK_PROVIDER`). La sortie structurée passe le JSON Schema à Gemini (`response_json_schema`),
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

Collez un brief (et une constitution facultative), lancez : chaque étape s'affiche au fil de l'eau. Quand le
Gardien lève une alerte, la **boîte d'alertes** affiche le détail (code, gravité, ids concernés, étape de reprise)
et vous **répondez**, **corrigez**, **acceptez le risque** (avec justification), **continuez** ou **arrêtez**.
La barre latérale règle le Gardien (mode usine / supervisé, contrôle sémantique, modèle du Gardien). Les artefacts et le code générés
sont visibles dans l'onglet « Fichiers générés », téléchargeables en zip, et écrits dans `out/ui/<session>/`.
L'interface ne parle jamais au modèle elle-même : elle passe par le même graphe et les mêmes validateurs que la CLI.

### Ligne de commande

```bash
sdd examples/brief.md --constitution examples/constitution.md --require-tests
sdd examples/brief.md --supervised      # pause à chaque étape, pas seulement sur alerte
sdd examples/brief.md --no-semantic     # Gardien 100 % déterministe (aucun appel LLM de contrôle)
sdd examples/brief.md --provider groq   # changer de fournisseur ponctuellement
python -m pytest                  # 42 tests hors ligne avec un faux LLM (dont les briefs piégés du Gardien)
```

Le code généré est exécuté par pytest : à lancer de préférence dans un environnement isolé.
