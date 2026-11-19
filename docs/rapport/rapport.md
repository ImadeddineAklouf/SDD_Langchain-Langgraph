---
title: "SDD Pipeline déterministe"
subtitle: "Un workflow Spec-Driven Development avec LangChain et LangGraph, où l'IA n'invente rien"
author: "Rapport de projet, de A à Z"
date: "Novembre 2026"
lang: fr
---

# Avant-propos : comment lire ce rapport

Ce document explique **tout** le projet : pourquoi il existe, quels problèmes il résout, comment il est construit,
comment chaque ligne de code s'articule avec les autres, quels choix ont été faits et pourquoi, et comment
l'utiliser. Il est écrit pour être lu dans l'ordre par quelqu'un qui découvre le projet, mais chaque chapitre
peut aussi être consulté seul.

- Les **chapitres 1 et 2** posent le problème et la méthode (Spec-Driven Development, spec-kit).
- Les **chapitres 3 et 4** expliquent les briques techniques : LangChain puis LangGraph, avec leurs concepts standards.
- Les **chapitres 5 à 8** décrivent l'architecture, chaque étape du pipeline, les garde-fous et le modèle de données.
- Les **chapitres 9 à 12** couvrent le choix du modèle, le code fichier par fichier, les tests et l'exécution réelle.
- Les **chapitres 13 à 15** donnent le mode d'emploi, les limites et un glossaire.
- L'**annexe** contient l'intégralité du code source.

Convention : les noms de fichiers, de fonctions et de commandes sont écrits `comme ceci`.

# 1. Le problème : pourquoi un SDD « déterministe » ?

## 1.1 Le développement assisté par IA et ses dérives

Quand on demande à un modèle de langage (LLM) de « coder une application de gestion de tâches », il produit
volontiers beaucoup plus que ce qui a été demandé : des priorités, des dates d'échéance, une authentification,
une base de données, une API REST… Rien de tout cela n'était dans la demande. Le modèle **complète** avec ce qui
lui semble probable, parce qu'il est entraîné à produire du texte plausible, pas à respecter un périmètre.

Ce comportement pose trois problèmes concrets :

1. **Dérive du périmètre** : on livre des fonctionnalités que personne n'a demandées, qu'il faut maintenir, tester et sécuriser.
2. **Hallucination** : le modèle affirme des contraintes ou des règles métier qui n'existent pas (« le titre est limité à 100 caractères »).
3. **Perte de traçabilité** : on ne sait plus pourquoi une ligne de code existe, ni quelle exigence elle sert.

## 1.2 Le Spec-Driven Development (SDD)

Le **Spec-Driven Development** répond à ce problème en inversant l'ordre habituel : au lieu de partir du code,
on part d'une **spécification** écrite, qu'on transforme progressivement en plan technique, puis en liste de
tâches, puis en code. Chaque étape produit un artefact relu et validé avant de passer à la suivante. La
spécification devient la source de vérité ; le code n'en est qu'une conséquence.

## 1.3 Ce que ce projet ajoute : le déterminisme

Le SDD classique avec un agent IA reste vulnérable : à chaque étape, le LLM peut encore ajouter des choses.
L'objectif de ce projet est de rendre le workflow **le plus déterministe possible** :

> **Le LLM propose, du code déterministe dispose.**

Concrètement :

- Tout ce qui peut être fait **sans LLM** est fait sans LLM (découpage du besoin, contrôles de cohérence, exécution des tests, rendu des documents).
- Chaque sortie du LLM est **structurée** (JSON conforme à un schéma) puis **vérifiée par du code** : citations exactes, couverture, références, périmètre des fichiers.
- Toute information doit avoir une **origine traçable** : une phrase du besoin, de la constitution, ou une réponse de l'humain.
- Quand une information manque, le LLM doit **poser une question** au lieu de supposer, et c'est l'humain qui répond.
- L'humain valide chaque étape (**points de contrôle**).

« Déterministe » ne veut pas dire que le LLM produit toujours exactement le même texte (même à température 0,
ce n'est pas garanti). Cela veut dire que **les règles d'acceptation sont déterministes** : un artefact qui
contient une information non sourcée est rejeté, à chaque fois, quel que soit le texte exact produit.

# 2. Le workflow SDD : les étapes de spec-kit

## 2.1 spec-kit en bref

**spec-kit** est la boîte à outils open source de GitHub pour le Spec-Driven Development. Elle fournit des
commandes à utiliser dans un agent de code (`/speckit.constitution`, `/speckit.specify`, etc.) qui guident
l'agent étape par étape. Ce projet **n'utilise pas l'outil spec-kit** : il reprend **le même enchaînement
d'étapes**, mais l'implémente lui-même avec LangGraph, afin d'ajouter à chaque étape des contrôles
déterministes que spec-kit ne fait pas (spec-kit s'appuie sur la discipline de l'agent ; ici, c'est du code qui
vérifie).

## 2.2 Les sept étapes et leur correspondance

| Étape spec-kit | Rôle dans spec-kit | Nœud(s) LangGraph de ce projet | LLM ? | Contrôle humain |
|---|---|---|---|---|
| `constitution` | Principes non négociables du projet | `constitution` | Non | — |
| `specify` | Décrire le *quoi* et le *pourquoi* | `specify` | Oui | — |
| `clarify` | Lever les ambiguïtés par questions | `clarify` + `review_spec` | Oui (questions seulement) | Oui |
| `plan` | Décider le *comment* technique | `plan` + `review_plan` | Oui | Oui |
| `tasks` | Découper en tâches ordonnées | `tasks` | Oui | (via analyze) |
| `analyze` | Vérifier la cohérence spec/plan/tâches | `analyze` | **Non** | Oui |
| `implement` | Exécuter les tâches | `implement` + `review_impl` + `verify` + `fix` + `review_verify` | Oui (sauf verify) | Oui |

Deux différences volontaires avec spec-kit :

- **constitution** : dans spec-kit, l'agent rédige la constitution à partir de vos principes. Ici, la constitution
  est un fichier que **vous** écrivez ; le pipeline la découpe en énoncés numérotés sans la modifier. Elle ne
  peut donc contenir aucune invention.
- **analyze** : dans spec-kit, c'est un LLM qui relit les artefacts. Ici, c'est **100 % du code** : les mêmes
  entrées donnent toujours le même rapport.

## 2.3 Les entrées du pipeline

Le pipeline prend deux textes écrits par l'humain :

- **la constitution** (`examples/constitution.md`) : les règles globales (langage, outils de test, dépendances autorisées…) ;
- **le brief** (`examples/brief.md`) : le besoin fonctionnel.

Exemple de brief fourni :

```markdown
# Gestionnaire de tâches en ligne de commande
- L'utilisateur peut ajouter une tâche avec un titre.
- L'utilisateur peut lister les tâches.
- L'utilisateur peut marquer une tâche comme terminée.
- Les tâches sont sauvegardées dans un fichier JSON local.
```

Ces deux textes, plus les réponses de l'humain pendant le processus, sont **les seules sources de vérité**.

# 3. LangChain : les concepts utilisés

LangChain est une bibliothèque Python qui fournit des abstractions standard pour travailler avec les LLM.
Le projet en utilise uniquement les briques classiques, décrites ici.

## 3.1 Le Chat Model

Un *chat model* est un objet qui représente un LLM conversationnel. Tous les fournisseurs (OpenAI, Groq,
Google, Ollama…) exposent la même interface : `invoke(messages)` renvoie un message de réponse. Dans le projet :

```python
from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=2)
```

Changer de fournisseur revient à changer de classe (`ChatOllama`, `ChatGoogleGenerativeAI`) : le reste du code
ne bouge pas. C'est tout l'intérêt de l'abstraction, et c'est ce que fait `src/sdd/llm.py`.

**`temperature=0`** demande au modèle de choisir systématiquement le mot le plus probable : c'est le réglage le
moins créatif et le plus reproductible.

## 3.2 Les messages

Une conversation est une liste de messages typés :

- `SystemMessage` : les instructions de rôle et les règles (« tu es analyste, tu n'inventes rien… ») ;
- `HumanMessage` : la demande (ici, les sources et artefacts à transformer) ;
- `AIMessage` : une réponse du modèle.

Le projet utilise `AIMessage` + `HumanMessage` pour la **boucle de correction** : on remet au modèle sa propre
réponse, suivie de la liste des erreurs détectées, et on lui demande de corriger.

## 3.3 Les prompt templates

Un `ChatPromptTemplate` est un gabarit de conversation avec des variables. Le projet en construit un par appel :

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", system),                                   # règles + rôle de l'étape
    ("human", "{input}"),                                 # sources / artefacts
    MessagesPlaceholder("corrections", optional=True),    # historique des corrections
])
```

`MessagesPlaceholder` est un emplacement où l'on injecte une liste de messages : vide au premier essai, puis
remplie des réponses rejetées et des erreurs.

## 3.4 LCEL : chaîner avec `|`

LCEL (*LangChain Expression Language*) permet de composer des composants avec l'opérateur `|`, comme un pipe
Unix. La sortie de l'un devient l'entrée du suivant :

```python
chain = prompt | llm.with_structured_output(Spec, method="json_schema")
spec = chain.invoke({"input": sources_text, "corrections": []})
```

Le gabarit produit des messages, le modèle les reçoit et renvoie un objet `Spec`.

## 3.5 La sortie structurée : `with_structured_output`

C'est la brique la plus importante pour le déterminisme. `llm.with_structured_output(Schema)` transforme le
modèle pour qu'il renvoie **un objet Python validé** (ici une classe Pydantic) au lieu de texte libre.

Il existe plusieurs méthodes :

- `function_calling` : le schéma est présenté comme un « outil » que le modèle doit appeler. Le modèle peut
  refuser et répondre en texte (c'est arrivé pendant les essais réels, voir chapitre 12).
- `json_schema` : le fournisseur **contraint la génération** pour que la sortie respecte le schéma JSON. C'est
  la méthode choisie par défaut (variable `SDD_STRUCTURED_METHOD`).
- `json_mode` : le modèle renvoie du JSON, sans garantie de schéma.

## 3.6 Pydantic

Pydantic est la bibliothèque de validation de données utilisée par LangChain. Chaque artefact (spec, plan,
tâches, implémentation) est une classe Pydantic. Le projet les déclare toutes avec `extra="forbid"` : un champ
inconnu provoque une erreur. Le LLM ne peut donc pas glisser d'information « en douce » hors du schéma.
Les contraintes `min_length=1` imposent par exemple qu'une exigence ait au moins une citation.

## 3.7 Les fallbacks : `with_fallbacks`

Tout composant LangChain (un *runnable*) possède la méthode standard `with_fallbacks([...])` : si l'appel
échoue (quota atteint, panne, erreur réseau), LangChain réessaie automatiquement avec le composant de secours.
Le projet s'en sert pour basculer vers un second modèle gratuit :

```python
model = llm.with_structured_output(Spec, method="json_schema")
model = model.with_fallbacks([secours.with_structured_output(Spec, method="json_schema")])
chain = prompt | model
```

Le secours est appliqué **après** `with_structured_output` : les deux modèles sont contraints par le même schéma
et passent par les mêmes validateurs. Changer de modèle ne change donc pas les règles d'acceptation.

# 4. LangGraph : les concepts utilisés

LangGraph est la bibliothèque de l'écosystème LangChain pour construire des **applications à états** sous forme
de graphe : des étapes (nœuds) reliées par des transitions (arêtes), avec un état partagé, des boucles, des
pauses pour l'humain et de la persistance. Le projet l'utilise de façon standard.

## 4.1 L'état (`State`)

L'état est un dictionnaire typé (`TypedDict`) qui circule entre les nœuds. Dans le projet, `SDDState` contient
notamment : les sources, la spec, le plan, les tâches, l'ordre d'exécution, la tâche courante, le résultat des
tests, la dernière décision humaine.

Chaque nœud **reçoit l'état** et **renvoie un dictionnaire partiel** : seules les clés renvoyées sont mises à
jour. L'état ne contient que des types simples (dict, list, str) pour rester facilement sérialisable.

## 4.2 Les nœuds

Un nœud est une simple fonction Python `state -> mise à jour`. Exemple réel :

```python
def verify(state: SDDState):
    res = run_tests(ws)
    return {"verify": res}
```

## 4.3 Les arêtes et les arêtes conditionnelles

- `add_edge("a", "b")` : après `a`, on va toujours à `b`.
- `add_conditional_edges("a", fonction_de_routage, [destinations possibles])` : après `a`, une fonction lit
  l'état et choisit la suite. C'est ce qui crée les **boucles** (réviser une étape, passer à la tâche suivante,
  relancer une correction).
- `START` et `END` sont les points d'entrée et de sortie.

## 4.4 La compilation

`StateGraph(...).compile(checkpointer=...)` transforme la description en un objet exécutable qui expose
`invoke`, `stream`, etc.

## 4.5 Le checkpointer et les threads

Un **checkpointer** enregistre l'état après chaque étape. Le projet utilise `SqliteSaver` (fichier
`out/checkpoints.db`) dans la CLI et `MemorySaver` dans les tests. Chaque exécution est identifiée par un
`thread_id` : on peut fermer le programme et reprendre plus tard exactement au même endroit
(`sdd ... --thread <id>`).

## 4.6 Human-in-the-loop : `interrupt()` et `Command(resume=...)`

`interrupt(payload)` met le graphe en pause et renvoie `payload` à l'appelant (dans `result["__interrupt__"]`).
L'appelant montre l'artefact à l'humain, récupère sa décision, puis relance le graphe avec
`graph.invoke(Command(resume=decision), config)`. La valeur de `decision` devient la valeur de retour
d'`interrupt()` à l'intérieur du nœud.

**Point important (et piège classique)** : à la reprise, LangGraph **ré-exécute le nœud depuis le début**. Si
un nœud appelle le LLM *puis* `interrupt()`, le LLM est rappelé à la reprise, ce qui coûte un appel et peut
produire un résultat différent. Le projet suit donc la bonne pratique : **les nœuds qui appellent le LLM ne
contiennent jamais d'`interrupt()`** ; les pauses sont dans des nœuds dédiés (`review_spec`, `review_plan`,
`analyze`, `review_impl`, `review_verify`) qui ne font que lire l'état, écrire les fichiers et attendre. Ce piège a
été rencontré et corrigé pendant le développement (voir chapitre 12).

# 5. Architecture générale

## 5.1 Vue d'ensemble du graphe

<figure class="diagram">
<!-- GRAPH_SVG -->
<figcaption>Le graphe LangGraph du pipeline. En vert pointillé : les retours en arrière décidés par l'humain. « Quitter » mène à END depuis chaque point de contrôle.</figcaption>
</figure>

Note : `analyze` est à la fois un contrôle déterministe et un point de contrôle humain ; `fix` boucle avec `verify` au plus `max_fix_attempts` fois avant de passer à `review_verify`.

## 5.2 Le schéma « générer → valider → corriger »

Tous les nœuds LLM utilisent la même fonction, `generate()` (dans `graph.py`) :

1. construire la chaîne LCEL `prompt | llm.with_structured_output(Schema)` ;
2. l'appeler ; si la sortie n'est pas conforme au schéma, c'est une erreur ;
3. passer l'objet au **validateur déterministe** de l'étape ;
4. s'il y a des erreurs, les renvoyer au modèle (« ta réponse est rejetée : … corrige uniquement ces points ») ;
5. recommencer, au plus `max_retries` fois (3 par défaut) ;
6. si des erreurs subsistent, l'artefact est quand même transmis **avec ses erreurs** au point de contrôle
   humain suivant, qui décide. Rien n'est jamais caché.

## 5.3 La traçabilité par identifiants

Tout porte un identifiant stable :

| Préfixe | Nature | Produit par |
|---|---|---|
| `C-001` | Énoncé de la constitution | code (découpage) |
| `B-001` | Énoncé du brief | code (découpage) |
| `A-001` | Réponse de l'humain à une question | code (à partir de la saisie humaine) |
| `F-001` | Retour libre de l'humain | code (découpage du retour) |
| `REQ-001` | Exigence | LLM, vérifié par code |
| `Q-001` | Question ouverte | LLM, renumérotée par code |
| `PLN-001` | Élément du plan | LLM, vérifié par code |
| `T-001` | Tâche | LLM, vérifié par code |

La chaîne complète est donc : **phrase source → exigence → élément de plan → tâche → fichier de code**. Le
fichier `tasks.md` contient une matrice de traçabilité qui la rend visible.

## 5.4 Les artefacts produits

Dans le dossier de sortie (`out/` par défaut) :

| Fichier | Contenu |
|---|---|
| `sources.json` | toutes les sources numérotées |
| `spec.md` / `spec.json` | exigences, citations, questions ouvertes |
| `plan.md` / `plan.json` | éléments du plan, choix techniques |
| `tasks.md` / `tasks.json` | tâches, ordre d'exécution, matrice de traçabilité |
| `analyze.md` | rapport de cohérence (erreurs, avertissements, couverture) |
| `workspace/` | le code généré |
| `run_log.json` | (scripts d'essai) journal des points de contrôle |
| `verify.md` | résultat de pytest sur le code généré |
| `checkpoints.db` | l'état LangGraph (reprise de session) |

# 6. Les étapes une par une

## 6.1 `constitution` (sans LLM)

**Entrée** : le texte de la constitution. **Sortie** : des énoncés `C-xxx`.

La fonction `split_statements()` (`ingest.py`) découpe le texte : une ligne = un énoncé (les puces `-`, `*`,
`1.` sont retirées, les titres `#` ignorés), et une ligne contenant plusieurs phrases est coupée après `.`, `!`
ou `?` suivi d'une majuscule. Le découpage est **purement déterministe** : le même texte donne toujours les
mêmes identifiants. C'est essentiel, car toutes les citations reposent sur ces identifiants.

## 6.2 `specify` (LLM)

Au premier passage, le brief est découpé de la même façon en `B-xxx`. Le LLM reçoit ensuite toutes les sources
et doit produire un objet `Spec` : une liste d'exigences et éventuellement des questions ouvertes.

Chaque exigence (`Requirement`) contient : un identifiant, un type (`functional`, `non_functional`,
`constraint`), un énoncé, **au moins une citation** (`source_id` + `quote` copiée mot pour mot) et des critères
d'acceptation, eux aussi cités.

Le validateur `validate_spec` vérifie :

1. que chaque citation pointe vers une source existante ;
2. que la citation est **littéralement présente** dans la source (après normalisation des espaces, de la casse et des apostrophes) ;
3. que l'énoncé de l'exigence est **ancré** dans ses sources (voir 7.2) ;
4. que chaque énoncé du brief est **couvert** par une exigence ou une question : rien n'est oublié silencieusement ;
5. l'absence d'identifiants dupliqués ;
6. qu'une exigence qui ne cite **que** la constitution n'est pas classée `functional` (une règle globale n'est pas
   une fonctionnalité ; cette règle vient de l'exécution réelle, chapitre 12).

## 6.3 `clarify` (LLM) et `review_spec` (humain)

Comme dans spec-kit, l'étape `clarify` cherche les zones ambiguës. Le LLM a un rôle **strictement limité** :
il ne produit que des questions (schéma `Clarifications`, 5 au maximum), jamais de réponse, et ne modifie pas la
spec. Le code fusionne ces questions avec celles que `specify` avait déjà posées, supprime les doublons et les
renumérote `Q-001`, `Q-002`… de façon stable.

Puis `review_spec` met le graphe en pause et présente la spec et les questions. L'humain peut :

- **approuver** : on passe au plan (les questions sans réponse restent signalées par `analyze`) ;
- **réviser** : il répond à tout ou partie des questions et/ou donne un retour libre. Chaque réponse devient une
  source `A-xxx` (« *question* Réponse : *réponse* »), chaque phrase de retour une source `F-xxx`, puis `specify`
  est relancé. Le LLM peut désormais citer ces nouvelles sources : **l'information ajoutée a une origine humaine
  traçable** ;
- **quitter**.

Exemple réel (chapitre 12) : sur le brief du gestionnaire de tâches, `clarify` a demandé comment identifier une
tâche pour la marquer terminée (identifiant, index, titre ?). C'est exactement le type d'information qu'un LLM
non contrôlé aurait inventé.

## 6.4 `plan` (LLM) et `review_plan` (humain)

Le LLM reçoit les sources et la spec validée, et produit un `Plan` : des éléments (`component`, `data_model`,
`interface`, `decision`), chacun lié à au moins une exigence, et une liste de **choix techniques**.

`validate_plan` vérifie que chaque exigence est couverte par au moins un élément, que les références existent,
et surtout qu'**aucune technologie n'est inventée** : le nom de chaque choix technique doit apparaître dans le
texte d'une des sources citées. Si la constitution dit « Python 3.11 » et « pytest », le plan peut les retenir ;
s'il propose « FastAPI » ou « SQLite » sans source, il est rejeté. Si aucune techno n'est imposée, le prompt
demande de laisser la liste vide et de poser une question.

## 6.5 `tasks` (LLM)

Le LLM découpe le plan en tâches `Task` : titre, description, exigences et éléments de plan servis, dépendances,
**fichiers** que la tâche a le droit d'écrire, critères d'acceptation.

`validate_tasks` vérifie :

- les références (exigences, éléments de plan, dépendances) ;
- que chaque exigence est couverte par au moins une tâche ;
- que les critères d'acceptation sont **des copies exactes** des critères des exigences référencées : une tâche ne peut pas s'inventer un critère ;
- que les chemins de fichiers sont relatifs et ne remontent pas (`..`) ;
- l'absence de cycle dans les dépendances.

**Option `--require-tests`.** Les essais réels ont montré qu'une phrase comme « chaque fonctionnalité est
couverte par un test » pouvait être « couverte » de façon purement nominale (une tâche qui configure pytest
sans écrire aucun test). Avec l'option `--require-tests`, une règle déterministe s'ajoute : **chaque exigence
fonctionnelle doit être servie par au moins une tâche qui écrit un fichier `test_*.py`**. C'est une option
explicite (et non une déduction du texte) pour que la règle reste prévisible.

Le code calcule ensuite l'**ordre d'exécution** par tri topologique, en départageant par identifiant : l'ordre
est donc toujours le même pour un même plan.

## 6.6 `analyze` (sans LLM, humain)

Cette étape correspond à `/speckit.analyze`, mais elle est entièrement déterministe. Elle relance tous les
validateurs sur l'ensemble spec + plan + tâches et ajoute des avertissements :

- éléments du plan qui n'ont aucune tâche ;
- fichiers modifiés par plusieurs tâches (et l'ordre dans lequel elles passeront) ;
- questions ouvertes restées sans réponse.

Elle écrit `analyze.md` et `tasks.md` (avec la matrice de traçabilité) et met le graphe en pause. Approuver lance
l'implémentation ; réviser renvoie à `tasks` avec le retour de l'humain.

## 6.7 `implement` (LLM) et `review_impl` (humain)

Les tâches sont exécutées **une par une**, dans l'ordre calculé. Pour chaque tâche, le LLM reçoit :

- la tâche, ses exigences et ses éléments de plan ;
- les choix techniques ;
- **le contenu de tous les fichiers déjà écrits** (pour réutiliser les classes et fonctions existantes au lieu de les réinventer) ;
- les retours humains.

Il renvoie un objet `Implementation` : la liste des fichiers avec leur contenu complet. `validate_implementation`
vérifie que **seuls les fichiers déclarés par la tâche** sont touchés, et que chaque fichier est syntaxiquement
valide (`ast.parse` pour Python, `json.loads` pour JSON, `tomllib` pour TOML).

`review_impl` montre le code à l'humain. Les fichiers ne sont écrits sur le disque **qu'après approbation**.

## 6.8 `verify` (sans LLM), `fix` (LLM) et `review_verify` (humain)

Quand toutes les tâches sont faites, `verify` lance `pytest` dans `workspace/` (avec un délai maximal) et écrit
`verify.md`. Si les tests échouent, `fix` demande au LLM une correction, avec deux règles strictes vérifiées par
`validate_fix` :

- **les tests sont l'oracle** : il est interdit de modifier un fichier de test (sinon le modèle pourrait « réparer » en affaiblissant les tests) ;
- seuls les fichiers déclarés par les tâches peuvent être modifiés.

La boucle `verify → fix → verify` est bornée (`max_fix_attempts`, 3 par défaut). Ensuite, `review_verify`
présente le résultat à l'humain, qui peut approuver, demander une nouvelle série de corrections avec un retour,
ou quitter.

Remarque : un projet sans fichier `test_*.py` est considéré comme « aucun test » et non comme un échec. Le
pipeline n'invente pas de tests : si vous en voulez, écrivez-le dans la constitution (c'est ce que fait
l'exemple : « Chaque fonctionnalité est couverte par au moins un test pytest »).

# 7. Les garde-fous de déterminisme en détail

## 7.1 Tableau récapitulatif

| Garde-fou | Où | Ce qu'il empêche |
|---|---|---|
| Découpage des entrées sans LLM | `ingest.py` | sources non identifiables |
| Schémas Pydantic `extra="forbid"` + sortie `json_schema` | `models.py`, `llm.py` | champs inventés, réponses en texte libre |
| Température 0 | `llm.py` | variabilité inutile |
| Citation mot pour mot | `validate_spec` | exigences sans origine, fausses citations |
| Ancrage lexical | `grounding_ratio` | ajouts de vocabulaire (« priorité », « OAuth »…) |
| Couverture du brief | `validate_spec` | oublis silencieux |
| Questions au lieu de suppositions | prompts, `clarify` | informations supposées |
| Réponses humaines = sources citables | `_add_human_input` | informations sans origine |
| Technos uniquement si nommées | `validate_plan` | stack inventée |
| Critères d'acceptation copiés | `validate_tasks` | critères inventés |
| Couverture exigences ↔ plan ↔ tâches | `validate_plan`, `validate_tasks`, `analyze` | exigences oubliées, tâches orphelines |
| Ordre topologique stable | `topological_order` | exécution non reproductible |
| Périmètre des fichiers | `validate_implementation`, `validate_fix` | effets de bord |
| Contrôle syntaxique | `syntax_errors` | code cassé écrit sur disque |
| Tests intouchables | `validate_fix` | « triche » sur les tests |
| Boucles bornées | `generate`, `max_fix_attempts` | boucles infinies, coûts |
| Points de contrôle humains | `review_*`, `analyze` | décisions prises sans l'humain |

## 7.2 L'ancrage lexical, expliqué

La citation garantit qu'une exigence *pointe* vers une source réelle, mais l'énoncé de l'exigence pourrait
quand même ajouter des éléments (« L'utilisateur peut ajouter une tâche avec un titre, **une priorité et une date
d'échéance** » en citant « ajouter une tâche avec un titre »). L'ancrage lexical détecte ce cas :

1. on extrait les **mots porteurs de sens** de l'énoncé : minuscules, accents retirés, mots de plus de 2 lettres,
   hors mots-outils français/anglais (« le », « de », « the »…) ;
2. on fait de même pour le texte des sources citées ;
3. un mot de l'énoncé est « connu » s'il apparaît dans les sources, ou s'il partage ses 5 premières lettres avec
   un mot des sources (tolérance pour « sauvegarder / sauvegardées ») ;
4. le **ratio d'ancrage** = mots connus / mots porteurs. En dessous de 70 %, l'exigence est rejetée, et le
   message d'erreur liste les mots inconnus, ce qui aide le modèle à se corriger.

C'est une **heuristique** : elle attrape les ajouts de vocabulaire, mais pas une paraphrase qui changerait le
sens avec les mêmes mots. C'est pour cela que la revue humaine reste indispensable.

## 7.3 Pourquoi pas un LLM « juge » ?

On aurait pu demander à un second LLM de vérifier le premier. Ce choix a été écarté : un juge LLM est lui-même
non déterministe et peut halluciner. Tous les contrôles sont donc du code ordinaire, testable et reproductible.

# 8. Le modèle de données

Tous les schémas sont dans `src/sdd/models.py` et héritent de `Strict` (Pydantic, `extra="forbid"`).

| Classe | Champs principaux | Rôle |
|---|---|---|
| `SourceStatement` | `id`, `text`, `origin` (brief / constitution / answer / feedback) | une source de vérité |
| `Citation` | `source_id`, `quote` | lien vérifiable vers une source |
| `AcceptanceCriterion` | `text`, `citations` | critère d'acceptation sourcé |
| `Requirement` | `id`, `kind`, `statement`, `citations`, `acceptance_criteria` | exigence |
| `OpenQuestion` | `id`, `question`, `source_ids` | question pour l'humain |
| `Spec` | `requirements`, `open_questions` | sortie de `specify` |
| `Clarifications` | `questions` (≤ 5) | sortie de `clarify` |
| `PlanItem` | `id`, `kind`, `title`, `description`, `requirement_ids` | élément du plan |
| `TechChoice` | `name`, `source_ids` | technologie imposée par une source |
| `Plan` | `items`, `tech_choices`, `open_questions` | sortie de `plan` |
| `Task` | `id`, `title`, `description`, `requirement_ids`, `plan_item_ids`, `depends_on`, `files`, `acceptance_criteria` | tâche |
| `TaskList` | `tasks` | sortie de `tasks` |
| `FileChange` | `path`, `content` | un fichier produit |
| `Implementation` | `task_id`, `files` | sortie de `implement` et `fix` |
| `Review` | `action` (approve / revise / abort), `answers`, `feedback` | décision humaine |

Les descriptions des champs (`Field(description=...)`) font partie du schéma JSON envoyé au modèle : elles lui
expliquent ce que chaque champ doit contenir.

# 9. Le choix du modèle

## 9.1 Les critères

Le projet est personnel : le modèle doit être **gratuit**. Il doit aussi bien gérer les **sorties structurées**
(c'est le cœur du pipeline), suivre précisément des consignes, être assez rapide pour un usage interactif, et
avoir une fenêtre de contexte suffisante pour recevoir les sources, la spec, le plan et le code déjà écrit.

## 9.2 Le choix : Groq + `openai/gpt-oss-120b`

**Groq** propose une API gratuite (avec des quotas journaliers) et une inférence très rapide. Le premier choix
était Llama 3.3 70B, mais au moment des essais ce modèle **n'était plus proposé** par Groq. La liste des modèles
disponibles a été lue directement depuis l'API, et le choix s'est porté sur **`openai/gpt-oss-120b`**, un modèle
à poids ouverts de 120 milliards de paramètres, qui supporte la sortie structurée par JSON Schema. Dans les
essais réels, une étape `specify` prend quelques secondes et une tâche d'implémentation 10 à 30 secondes.

## 9.3 Les quotas gratuits et le modèle de secours

Le tier gratuit de Groq limite notamment le nombre de **tokens par jour et par modèle** (200 000 pour
`gpt-oss-120b` au moment des essais). Un projet complet consomme beaucoup de tokens, surtout à l'étape
`implement`, qui renvoie au modèle tout le code déjà écrit. Pendant les essais, le quota journalier a été atteint
(erreur HTTP 429). D'où le **modèle de secours** (`SDD_FALLBACK_MODEL`, par défaut `openai/gpt-oss-20b`, qui a son
propre quota), branché avec le mécanisme standard `with_fallbacks` (chapitre 3.7). Le secours est plus petit,
donc un peu moins fiable, mais il est soumis aux mêmes schémas et aux mêmes validateurs.

## 9.4 Les alternatives (changement par simple variable d'environnement)

| `SDD_PROVIDER` | Modèle par défaut | Avantages | Inconvénients |
|---|---|---|---|
| `groq` (défaut) | `openai/gpt-oss-120b` | gratuit, très rapide, gros modèle | quotas, données envoyées à un tiers |
| `ollama` | `qwen2.5:14b` | 100 % local, hors ligne, privé | lent sans GPU, moins fiable sur les gros schémas |
| `gemini` | `gemini-2.5-flash` | quota gratuit généreux, gros contexte | données du tier gratuit utilisables pour l'entraînement |

Le modèle exact se règle avec `SDD_MODEL` ; la méthode de sortie structurée avec `SDD_STRUCTURED_METHOD`. La
liste des modèles gratuits évolue vite : c'est pourquoi rien n'est figé dans le code métier.

# 10. Le code, fichier par fichier

Le code complet est en annexe. Ce chapitre explique le rôle de chaque fichier et ses points clés.

## 10.1 Arborescence

```text
sdd-pipeline/
├── pyproject.toml          dépendances et commande `sdd`
├── .env.example            modèle de configuration (clé API, fournisseur)
├── .gitignore              exclut .env, out/, caches
├── README.md               présentation courte
├── examples/
│   ├── brief.md            besoin d'exemple
│   ├── constitution.md     constitution d'exemple
│   └── sortie-exemple/     sortie réelle complète (chapitre 12)
├── docs/
│   ├── rapport-sdd-pipeline.pdf   ce rapport
│   └── rapport/            sources du rapport (Markdown, CSS, scripts)
├── src/sdd/
│   ├── models.py           schémas Pydantic
│   ├── ingest.py           découpage déterministe en sources
│   ├── prompts.py          prompts système de chaque étape
│   ├── llm.py              fabrique du chat model
│   ├── validators.py       tous les contrôles déterministes
│   ├── render.py           rendu Markdown + écriture des fichiers
│   ├── graph.py            le graphe LangGraph
│   ├── cli.py              interface en ligne de commande
│   ├── runner.py           pilotage pas à pas du graphe (utilisé par l'UI)
│   └── ui.py               interface web locale (Streamlit)
└── tests/
    ├── test_pipeline.py    tests hors ligne du pipeline
    └── test_ui.py          tests hors ligne de l'interface
```

## 10.2 `models.py`

Décrit au chapitre 8. Point clé : `Strict` avec `extra="forbid"`, et des `min_length=1` là où une référence est
obligatoire (citations, exigences servies, fichiers d'une tâche).

## 10.3 `ingest.py`

Une seule fonction, `split_statements(text, prefix, origin, start)`, avec deux expressions régulières : une pour
retirer les puces, une pour couper les phrases. Le paramètre `start` permet de continuer une numérotation (par
exemple `F-003` après deux retours précédents).

## 10.4 `prompts.py`

Un bloc commun `GROUND_RULES` (n'utiliser que les sources, ne rien ajouter, poser des questions, citer mot pour
mot, respecter le schéma, ne jamais répondre en texte libre, rédiger en français), puis un prompt par rôle :
`SPECIFY` (analyste), `CLARIFY` (relecteur), `PLAN` (architecte), `TASKS` (chef de projet), `IMPLEMENT`
(développeur), `FIX` (développeur qui corrige). Les prompts **orientent** le modèle ; ce sont les validateurs qui
**garantissent**.

## 10.5 `llm.py`

`get_llm()` lit `SDD_PROVIDER` et `SDD_MODEL` et instancie le bon chat model, toujours avec `temperature=0`
(et un `seed` quand le fournisseur le permet). Les imports sont faits à l'intérieur de la fonction : on
n'installe que le paquet du fournisseur utilisé. `structured_method()` renvoie la méthode de sortie structurée.

## 10.6 `validators.py`

Le cœur du déterminisme. Fonctions principales :

- `_norm`, `_tokens`, `grounding_ratio` : normalisation et ancrage lexical ;
- `_check_citations` : existence de la source et présence littérale de la citation ;
- `validate_spec`, `validate_clarify`, `validate_plan`, `validate_tasks` : une par étape ;
- `_has_cycle`, `topological_order` : graphe des dépendances ;
- `syntax_errors`, `validate_implementation`, `validate_fix`, `is_test_file`, `fixable_files` : implémentation ;
- `analyze` : contrôle croisé.

Chaque validateur renvoie une **liste de messages en français**, vide si tout va bien. Ces messages servent à la
fois au LLM (pour se corriger) et à l'humain (dans les fichiers `.md`).

## 10.7 `render.py`

Transforme l'état en Markdown lisible (`spec_md`, `plan_md`, `tasks_md` avec matrice de traçabilité,
`analyze_md`) et écrit les fichiers (`write` crée les dossiers au besoin). Aucun LLM.

## 10.8 `graph.py`

- `SDDState` : l'état du graphe.
- `generate()` : la boucle générer → valider → corriger (chapitre 5.2).
- `_add_human_input()` : transforme réponses et retours en sources `A-xxx` / `F-xxx`.
- `_read_workspace()`, `run_tests()` : lecture du code produit et exécution de pytest dans un sous-processus.
- `build_graph(llm, out_dir, max_retries, checkpointer, max_fix_attempts)` : définit les nœuds (fonctions
  internes qui « capturent » `llm` et `out_dir`), les fonctions de routage, puis assemble et compile le
  `StateGraph`. Le LLM est passé en paramètre : c'est ce qui permet aux tests d'utiliser un faux modèle.

## 10.9 `cli.py`

Lit les fichiers, crée le checkpointer SQLite, construit le graphe et boucle : `invoke` → si
`__interrupt__`, afficher l'artefact, les erreurs, les avertissements et les questions, demander
`[a]pprouver / [r]éviser / [q]uitter`, recueillir réponses et retour, puis reprendre avec
`Command(resume=...)`. Option `--thread` pour reprendre une session.

## 10.10 `runner.py`

Le pont entre le graphe et l'interface web, sans dépendance à Streamlit (donc testable seul) :

- `advance(graph, inp, cfg, on_update)` appelle `graph.stream(inp, cfg, stream_mode="updates")`. Ce mode de
  streaming standard de LangGraph renvoie, nœud par nœud, la mise à jour d'état produite : l'interface peut
  ainsi afficher « specify terminé : 6 exigences » pendant que le graphe continue. À la fin, `graph.get_state(cfg)`
  donne le point de contrôle en attente (`snapshot.interrupts`), ou rien si le graphe est terminé.
- `resume(graph, cfg, review)` transmet la décision humaine avec `Command(resume=review)`, exactement comme la CLI.
- `STEPS` et `NODE_TO_STEP` rattachent chacun des 13 nœuds à l'une des 7 étapes de spec-kit (un test vérifie
  qu'aucun nœud n'est oublié).

## 10.11 `ui.py`

L'interface web locale, écrite avec **Streamlit** (une bibliothèque Python qui transforme un script en page web,
sans HTML ni JavaScript). Choix : Streamlit est le standard de fait pour les outils internes en Python, il
s'installe en option (`.[ui]`) et ne change rien au cœur du pipeline.

Fonctionnement : Streamlit ré-exécute le script à chaque clic ; ce qui doit survivre (identifiant de session,
journal, point de contrôle en attente) est rangé dans `st.session_state`. Le graphe est construit une fois par
session (`st.cache_resource`) avec le même checkpointer SQLite que la CLI : une session interrompue (quota,
fermeture du navigateur) se reprend avec son identifiant.

**Ce que l'interface ne fait pas** : elle n'appelle jamais le modèle elle-même et n'ajoute aucun texte. Elle
envoie le brief et la constitution tels quels, puis les décisions humaines (`approve`, `revise` avec réponses et
retour, `abort`), qui passent par le même modèle `Review` et deviennent des sources `A-xxx` / `F-xxx`. Tous les
garde-fous du chapitre 7 s'appliquent donc à l'identique.

# 11. Les tests

Les tests (`tests/test_pipeline.py`) tournent **sans réseau et sans clé** grâce à un `FakeLLM` : un faux chat
model dont `with_structured_output(schema)` renvoie un `RunnableLambda` qui sert des réponses préparées à
l'avance, schéma par schéma. Il s'insère dans la chaîne LCEL exactement comme un vrai modèle.

| Test | Ce qu'il prouve |
|---|---|
| `test_ingest_is_deterministic` | même texte, mêmes identifiants |
| `test_good_spec_passes` | une spec correcte passe |
| `test_hallucinated_spec_rejected` | « priorité, date d'échéance, OAuth » est rejeté, et l'oubli d'une source est signalé |
| `test_fake_quote_rejected` | une citation qui n'est pas dans la source est rejetée |
| `test_invented_tech_rejected` | « FastAPI » non sourcé est rejeté |
| `test_task_coverage_and_invented_criteria` | exigence non couverte et critère inventé détectés |
| `test_out_of_scope_file_rejected` | écrire hors des fichiers de la tâche est interdit |
| `test_escaped_newlines_rejected` | du code Python mal échappé est rejeté |
| `test_full_pipeline_with_retry_and_human_checkpoints` | le graphe complet : spec hallucinée corrigée au 2e essai, passage par chaque point de contrôle, ordre des tâches, fichiers écrits, matrice de traçabilité |
| `test_human_answers_become_citable_sources` | les questions de specify et clarify sont fusionnées et renumérotées ; une réponse devient la source `A-001` ; quitter termine le graphe |
| `test_verify_runs_tests_and_fix_cannot_touch_tests` | pytest est vraiment lancé ; un correctif qui modifie un test est rejeté ; le correctif légitime fait passer les tests |
| `test_require_tests_option` | avec `--require-tests`, une exigence fonctionnelle sans tâche de test est signalée |
| `test_fallback_model_used_when_primary_fails` | si le modèle principal échoue, le modèle de secours répond et la sortie est validée normalement |
| `test_constitution_rule_cannot_be_functional` | une règle de constitution classée `functional` est rejetée |
| `test_every_graph_node_maps_to_a_speckit_step` | chaque nœud du graphe est rattaché à une des 7 étapes affichées par l'UI |
| `test_runner_stops_at_each_checkpoint_and_reports_nodes` | `advance`/`resume` s'arrêtent à chaque point de contrôle et signalent chaque nœud |
| `test_streamlit_app_runs_full_workflow` | l'application Streamlit elle-même (via `AppTest`) : saisie du brief, 6 approbations, fichiers écrits |

Un exemple complet de sortie réelle est fourni dans `examples/sortie-exemple/`.

Commande : `pytest` (17 tests, environ 4 secondes).

# 12. Exécution réelle et enseignements

Le pipeline a été exécuté de bout en bout contre Groq (`openai/gpt-oss-120b`) sur l'exemple du gestionnaire de
tâches, en mode « tout approuver » automatique. Ces essais ont révélé plusieurs problèmes, tous corrigés.

<!-- RESULTATS_REELS -->

## 12.2 Problèmes rencontrés et corrections

**1. Modèle retiré.** `llama-3.3-70b-versatile` n'était plus servi par Groq. → La liste des modèles a été lue via
l'API et le défaut est passé à `openai/gpt-oss-120b`. Leçon : ne jamais figer un nom de modèle gratuit dans le
code métier.

**2. Réponse en texte libre au lieu de la structure.** Avec la méthode `function_calling`, lors d'une tâche
d'implémentation, le modèle a répondu par des questions en texte libre (« Pouvez-vous fournir la définition de la
classe Task… ? ») au lieu d'appeler l'outil ; l'API a renvoyé une erreur `tool_use_failed`. → Passage à
`method="json_schema"` (génération contrainte par le schéma) et ajout de la règle « tu ne poses jamais de question
en texte libre ».

**3. Tâches qui s'ignorent entre elles.** Chaque tâche ne voyait que ses propres fichiers ; une tâche réinventait
donc une classe déjà écrite par une tâche précédente. → Le prompt d'implémentation reçoit désormais tout le
contenu déjà écrit, avec la consigne de réutiliser les noms existants.

**4. Aucun test généré.** La constitution disait « les tests utilisent pytest », ce qui n'oblige pas à écrire des
tests ; fidèle à la règle « rien d'inventé », le modèle n'en a pas écrit. La vérification lançait alors pytest sur
zéro test (code de sortie 5) et le traitait comme un échec. → Le code 5 signifie maintenant « aucun test », et la
constitution d'exemple exige explicitement des tests. C'est le comportement voulu : **si vous voulez quelque
chose, écrivez-le dans les sources.**

**5. Code mal échappé.** Le modèle a produit un fichier de test dont les retours à la ligne étaient écrits `\n`
littéralement : fichier Python invalide, et comme les tests sont intouchables à l'étape `fix`, la correction
était impossible. → Contrôle syntaxique déterministe (`ast.parse`, `json.loads`, `tomllib`) dès l'étape
`implement`, avec un message d'aide sur l'échappement : l'erreur est attrapée et corrigée avant écriture.

**6. Appel LLM rejoué à la reprise.** La première version de `clarify` appelait le LLM puis `interrupt()` dans le
même nœud ; à la reprise, LangGraph ré-exécutait le nœud et rappelait le LLM. → Séparation en `clarify` (LLM)
puis `review_spec` (pause), conformément à la bonne pratique LangGraph.

**7. Quota journalier atteint.** Après plusieurs exécutions complètes, Groq a renvoyé une erreur 429 (limite de
tokens par jour). → Ajout d'un modèle de secours via `with_fallbacks`. Leçon : sur un tier gratuit, prévoir la
bascule dès le départ.

**8. Couverture des tests seulement nominale.** Avec la constitution « chaque fonctionnalité est couverte par au
moins un test pytest », le modèle a créé une tâche « configurer pytest » rattachée à cette exigence, sans écrire
de test. Formellement, l'exigence était couverte. → Option `--require-tests` et règle déterministe associée
(chapitre 6.5). Leçon : une règle que l'on veut garantie doit être **vérifiée par du code**, pas seulement
écrite dans un prompt.

# 13. Installation et utilisation

## 13.1 Installation

```bash
cd sdd-pipeline
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"         # ajouter ",ollama" ou ",gemini" si besoin
cp .env.example .env               # puis renseigner GROQ_API_KEY
```

**Version de Python** : le projet accepte Python 3.11 à 3.13 (`requires-python = ">=3.11,<3.14"`). Avec une
version plus récente (par exemple 3.15, installée par défaut par Homebrew sur certains Mac), une dépendance
indirecte (`zstandard`) doit être compilée et l'installation échoue. Sur macOS : `brew install python@3.13`, puis
créer l'environnement avec `python3.13` comme ci-dessus.

La clé Groq se crée gratuitement sur la console Groq. Le fichier `.env` est exclu de git par `.gitignore` : ne
jamais mettre la clé dans le code.

## 13.2 Lancer un projet

```bash
sdd examples/brief.md --constitution examples/constitution.md --out out --require-tests
```

Le programme affiche un identifiant de session, puis s'arrête à chaque point de contrôle :

```text
======================================================================
POINT DE CONTRÔLE : clarify
Artefact : out/spec.md
  ❓ Q-001 Comment les tâches doivent-elles être identifiées ... ?
[a]pprouver / [r]éviser / [q]uitter > r
Réponse à Q-001 (vide = passer) > par un numéro entier attribué à la création
Retour libre (vide = aucun) >
```

Ouvrez le fichier `.md` indiqué pour relire l'artefact avant de décider.

## 13.3 Utiliser l'interface web

```bash
sdd-ui          # ouvre http://localhost:8501 dans le navigateur
```

1. Collez votre brief (et, si vous le souhaitez, une constitution), ou cliquez sur « Charger l'exemple ».
2. Cliquez sur « Lancer le workflow SDD » : les nœuds s'affichent au fil de l'exécution, et la barre d'étapes
   montre où en est le workflow (constitution → … → implement).
3. À chaque point de contrôle, l'artefact (spec, plan, analyse, code proposé, résultat des tests) est affiché.
   Trois choix : **Approuver** ; **Réviser** en répondant aux questions ouvertes et/ou avec un retour libre ;
   **Arrêter**.
4. L'onglet « Fichiers générés » montre tous les artefacts et le code du workspace ; un bouton télécharge le tout
   en `.zip`. Les fichiers sont aussi sur disque dans `out/ui/<session>/`.

Si la clé API est absente du `.env`, la barre latérale permet de la saisir : elle reste en mémoire et n'est
jamais écrite sur disque. En cas d'erreur (quota, réseau), un bouton reprend le workflow au dernier checkpoint ;
une session peut aussi être reprise plus tard avec son identifiant.

## 13.4 Reprendre une session (CLI)

```bash
sdd examples/brief.md --out out --thread <identifiant>
```

## 13.5 Options

| Option / variable | Effet |
|---|---|
| `--out` | dossier de sortie |
| `--thread` | reprendre une session |
| `--max-retries` | nombre de corrections automatiques par étape (3) |
| `--require-tests` | chaque exigence fonctionnelle doit avoir une tâche qui écrit un test |
| `SDD_FALLBACK_MODEL` | modèle de secours (vide pour désactiver) |
| `SDD_PROVIDER` | `groq`, `ollama`, `gemini` |
| `SDD_MODEL` | nom du modèle |
| `SDD_STRUCTURED_METHOD` | `json_schema`, `function_calling`, `json_mode` |
| `SDD_UI_OUT` | dossier de sortie de l'interface web (`out/ui`) |

## 13.6 Écrire un bon brief

- Une idée par ligne : chaque ligne devient une source citable. Les titres (`#`) sont ignorés.
- Écrivez les critères d'acceptation si vous en avez (« le titre est obligatoire ») : sinon le pipeline n'en inventera pas.
- Mettez les choix techniques imposés dans la constitution : le plan ne peut retenir que ceux-là.

# 14. Limites et évolutions possibles

**Limites actuelles**

- L'ancrage lexical est une heuristique (voir 7.2) : la revue humaine reste nécessaire.
- Les **commentaires et docstrings** du code généré ne sont pas contrôlés : ils peuvent mentionner des outils
  absents (cas réel : `pytest-cov` et une CI dans un test bouche-trou). La revue `review_impl` est là pour ça.
- Les détails d'implémentation non spécifiés (noms de statuts, messages d'erreur) sont choisis par le modèle :
  pour les maîtriser, répondez aux questions de `clarify` ou écrivez-les dans le brief.
- Les titres Markdown du brief ne sont pas des sources.
- Les descriptions du plan et des tâches sont des choix de conception : elles sont rattachées à des exigences mais
  ne sont pas soumises à l'ancrage lexical (concevoir, c'est forcément ajouter du « comment »). C'est pour cela
  que le plan a un point de contrôle humain.
- Le code généré est exécuté (pytest) sur la machine : à faire tourner dans un environnement isolé (conteneur)
  pour un usage sérieux.
- Le tier gratuit de Groq a des quotas : un gros projet peut les atteindre.
- Température 0 ne garantit pas une sortie identique d'un appel à l'autre ; ce sont les règles d'acceptation qui sont déterministes.

**Évolutions possibles**

- Exécuter `verify` après chaque tâche et non seulement à la fin.
- Un mode « spec seulement » ou « jusqu'aux tâches » pour utiliser le pipeline comme générateur de documents.
- Un historique versionné des artefacts (un commit git par étape approuvée).
- Des validateurs spécifiques au domaine (vocabulaire métier autorisé, règles de nommage).

# 15. Glossaire

| Terme | Définition |
|---|---|
| **SDD** | Spec-Driven Development : développer à partir d'une spécification validée, étape par étape |
| **spec-kit** | boîte à outils de GitHub pour le SDD, dont ce projet reprend les étapes |
| **Constitution** | règles globales et non négociables du projet |
| **Brief** | description du besoin fonctionnel |
| **Source** | énoncé numéroté (C, B, A, F) qui sert de vérité citable |
| **Exigence (REQ)** | besoin extrait des sources, avec citations |
| **Hallucination** | information produite par un LLM sans fondement dans ses entrées |
| **Ancrage (grounding)** | part du contenu d'un énoncé présente dans ses sources |
| **Traçabilité** | capacité à relier chaque élément à son origine |
| **LLM** | Large Language Model, modèle de langage |
| **Chat model** | abstraction LangChain d'un LLM conversationnel |
| **LCEL** | LangChain Expression Language, composition avec `\|` |
| **Sortie structurée** | réponse du LLM conforme à un schéma (objet Pydantic) |
| **Pydantic** | bibliothèque Python de validation de données |
| **StateGraph** | graphe LangGraph à état partagé |
| **Nœud / arête** | étape du graphe / transition entre étapes |
| **Checkpointer** | composant LangGraph qui sauvegarde l'état à chaque étape |
| **Thread** | une exécution identifiée du graphe, reprenable |
| **interrupt** | pause du graphe en attente d'une décision humaine |
| **Human-in-the-loop** | humain intégré dans la boucle de décision |
| **Tri topologique** | ordre qui respecte les dépendances entre tâches |
| **Température** | paramètre de créativité d'un LLM (0 = le plus déterministe) |
| **Groq** | fournisseur d'inférence rapide avec API gratuite |

# Annexe : code source complet

<!-- ANNEXE_CODE -->
