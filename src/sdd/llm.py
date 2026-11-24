"""Fabrique de modèle. Température 0 partout, fournisseur choisi par variable d'env.

SDD_PROVIDER = gemini (défaut, API Google gratuite) | groq | ollama (local, optionnel)
SDD_MODEL    = nom du modèle (défaut selon le fournisseur)
SDD_FALLBACK_MODEL = modèle de secours si le principal échoue (quota...)
SDD_FALLBACK_PROVIDER = fournisseur du secours (défaut : l'autre fournisseur cloud si sa clé est présente)
SDD_GUARDIAN_MODEL = modèle du Gardien (contrôle sémantique), idéalement différent du producteur
SDD_GUARDIAN_SEMANTIC = 1 (défaut) | 0 : active le contrôle sémantique par LLM du Gardien
SDD_CORRECTOR = 1 (défaut) | 0 : Correcteur (3e agent) qui corrige par patchs ce que le Gardien refuse
SDD_CORRECTOR_MODEL = modèle du Correcteur (défaut : modèle principal) ; SDD_CORRECTOR_ROUNDS = tours (2)
SDD_TIMEOUT  = délai maximal d'un appel au modèle, essais du client compris, en secondes (défaut 120) :
               garanti par une horloge côté Python, jamais d'attente infinie
SDD_THINKING = niveau de réflexion des modèles Gemini 3 : minimal | low (défaut) | medium | high
SDD_NUM_CTX  = taille du contexte Ollama en tokens (défaut 16384)
SDD_STRUCTURED_METHOD = json_schema (défaut) | function_calling | json_mode
OLLAMA_HOST  = adresse du serveur Ollama (défaut http://localhost:11434)
"""
from __future__ import annotations

import contextvars
import logging
import os
import re
import threading
import time

log = logging.getLogger("sdd")

PROVIDERS = ["gemini", "groq", "ollama"]
DEFAULTS = {
    "gemini": "gemini-3.8-flash",  # modèle recommandé par l'API (gemini-2.5-flash est fermé aux nouveaux comptes)
    "groq": "openai/gpt-oss-120b",
    "ollama": "qwen3:14b",
}
# Secours chez le même fournisseur (même clé, quota séparé) : bascule automatique si le principal échoue.
# Gemini : aucun par défaut, car les modèles ouverts dépendent du compte ; à choisir dans l'UI ou le .env.
FALLBACKS = {"groq": "openai/gpt-oss-20b"}
API_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY"}


def provider() -> str:
    return os.getenv("SDD_PROVIDER", "gemini").lower()


def model_name() -> str:
    return os.getenv("SDD_MODEL") or DEFAULTS.get(provider(), "")


def fallback_provider() -> str:
    """Fournisseur du modèle de secours.

    SDD_FALLBACK_PROVIDER s'il est défini ; sinon, si le fournisseur n'a pas de secours interne (Gemini),
    qu'aucun secours n'est fixé et que la clé de l'autre fournisseur cloud est présente, on bascule chez lui : une panne, une lenteur ou un quota
    épuisé chez l'un ne bloque plus le workflow."""
    explicit = os.getenv("SDD_FALLBACK_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    p = provider()
    if "SDD_FALLBACK_MODEL" not in os.environ and not FALLBACKS.get(p):
        other = {"gemini": "groq", "groq": "gemini"}.get(p)
        if other and os.getenv(API_KEYS[other]):
            return other
    return p


def fallback_name() -> str:
    fp = fallback_provider()
    if fp != provider():
        return os.getenv("SDD_FALLBACK_MODEL") or DEFAULTS.get(fp, "")
    return os.getenv("SDD_FALLBACK_MODEL", FALLBACKS.get(provider(), ""))


def fallback_label() -> str:
    """« modèle » ou « fournisseur/modèle » si le secours est chez un autre fournisseur ; "" si aucun."""
    name = fallback_name()
    if not name:
        return ""
    return name if fallback_provider() == provider() else f"{fallback_provider()}/{name}"


def get_fallback_llm():
    """Modèle de secours (quota atteint, panne, délai dépassé). SDD_FALLBACK_MODEL="" pour désactiver."""
    model = fallback_name()
    return get_llm(model, fallback_provider()) if model else None


def guardian_name() -> str:
    """Modèle du Gardien : SDD_GUARDIAN_MODEL, sinon le modèle principal."""
    return os.getenv("SDD_GUARDIAN_MODEL") or model_name()


def guardian_semantic_enabled() -> bool:
    return os.getenv("SDD_GUARDIAN_SEMANTIC", "1").strip().lower() not in ("0", "false", "non", "no", "")


def get_guardian_llm():
    """LLM du Gardien, ou None si le contrôle sémantique est désactivé (contrôles déterministes seuls)."""
    return get_llm(guardian_name()) if guardian_semantic_enabled() else None


def corrector_name() -> str:
    """Modèle du Correcteur : SDD_CORRECTOR_MODEL, sinon le modèle principal."""
    return os.getenv("SDD_CORRECTOR_MODEL") or model_name()


def corrector_enabled() -> bool:
    return os.getenv("SDD_CORRECTOR", "1").strip().lower() not in ("0", "false", "non", "no", "")


def corrector_rounds() -> int:
    return int(os.getenv("SDD_CORRECTOR_ROUNDS", "2"))


def get_corrector_llm():
    """LLM du Correcteur, ou None s'il est désactivé (toute alerte va alors directement à l'humain)."""
    return get_llm(corrector_name()) if corrector_enabled() else None


def llm_timeout() -> float:
    return float(os.getenv("SDD_TIMEOUT", "120"))


def thinking_level() -> str:
    return os.getenv("SDD_THINKING", "low").strip().lower()


def get_llm(model: str | None = None, prov: str | None = None):
    p = prov or provider()
    model = model or (model_name() if p == provider() else DEFAULTS.get(p, ""))
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # timeout : un appel ne peut jamais bloquer indéfiniment (l'erreur s'affiche au lieu d'un écran figé).
        # max_retries : nombre total d'essais du client (429 / 503 / délai dépassé), avec attente croissante.
        # thinking_level : les modèles Gemini 3 « réfléchissent » longuement par défaut ; pour une tâche de
        # transformation contrainte par un schéma, une réflexion courte suffit et divise le temps de réponse.
        extra = {"thinking_config": {"thinking_level": thinking_level()}} if model.startswith("gemini-3") else {}
        return ChatGoogleGenerativeAI(model=model, temperature=0, seed=42, timeout=llm_timeout(), max_retries=2, **extra)
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0, max_retries=1, timeout=llm_timeout(), model_kwargs={"seed": 42})
    if p == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx : le contexte par défaut d'Ollama est court ; au-delà, l'entrée serait tronquée sans erreur.
        return ChatOllama(model=model, temperature=0, seed=42, num_ctx=int(os.getenv("SDD_NUM_CTX", "16384")))
    raise ValueError(f"Fournisseur inconnu : {p}")


def gemini_models() -> list[str] | None:
    """Modèles Gemini utilisables avec la clé (generateContent), ou None si la liste est inaccessible."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        from google import genai

        models = genai.Client(api_key=key).models.list()
        return sorted(m.name.removeprefix("models/") for m in models if "generateContent" in (m.supported_actions or []))
    except Exception:
        return None


_NETWORK = {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout",
            "WriteTimeout", "PoolTimeout", "TimeoutException", "TimeoutError", "ModelTimeoutError"}


def _is_network_error(exc: Exception) -> bool:
    """Délai dépassé ou connexion impossible (httpx, httpx2, openai…), reconnu par le nom de classe."""
    return any(c.__name__ in _NETWORK for c in type(exc).__mro__)


def is_api_error(exc: Exception) -> bool:
    """Erreur du fournisseur (clé, modèle introuvable, réseau, quota épuisé) : redemander ne sert à rien.

    Les exceptions standard de langchain-core (ModelError) couvrent Gemini ; les codes HTTP couvrent les autres.
    Une sortie qui ne respecte pas le schéma (OutputParserException, ValidationError, ou un 400 de Groq)
    n'en fait pas partie : elle est corrigée par la boucle de génération."""
    from langchain_core.exceptions import ModelError

    if isinstance(exc, (ModelError, AllModelsFailed)):
        return True
    if _is_network_error(exc) or is_request_too_large(exc):
        return True
    return getattr(exc, "status_code", None) in (401, 403, 404, 413, 429)


_TOO_LARGE = re.compile(r"request too large|reduce your message size|request entity too large|payload too large", re.I)
_LIMIT_REQUESTED = re.compile(r"Limit\W+(\d+)\W+Requested\W+(\d+)", re.I)


def is_request_too_large(exc: Exception) -> bool:
    """La requête dépasse ce que le fournisseur accepte EN UNE FOIS (HTTP 413, ou 429 dont la taille demandée
    excède la limite par minute, comme Groq gratuit : « Limit 8000, Requested 8604 »).

    Ni attendre, ni redemander une correction (qui ALLONGE la requête) ne peut réussir : il faut un autre
    fournisseur. Avant, ce 413 était pris pour une « sortie non conforme au schéma » et redemandé en boucle."""
    text = str(exc)
    if getattr(exc, "status_code", None) == 413 or _TOO_LARGE.search(text):
        return True
    m = _LIMIT_REQUESTED.search(text)
    return bool(m) and int(m.group(2)) > int(m.group(1))


def _provider_of(llm) -> str | None:
    """Fournisseur d'un modèle LangChain, d'après sa classe (ChatGroq, ChatGoogleGenerativeAI, ChatOllama)."""
    name = type(llm).__name__.lower()
    for p, key in (("groq", "groq"), ("gemini", "google"), ("ollama", "ollama")):
        if key in name:
            return p
    return None


def rescue_llm(failed_provider: str | None):
    """Modèle de dernier recours quand la requête est trop grosse pour le fournisseur en cause : Gemini (fenêtre
    et quota par minute bien plus larges) si sa clé est présente et qu'il n'est pas lui-même en cause.
    Désactivé si l'utilisateur a choisi « (aucun) » comme secours (SDD_FALLBACK_MODEL vide)."""
    if failed_provider == "gemini" or not os.getenv(API_KEYS["gemini"]):
        return None
    if "SDD_FALLBACK_MODEL" in os.environ and not os.environ["SDD_FALLBACK_MODEL"].strip():
        return None
    return get_llm(DEFAULTS["gemini"], "gemini")


def explain_error(exc: Exception) -> str:
    """Message clair (en français) pour une erreur du fournisseur."""
    from langchain_core.exceptions import (
        ModelAuthenticationError, ModelConnectionError, ModelNotFoundError, ModelPermissionDeniedError,
        ModelRateLimitError, ModelTimeoutError,
    )

    if isinstance(exc, AllModelsFailed):
        return ("Aucun modèle n'a pu répondre :\n\n" + "\n\n".join(
            f"- **{label}** : {explain_error(e)}" for label, e in exc.failures)
            + "\n\nSi un seul fournisseur est en cause, choisissez « (aucun) » ou un autre modèle comme modèle de "
              "secours dans la barre latérale.")
    detail = str(exc)[:400]
    status = getattr(exc, "status_code", None)
    if is_request_too_large(exc):
        m = _LIMIT_REQUESTED.search(str(exc))
        size = (f" : {m.group(2)} tokens demandés pour une limite de {m.group(1)} tokens par minute" if m else "")
        who = "Groq" if "groq" in type(exc).__module__.lower() or "gpt-oss" in str(exc) else "du fournisseur"
        return (f"Requête trop volumineuse pour l'offre gratuite {who}{size}. Attendre ou réessayer ne changera "
                "rien. Choisissez « gemini » comme fournisseur dans la barre latérale (limites bien plus larges), "
                "ou ajoutez GOOGLE_API_KEY dans .env pour que la bascule vers Gemini se fasse toute seule, puis "
                f"cliquez sur « Reprendre ». Détail : {detail}")
    if isinstance(exc, ModelNotFoundError) or status == 404:
        return (f"Modèle introuvable pour cette clé. Choisissez un autre modèle dans la barre "
                f"latérale (ou SDD_MODEL dans .env). Détail : {detail}")
    if isinstance(exc, (ModelAuthenticationError, ModelPermissionDeniedError)) or status in (401, 403):
        key = API_KEYS.get(provider(), "la clé API")
        return f"Clé API refusée : vérifiez {key} dans le fichier .env. Détail : {detail}"
    if isinstance(exc, ModelRateLimitError) or status == 429:
        if rate_limit_wait(exc) is not None:
            return ("Limite par minute du fournisseur toujours atteinte après plusieurs attentes. Cliquez sur "
                    "« Reprendre » dans une minute, ou choisissez un autre fournisseur dans la barre latérale. "
                    f"Détail : {detail}")
        if _DAILY.search(str(exc)):
            return ("Quota JOURNALIER gratuit du fournisseur épuisé. Choisissez un autre "
                    f"fournisseur ou modèle dans la barre latérale, ou réessayez demain. Détail : {detail}")
        return ("Quota gratuit du fournisseur atteint. Attendez un peu, ou choisissez "
                f"un autre modèle ou fournisseur dans la barre latérale. Détail : {detail}")
    if isinstance(exc, ModelTimeoutError) or any("Timeout" in c.__name__ for c in type(exc).__mro__):
        return (f"Le modèle n'a pas répondu dans le délai de {llm_timeout():.0f} s. "
                "Le service est peut-être surchargé : réessayez, choisissez un modèle plus léger, ou augmentez "
                f"SDD_TIMEOUT dans .env. Détail : {detail}")
    if isinstance(exc, ModelConnectionError) or "Connect" in type(exc).__name__:
        return f"Connexion au fournisseur impossible : vérifiez votre connexion Internet. Détail : {detail}"
    return f"Erreur du fournisseur : {detail}"


def ollama_models() -> list[str] | None:
    """Modèles téléchargés sur le serveur Ollama local, ou None s'il ne répond pas."""
    try:
        import ollama

        return sorted(m.model for m in ollama.Client(host=os.getenv("OLLAMA_HOST")).list().models)
    except Exception:
        return None


def structured_method() -> str:
    """Méthode de sortie structurée. json_schema = le fournisseur contraint la génération
    au schéma JSON (le modèle ne peut pas répondre en texte libre). Gemini reçoit le schéma dans
    `response_json_schema`, Ollama dans `format` : dans les deux cas la génération est contrainte."""
    return os.getenv("SDD_STRUCTURED_METHOD", "json_schema")


_WAIT = re.compile(r"(?:try again|retry) in\s*([\d.]+)\s*(ms|s|m)\b|retry[_ ]?delay\W+([\d.]+)\s*s", re.I)
_DAILY = re.compile(r"per day|\(TPD\)|\(RPD\)|PerDay|daily", re.I)
MAX_RATE_WAIT = 90.0  # au-delà, ce n'est plus une limite « par minute »


def rate_limit_wait(exc: Exception) -> float | None:
    """Secondes à attendre si l'erreur est une limite PAR MINUTE (429 « try again in 4.68s »), sinon None.

    Une limite journalière (tokens/requêtes par jour) n'est pas une attente courte : c'est un quota épuisé.
    Le délai vient de l'en-tête retry-after s'il existe, sinon du message du fournisseur (Groq, Gemini)."""
    from langchain_core.exceptions import ModelRateLimitError

    text = str(exc)
    if not (isinstance(exc, ModelRateLimitError) or getattr(exc, "status_code", None) == 429
            or "429" in text or "RESOURCE_EXHAUSTED" in text):
        return None
    if _DAILY.search(text) or is_request_too_large(exc):  # quota du jour, ou requête plus grosse que la limite
        return None
    wait = None
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        wait = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        m = _WAIT.search(text)
        if m:
            value, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), "s")
            wait = float(value) * {"ms": 0.001, "s": 1, "m": 60}[unit.lower()]
    if wait is None:  # 429 sans délai indiqué : traité comme un quota épuisé (bascule sur le secours)
        return None
    return wait + 1.0 if wait <= MAX_RATE_WAIT else None


class AllModelsFailed(Exception):
    """Le modèle principal ET le secours ont échoué : on garde les DEUX erreurs (avant, seule celle du secours
    s'affichait, ce qui faisait croire que le fournisseur choisi n'était pas utilisé)."""

    def __init__(self, failures: list[tuple[str, Exception]]):
        self.failures = failures
        super().__init__(" | ".join(f"{label} : {type(e).__name__} : {str(e)[:300]}" for label, e in failures))


def _model_label(llm) -> str:
    name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__
    return str(name).removeprefix("models/")


class CallTimeout(TimeoutError):
    """Le modèle n'a pas rendu la main dans le délai SDD_TIMEOUT (horloge côté Python)."""


def invoke_with_deadline(runnable, inp, seconds: float):
    """Appelle runnable.invoke(inp) avec un délai DUR.

    Le `timeout` des clients HTTP ne suffit pas : il s'applique à chaque essai (et le client en refait
    plusieurs), et il est remis à zéro tant que le serveur envoie quelques octets. Ici, l'appel tourne dans un
    fil séparé (avec le contexte LangGraph/LangChain copié) et l'on n'attend jamais plus de `seconds` secondes.
    Si le délai est dépassé, l'appel est abandonné (le fil se termine seul en arrière-plan) et CallTimeout est
    levée : l'étape s'arrête avec un message clair, ou bascule sur le modèle de secours."""
    ctx = contextvars.copy_context()
    box: dict = {}

    def run() -> None:
        try:
            box["out"] = ctx.run(runnable.invoke, inp)
        except BaseException as exc:  # transmis au fil principal
            box["err"] = exc

    # fil « démon » : s'il reste bloqué, il n'empêche ni la suite du workflow ni la fermeture du programme
    worker = threading.Thread(target=run, name="sdd-llm", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise CallTimeout(f"aucune réponse du modèle en {seconds:.0f} s (délai SDD_TIMEOUT)")
    if "err" in box:
        raise box["err"]
    return box["out"]


def generate(llm, schema, system: str, payload: str, validate, max_retries: int, fallback_llm=None):
    """Appel LLM structuré (LCEL : prompt | modèle structuré) + validation déterministe + correction bornée.

    Chaque appel a un délai dur (SDD_TIMEOUT). Si le modèle principal échoue (délai, panne, quota, clé),
    on bascule sur le modèle de secours, et l'on y reste pour le reste de l'étape.
    Ne lève pas si l'objet reste invalide : renvoie le dernier objet et ses erreurs,
    que le Gardien transformera en alertes.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{input}"), MessagesPlaceholder("corrections", optional=True)]
    )
    def chain_of(role: str, model) -> tuple[str, object, str | None]:
        return (f"{role} ({_model_label(model)})",
                prompt | model.with_structured_output(schema, method=structured_method()),  # LCEL
                _provider_of(model))

    chains = [chain_of("principal", llm)]
    if fallback_llm is not None:
        chains.append(chain_of("secours", fallback_llm))
    failures: list[tuple[str, Exception]] = []
    corrections: list = []
    obj, errors = None, ["Aucune sortie."]
    step = schema.__name__
    deadline = llm_timeout()
    waits = 0
    for attempt in range(1, max_retries + 2):
        answered = False
        while not answered:
            label, chain, prov = chains[0]
            _progress(f"{step} : appel au modèle {label} (essai {attempt}/{max_retries + 1}, "
                      f"délai max {deadline:.0f} s)…")
            t0 = time.monotonic()
            try:
                out = invoke_with_deadline(chain, {"input": payload, "corrections": corrections}, deadline)
                answered = True
                obj = out if isinstance(out, schema) else schema.model_validate(out)
                errors = validate(obj)
            except Exception as exc:
                wait = rate_limit_wait(exc)
                if wait is not None and waits < 6:  # limite par minute : on patiente puis on réessaie
                    waits += 1
                    _progress(f"{step} : limite par minute du modèle {label} atteinte, nouvel essai dans "
                              f"{wait:.0f} s ({waits}/6)…")
                    time.sleep(wait)
                    continue
                if is_api_error(exc):  # clé, modèle introuvable, réseau, délai, quota : inutile de redemander
                    _progress(f"{step} : échec du modèle {label} après {time.monotonic() - t0:.0f} s : "
                              f"{type(exc).__name__} {str(exc)[:120]}")
                    failed = chains.pop(0)
                    if is_request_too_large(exc) and prov:
                        # même fournisseur = même limite de taille : inutile d'essayer son autre modèle
                        chains = [c for c in chains if c[2] != prov]
                        if not chains and (rescue := rescue_llm(prov)) is not None:
                            chains.append(chain_of("secours", rescue))
                    if chains:
                        failures.append((label, exc))
                        _progress(f"{step} : bascule sur le modèle {chains[0][0]}")
                        continue
                    chains.insert(0, failed)
                    if failures:
                        raise AllModelsFailed(failures + [(label, exc)]) from exc
                    raise
                errors = [f"Sortie non conforme au schéma : {str(exc)[:500]}"]
                answered = True  # sortie reçue mais invalide : on passe à la correction
        _progress(f"{step} : réponse en {time.monotonic() - t0:.0f} s, "
                  + ("valide" if not errors else f"{len(errors)} erreur(s) de validation, correction demandée"))
        if not errors:
            return obj, []
        if obj is not None:
            corrections.append(AIMessage(obj.model_dump_json()))
        corrections.append(
            HumanMessage(
                "Ta réponse est rejetée par le validateur :\n- "
                + "\n- ".join(errors)
                + "\nCorrige UNIQUEMENT ces points, sans rien ajouter d'autre."
            )
        )
    if obj is None:
        raise RuntimeError("Le modèle n'a produit aucune sortie valide : " + "; ".join(errors))
    return obj, errors


def _progress(message: str) -> None:
    """Trace chaque appel au modèle : dans le terminal (logging) et, en direct, dans l'interface
    (flux « custom » standard de LangGraph, via get_stream_writer)."""
    log.info(message)
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"progress": message})
    except Exception:  # hors d'un graphe (tests unitaires, appel direct) : seulement le journal
        pass


def setup_logging() -> None:
    """Affiche les traces du pipeline dans le terminal (sdd, sdd-ui)."""
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [sdd] %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
