"""Fabrique de modèle. Température 0 partout, fournisseur choisi par variable d'env.

SDD_PROVIDER = gemini (défaut, API Google gratuite) | groq | ollama (local, optionnel)
SDD_MODEL    = nom du modèle (défaut selon le fournisseur)
SDD_FALLBACK_MODEL = modèle de secours si le principal échoue (quota...)
SDD_FALLBACK_PROVIDER = fournisseur du secours (défaut : l'autre fournisseur cloud si sa clé est présente)
SDD_GUARDIAN_MODEL = modèle du Gardien (contrôle sémantique), idéalement différent du producteur
SDD_GUARDIAN_SEMANTIC = 1 (défaut) | 0 : active le contrôle sémantique par LLM du Gardien
SDD_TIMEOUT  = délai maximal d'un appel au modèle, en secondes (défaut 120) : jamais d'attente infinie
SDD_THINKING = niveau de réflexion des modèles Gemini 3 : minimal | low (défaut) | medium | high
SDD_NUM_CTX  = taille du contexte Ollama en tokens (défaut 16384)
SDD_STRUCTURED_METHOD = json_schema (défaut) | function_calling | json_mode
OLLAMA_HOST  = adresse du serveur Ollama (défaut http://localhost:11434)
"""
from __future__ import annotations

import logging
import os
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
        return ChatGoogleGenerativeAI(model=model, temperature=0, seed=42, timeout=llm_timeout(), max_retries=3, **extra)
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0, max_retries=2, timeout=llm_timeout(), model_kwargs={"seed": 42})
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

    if isinstance(exc, ModelError):
        return True
    if _is_network_error(exc):
        return True
    return getattr(exc, "status_code", None) in (401, 403, 404, 429)


def explain_error(exc: Exception) -> str:
    """Message clair (en français) pour une erreur du fournisseur."""
    from langchain_core.exceptions import (
        ModelAuthenticationError, ModelConnectionError, ModelNotFoundError, ModelPermissionDeniedError,
        ModelRateLimitError, ModelTimeoutError,
    )

    detail = str(exc)[:400]
    status = getattr(exc, "status_code", None)
    if isinstance(exc, ModelNotFoundError) or status == 404:
        return (f"Modèle introuvable pour cette clé ({model_name()}). Choisissez un autre modèle dans la barre "
                f"latérale (ou SDD_MODEL dans .env). Détail : {detail}")
    if isinstance(exc, (ModelAuthenticationError, ModelPermissionDeniedError)) or status in (401, 403):
        key = API_KEYS.get(provider(), "la clé API")
        return f"Clé API refusée : vérifiez {key} dans le fichier .env. Détail : {detail}"
    if isinstance(exc, ModelRateLimitError) or status == 429:
        return ("Quota gratuit du fournisseur atteint (modèle de secours compris). Attendez un peu, ou choisissez "
                f"un autre modèle ou fournisseur dans la barre latérale. Détail : {detail}")
    if isinstance(exc, ModelTimeoutError) or any("Timeout" in c.__name__ for c in type(exc).__mro__):
        return (f"Le modèle {model_name()} n'a pas répondu dans le délai de {llm_timeout():.0f} s (essais compris). "
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


def generate(llm, schema, system: str, payload: str, validate, max_retries: int, fallback_llm=None):
    """Appel LLM structuré (LCEL : prompt | modèle structuré) + validation déterministe + correction bornée.

    Ne lève pas si l'objet reste invalide : renvoie le dernier objet et ses erreurs,
    que le Gardien transformera en alertes.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{input}"), MessagesPlaceholder("corrections", optional=True)]
    )
    model = llm.with_structured_output(schema, method=structured_method())
    if fallback_llm is not None:  # mécanisme standard LangChain : bascule si le modèle principal échoue
        model = model.with_fallbacks([fallback_llm.with_structured_output(schema, method=structured_method())])
    chain = prompt | model  # LCEL
    corrections: list = []
    obj, errors = None, ["Aucune sortie."]
    step = schema.__name__
    for attempt in range(1, max_retries + 2):
        _progress(f"{step} : appel au modèle (essai {attempt}/{max_retries + 1})…")
        t0 = time.monotonic()
        try:
            out = chain.invoke({"input": payload, "corrections": corrections})
            obj = out if isinstance(out, schema) else schema.model_validate(out)
            errors = validate(obj)
        except Exception as exc:
            if is_api_error(exc):  # clé, modèle introuvable, réseau, délai, quota : inutile de redemander
                _progress(f"{step} : erreur du fournisseur après {time.monotonic() - t0:.0f} s : {type(exc).__name__}")
                raise
            errors = [f"Sortie non conforme au schéma : {str(exc)[:500]}"]
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
