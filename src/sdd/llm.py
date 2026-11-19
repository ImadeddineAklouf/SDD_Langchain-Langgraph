"""Fabrique de modèle. Température 0 partout, fournisseur choisi par variable d'env.

SDD_PROVIDER = ollama (défaut, 100 % local) | groq | gemini
SDD_MODEL    = nom du modèle (défaut selon le fournisseur)
SDD_FALLBACK_MODEL = modèle de secours si le principal échoue (quota...)
SDD_NUM_CTX  = taille du contexte Ollama en tokens (défaut 16384)
SDD_STRUCTURED_METHOD = json_schema (défaut) | function_calling | json_mode
OLLAMA_HOST  = adresse du serveur Ollama (défaut http://localhost:11434)
"""
from __future__ import annotations

import os

PROVIDERS = ["ollama", "groq", "gemini"]
DEFAULTS = {
    "ollama": "qwen3:14b",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
}
FALLBACKS = {"groq": "openai/gpt-oss-20b"}
API_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY"}


def provider() -> str:
    return os.getenv("SDD_PROVIDER", "ollama").lower()


def model_name() -> str:
    return os.getenv("SDD_MODEL") or DEFAULTS.get(provider(), "")


def fallback_name() -> str:
    return os.getenv("SDD_FALLBACK_MODEL", FALLBACKS.get(provider(), ""))


def get_fallback_llm():
    """Modèle de secours (quota atteint, panne). SDD_FALLBACK_MODEL="" pour désactiver."""
    model = fallback_name()
    return get_llm(model) if model else None


def get_llm(model: str | None = None):
    p = provider()
    model = model or model_name()
    if p == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx : le contexte par défaut d'Ollama est court ; au-delà, l'entrée serait tronquée sans erreur.
        return ChatOllama(model=model, temperature=0, seed=42, num_ctx=int(os.getenv("SDD_NUM_CTX", "16384")))
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0, max_retries=2, model_kwargs={"seed": 42})
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0)
    raise ValueError(f"Fournisseur inconnu : {p}")


def ollama_models() -> list[str] | None:
    """Modèles téléchargés sur le serveur Ollama local, ou None s'il ne répond pas."""
    try:
        import ollama

        return sorted(m.model for m in ollama.Client(host=os.getenv("OLLAMA_HOST")).list().models)
    except Exception:
        return None


def structured_method() -> str:
    """Méthode de sortie structurée. json_schema = le fournisseur contraint la génération
    au schéma JSON (le modèle ne peut pas répondre en texte libre). Avec Ollama, le schéma est
    passé dans le paramètre `format` : la génération est contrainte token par token."""
    return os.getenv("SDD_STRUCTURED_METHOD", "json_schema")
