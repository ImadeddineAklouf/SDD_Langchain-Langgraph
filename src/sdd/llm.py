"""Fabrique de modèle. Température 0 partout, fournisseur choisi par variable d'env.

SDD_PROVIDER = groq (défaut) | ollama | gemini
SDD_MODEL    = nom du modèle (défaut selon le fournisseur)
SDD_FALLBACK_MODEL = modèle de secours si le principal échoue (quota...)
SDD_STRUCTURED_METHOD = json_schema (défaut) | function_calling | json_mode
"""
from __future__ import annotations

import os

DEFAULTS = {
    "groq": "openai/gpt-oss-120b",
    "ollama": "qwen2.5:14b",
    "gemini": "gemini-2.5-flash",
}


FALLBACKS = {"groq": "openai/gpt-oss-20b"}


def get_fallback_llm():
    """Modèle de secours (quota atteint, panne). SDD_FALLBACK_MODEL="" pour désactiver."""
    provider = os.getenv("SDD_PROVIDER", "groq").lower()
    model = os.getenv("SDD_FALLBACK_MODEL", FALLBACKS.get(provider, ""))
    return get_llm(model) if model else None


def get_llm(model: str | None = None):
    provider = os.getenv("SDD_PROVIDER", "groq").lower()
    model = model or os.getenv("SDD_MODEL", DEFAULTS.get(provider, ""))
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0, max_retries=2, model_kwargs={"seed": 42})
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0, seed=42)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0)
    raise ValueError(f"Fournisseur inconnu : {provider}")


def structured_method() -> str:
    """Méthode de sortie structurée. json_schema = le fournisseur contraint la génération
    au schéma JSON (le modèle ne peut pas répondre en texte libre)."""
    return os.getenv("SDD_STRUCTURED_METHOD", "json_schema")
