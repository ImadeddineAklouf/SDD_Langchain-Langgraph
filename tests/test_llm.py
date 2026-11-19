"""Fabrique de modèle : Ollama par défaut, autres fournisseurs par variable d'env (sans réseau)."""
import pytest

from sdd import llm
from sdd.models import Spec


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ("SDD_PROVIDER", "SDD_MODEL", "SDD_FALLBACK_MODEL", "SDD_NUM_CTX"):
        monkeypatch.delenv(v, raising=False)


def test_ollama_is_default_local_provider():
    from langchain_ollama import ChatOllama

    m = llm.get_llm()
    assert isinstance(m, ChatOllama)
    assert (m.model, m.temperature, m.seed, m.num_ctx) == ("qwen3:14b", 0, 42, 16384)
    assert llm.get_fallback_llm() is None  # pas de quota en local : pas de secours


def test_ollama_structured_output_constrains_format_to_schema():
    chain = llm.get_llm().with_structured_output(Spec, method=llm.structured_method())
    bound = chain.first  # RunnableBinding du modèle, avant le parseur
    assert bound.kwargs["format"] == Spec.model_json_schema()


def test_provider_and_model_from_env(monkeypatch):
    from langchain_groq import ChatGroq

    monkeypatch.setenv("SDD_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    assert isinstance(llm.get_llm(), ChatGroq)
    assert llm.model_name() == "openai/gpt-oss-120b" and llm.fallback_name() == "openai/gpt-oss-20b"
    monkeypatch.setenv("SDD_PROVIDER", "ollama")
    monkeypatch.setenv("SDD_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("SDD_NUM_CTX", "32768")
    m = llm.get_llm()
    assert (m.model, m.num_ctx) == ("gpt-oss:20b", 32768)


def test_ollama_models_returns_none_when_server_down(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    assert llm.ollama_models() is None
