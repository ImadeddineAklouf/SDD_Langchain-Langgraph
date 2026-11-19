"""Fabrique de modèle : Gemini par défaut, autres fournisseurs par variable d'env (sans réseau)."""
import pytest

from sdd import llm
from sdd.models import Spec


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ("SDD_PROVIDER", "SDD_MODEL", "SDD_FALLBACK_MODEL", "SDD_NUM_CTX"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")


def test_gemini_is_default_provider_with_fallback():
    from langchain_google_genai import ChatGoogleGenerativeAI

    m = llm.get_llm()
    assert isinstance(m, ChatGoogleGenerativeAI)
    assert (m.model, m.temperature, m.seed) == ("gemini-2.5-flash", 0, 42)
    assert llm.get_fallback_llm().model == "gemini-2.5-flash-lite"  # même clé, quota séparé


def test_gemini_structured_output_constrains_generation_to_schema():
    chain = llm.get_llm().with_structured_output(Spec, method=llm.structured_method())
    kwargs = chain.first.kwargs  # liaison du modèle, avant le parseur
    assert kwargs["response_mime_type"] == "application/json"
    assert kwargs["response_json_schema"]["properties"].keys() == Spec.model_json_schema()["properties"].keys()


def test_groq_still_selectable(monkeypatch):
    from langchain_groq import ChatGroq

    monkeypatch.setenv("SDD_PROVIDER", "groq")
    assert isinstance(llm.get_llm(), ChatGroq)
    assert llm.model_name() == "openai/gpt-oss-120b" and llm.fallback_name() == "openai/gpt-oss-20b"


def test_ollama_optional(monkeypatch):
    pytest.importorskip("langchain_ollama")
    monkeypatch.setenv("SDD_PROVIDER", "ollama")
    monkeypatch.setenv("SDD_NUM_CTX", "32768")
    m = llm.get_llm()
    assert (m.model, m.num_ctx) == ("qwen3:14b", 32768)
    assert llm.get_fallback_llm() is None


def test_ollama_models_returns_none_when_server_down(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    assert llm.ollama_models() is None
