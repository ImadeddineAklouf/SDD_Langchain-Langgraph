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


def test_gemini_is_default_provider(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    m = llm.get_llm()
    assert isinstance(m, ChatGoogleGenerativeAI)
    assert (m.model, m.temperature, m.seed) == ("gemini-3.8-flash", 0, 42)
    assert llm.get_fallback_llm() is None  # modèles ouverts variables selon le compte : secours au choix
    monkeypatch.setenv("SDD_FALLBACK_MODEL", "gemini-3.8-flash-lite")
    assert llm.get_fallback_llm().model == "gemini-3.8-flash-lite"


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


def test_api_errors_are_explained_not_reported_as_schema_errors():
    from langchain_core.exceptions import ModelNotFoundError, ModelRateLimitError, OutputParserException
    from pydantic import ValidationError

    nf = ModelNotFoundError("404 NOT_FOUND models/gemini-2.5-flash is no longer available")
    assert llm.is_api_error(nf) and llm.explain_error(nf).startswith("Modèle introuvable")
    assert llm.explain_error(ModelRateLimitError("429")).startswith("Quota gratuit")
    auth = type("AuthenticationError", (Exception,), {"status_code": 401})("bad key")
    assert llm.is_api_error(auth) and "GOOGLE_API_KEY" in llm.explain_error(auth)
    assert not llm.is_api_error(OutputParserException("json invalide"))
    try:
        Spec.model_validate({"oops": 1})
    except ValidationError as exc:
        assert not llm.is_api_error(exc)
    bad_json = type("BadRequestError", (Exception,), {"status_code": 400})("json_validate_failed")
    assert not llm.is_api_error(bad_json)  # Groq : sortie hors schéma -> boucle de correction


def test_generate_stops_immediately_on_api_error():
    from langchain_core.exceptions import ModelNotFoundError
    from langchain_core.runnables import RunnableLambda

    from sdd.graph import generate

    calls = []

    class DownLLM:
        def with_structured_output(self, schema, **_kw):
            def fail(_):
                calls.append(1)
                raise ModelNotFoundError("404 NOT_FOUND")
            return RunnableLambda(fail)

    with pytest.raises(ModelNotFoundError):
        generate(DownLLM(), Spec, "sys", "payload", lambda s: [], max_retries=3)
    assert len(calls) == 1  # pas de 3 relances inutiles, pas de « Sortie non conforme au schéma »
