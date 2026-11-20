"""Fabrique de modèle : Gemini par défaut, autres fournisseurs par variable d'env (sans réseau)."""
import pytest

from sdd import llm
from sdd.models import Spec


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ("SDD_PROVIDER", "SDD_MODEL", "SDD_FALLBACK_MODEL", "SDD_FALLBACK_PROVIDER", "SDD_NUM_CTX"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")


def test_gemini_is_default_provider(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    m = llm.get_llm()
    assert isinstance(m, ChatGoogleGenerativeAI)
    assert (m.model, m.temperature, m.seed) == ("gemini-3.8-flash", 0, 42)
    assert llm.fallback_label() == "groq/openai/gpt-oss-120b"  # clé Groq présente : secours chez Groq
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

    from sdd.llm import generate

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


def test_gemini_call_has_timeout_and_short_thinking(monkeypatch):
    """Un appel ne peut jamais bloquer indéfiniment (écran figé sur la première étape)."""
    m = llm.get_llm()
    assert m.timeout == 120 and m.max_retries == 2
    assert m.thinking_config == {"thinking_level": "low"}
    monkeypatch.setenv("SDD_TIMEOUT", "30")
    monkeypatch.setenv("SDD_THINKING", "high")
    m = llm.get_llm()
    assert m.timeout == 30 and m.thinking_config == {"thinking_level": "high"}
    assert llm.get_llm("gemini-2.5-flash").thinking_config is None  # thinking_level : Gemini 3 seulement


def test_timeout_stops_the_step_with_a_clear_message():
    class ReadTimeout(Exception):
        pass

    exc = ReadTimeout("timed out")
    assert llm.is_api_error(exc)  # pas de 4 relances de 2 minutes chacune
    assert "n'a pas répondu dans le délai" in llm.explain_error(exc)


def test_secours_chez_l_autre_fournisseur(monkeypatch):
    """Clé Groq présente et aucun secours fixé : Gemini bascule sur Groq en cas de panne ou de lenteur."""
    from sdd import llm

    for var in ("SDD_FALLBACK_MODEL", "SDD_FALLBACK_PROVIDER", "SDD_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SDD_PROVIDER", "gemini")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert llm.fallback_provider() == "groq"
    assert llm.fallback_label() == "groq/" + llm.DEFAULTS["groq"]
    fb = llm.get_fallback_llm()
    assert type(fb).__name__ == "ChatGroq" and fb.request_timeout == llm.llm_timeout()
    monkeypatch.setenv("SDD_FALLBACK_MODEL", "")  # désactivation explicite
    assert llm.fallback_label() == "" and llm.get_fallback_llm() is None
    monkeypatch.delenv("SDD_FALLBACK_MODEL")
    monkeypatch.delenv("GROQ_API_KEY")
    assert llm.fallback_provider() == "gemini" and llm.get_fallback_llm() is None


def test_rate_limit_per_minute_waits_daily_quota_stops():
    from langchain_core.exceptions import ModelRateLimitError

    class Groq429(Exception):
        status_code = 429

    tpm = Groq429("Error code: 429 - Rate limit reached for model openai/gpt-oss-120b on tokens per minute (TPM): "
                  "Limit 8000, Used 3634, Requested 4990. Please try again in 4.68s.")
    assert abs(llm.rate_limit_wait(tpm) - 5.68) < 0.01
    assert llm.rate_limit_wait(Groq429("... try again in 640ms")) < 2
    tpd = Groq429("Rate limit reached on tokens per day (TPD): Limit 200000. Please try again in 7m12s.")
    assert llm.rate_limit_wait(tpd) is None and "JOURNALIER" in llm.explain_error(tpd)
    assert llm.rate_limit_wait(ModelRateLimitError("429 RESOURCE_EXHAUSTED retryDelay: '30s'")) == 31
    assert llm.rate_limit_wait(Groq429("Please try again in 3m0s")) is None  # trop long : pas une limite par minute
    assert llm.rate_limit_wait(ValueError("schéma invalide")) is None


def test_generate_waits_on_per_minute_limit(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from sdd.models import Spec

    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    class Groq429(Exception):
        status_code = 429

    calls = {"n": 0}

    class Limited:
        def with_structured_output(self, schema, **_kw):
            def run(_):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise Groq429("tokens per minute (TPM). Please try again in 4.68s.")
                return {"invented": None}
            return RunnableLambda(run)

    import pytest
    with pytest.raises(Exception):  # la 2e réponse est invalide pour le schéma : peu importe ici
        llm.generate(Limited(), Spec, "s", "x", lambda o: [], 0)
    assert calls["n"] == 2 and len(slept) == 1 and 5 < slept[0] < 6
