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
    # Gemini 3 : température 1.0 recommandée par Google (en dessous : boucles, délai dépassé)
    assert (m.model, m.temperature, m.seed) == ("gemini-3.8-flash", 1.0, 42)
    assert m.max_output_tokens == 16384
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


def test_both_errors_shown_when_primary_and_fallback_fail():
    """Gemini choisi, Groq en secours, les deux en échec : on voit les DEUX erreurs, pas seulement Groq."""
    import pytest
    from langchain_core.runnables import RunnableLambda

    from sdd.models import Spec

    class E(Exception):
        def __init__(self, msg, code):
            super().__init__(msg)
            self.status_code = code

    class M:
        def __init__(self, name, exc):
            self.model, self.exc = name, exc

        def with_structured_output(self, schema, **_kw):
            def boom(_):
                raise self.exc
            return RunnableLambda(boom)

    primary = M("gemini-3.8-flash", E("404 model not found", 404))
    backup = M("openai/gpt-oss-120b", E("Rate limit ... tokens per day (TPD) ... try again in 13m58s", 429))
    with pytest.raises(llm.AllModelsFailed) as err:
        llm.generate(primary, Spec, "s", "x", lambda o: [], 0, fallback_llm=backup)
    msg = llm.explain_error(err.value)
    assert "gemini-3.8-flash" in msg and "introuvable" in msg
    assert "openai/gpt-oss-120b" in msg and "JOURNALIER" in msg


GROQ_413 = ("Error code: 413 - {'error': {'message': 'Request too large for model openai/gpt-oss-120b in organization "
            "org_x service tier on_demand on tokens per minute (TPM): Limit 8000, Requested 8604, please reduce your "
            "message size and try again.'}}")


class _StatusError(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.status_code = code


def _fake_model(cls_name, model, outputs):
    """Faux modèle LangChain dont la classe porte le nom du vrai (ChatGroq…) ; `outputs` : exception ou objet."""
    from langchain_core.runnables import RunnableLambda

    calls = []

    def with_structured_output(self, schema, **_kw):
        def run(_inp):
            calls.append(model)
            out = outputs.pop(0) if len(outputs) > 1 else outputs[0]
            if isinstance(out, Exception):
                raise out
            return out
        return RunnableLambda(run)

    cls = type(cls_name, (), {"with_structured_output": with_structured_output})
    m = cls()
    m.model, m.calls = model, calls
    return m


def test_groq_413_request_too_large_is_an_api_error_with_a_clear_french_message():
    exc = _StatusError(GROQ_413, 413)
    assert llm.is_request_too_large(exc) and llm.is_api_error(exc)
    assert llm.rate_limit_wait(exc) is None  # attendre ne sert à rien
    msg = llm.explain_error(exc)
    assert msg.startswith("Requête trop volumineuse") and "8604 tokens demandés" in msg and "8000" in msg
    assert "Reprendre" in msg and "gemini" in msg
    # même message de taille renvoyé en 429 : ce n'est pas une limite qu'une attente peut lever
    as_429 = _StatusError(GROQ_413.replace("413", "429"), 429)
    assert llm.is_request_too_large(as_429) and llm.rate_limit_wait(as_429) is None
    # un vrai 429 par minute (taille < limite) reste une attente
    tpm = _StatusError("Rate limit reached on tokens per minute (TPM): Limit 8000, Used 7000, Requested 2000. "
                       "Please try again in 4.5s", 429)
    assert not llm.is_request_too_large(_StatusError("Limit 8000, Requested 500", 429))
    assert llm.rate_limit_wait(tpm) is not None


def test_413_is_not_retried_as_a_schema_error_and_skips_same_provider_backup(monkeypatch):
    """Avant : le 413 était pris pour une sortie hors schéma, redemandé avec une correction (requête encore plus
    grosse), puis « aucune sortie valide ». Maintenant : arrêt immédiat, et le secours Groq (même limite) est sauté."""
    import pytest
    from sdd.models import Spec

    groq = _fake_model("ChatGroq", "openai/gpt-oss-120b", [_StatusError(GROQ_413, 413)])
    groq_small = _fake_model("ChatGroq", "openai/gpt-oss-20b", [_StatusError(GROQ_413, 413)])
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(Exception) as err:
        llm.generate(groq, Spec, "s", "x", lambda o: [], 3, fallback_llm=groq_small)
    assert groq.calls == ["openai/gpt-oss-120b"]  # un seul appel, pas 4
    assert groq_small.calls == []                 # même fournisseur = même limite : inutile
    assert "Requête trop volumineuse" in llm.explain_error(err.value)


def test_413_on_groq_switches_to_gemini_when_its_key_is_present(monkeypatch):
    groq = _fake_model("ChatGroq", "openai/gpt-oss-120b", [_StatusError(GROQ_413, 413)])
    gemini = _fake_model("ChatGoogleGenerativeAI", "gemini-3.8-flash", [{"ok": True}])
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.delenv("SDD_FALLBACK_MODEL", raising=False)
    monkeypatch.setattr(llm, "get_llm", lambda model=None, prov=None: gemini)

    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    obj, errs = llm.generate(groq, Out, "s", "x", lambda o: [], 2)
    assert obj.ok and errs == [] and gemini.calls == ["gemini-3.8-flash"]
    # secours désactivé (« (aucun) ») : pas de bascule implicite
    monkeypatch.setenv("SDD_FALLBACK_MODEL", "")
    assert llm.rescue_llm("groq") is None
    assert llm.rescue_llm("gemini") is None


def test_groq_reasoning_is_short_and_older_gemini_keeps_temperature_zero(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    g = llm.get_llm("openai/gpt-oss-120b", "groq")
    assert g.reasoning_effort == "low" and g.temperature < 0.01
    assert llm.gemini_temperature("gemini-2.5-flash") == 0.0
    monkeypatch.setenv("SDD_GEMINI_TEMPERATURE", "0.5")
    assert llm.gemini_temperature("gemini-3.8-flash") == 0.5


def test_groq_answer_is_bounded_by_what_remains_of_the_per_minute_limit(monkeypatch):
    """Groq compte la réponse attendue dans sa limite de 8 000 tokens/minute : on borne max_tokens."""
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    monkeypatch.setenv("GROQ_API_KEY", "x")
    seen = {}
    real = llm.get_llm("openai/gpt-oss-120b", "groq")

    def fake_structured(self, schema, **_kw):
        from langchain_core.runnables import RunnableLambda
        seen["max_tokens"] = self.max_tokens
        return RunnableLambda(lambda _i: Out(ok=True))

    monkeypatch.setattr(type(real), "with_structured_output", fake_structured)
    payload = "x" * 9000  # ≈ 3 000 tokens
    obj, errs = llm.generate(real, Out, "s", payload, lambda o: [], 0)
    assert obj.ok and errs == []
    assert 1024 <= seen["max_tokens"] <= 8000 - 3000 and real.max_tokens is None  # copie, l'original intact


def test_no_rescue_to_gemini_when_gemini_already_timed_out(monkeypatch):
    """Cas réel : Gemini dépasse le délai, Groq refuse (413) ; on ne relance pas Gemini 120 s de plus."""
    import pytest
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    gemini = _fake_model("ChatGoogleGenerativeAI", "gemini-3.8-flash", [llm.CallTimeout("aucune réponse en 120 s")])
    groq = _fake_model("ChatGroq", "openai/gpt-oss-120b", [_StatusError(GROQ_413, 413)])
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.delenv("SDD_FALLBACK_MODEL", raising=False)
    monkeypatch.setattr(llm, "get_llm", lambda *a, **k: pytest.fail("secours Gemini relancé"))
    with pytest.raises(llm.AllModelsFailed) as err:
        llm.generate(gemini, Out, "s", "x", lambda o: [], 2, fallback_llm=groq)
    assert gemini.calls == ["gemini-3.8-flash"] and groq.calls == ["openai/gpt-oss-120b"]
    assert len(err.value.failures) == 2


def test_groq_413_recalibrates_answer_budget_once_and_succeeds(monkeypatch):
    """Groq annonce le dépassement exact (« Limit 8000, Requested N ») : on réduit max_tokens d'autant, une fois."""
    from langchain_core.runnables import RunnableLambda
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    monkeypatch.setenv("GROQ_API_KEY", "x")
    real = llm.get_llm("openai/gpt-oss-120b", "groq")
    true_input = 5650  # taille réelle comptée par Groq, plus grosse que notre estimation
    tried = []

    def fake_structured(self, schema, **_kw):
        def run(_i):
            tried.append(self.max_tokens)
            requested = true_input + self.max_tokens
            if requested > 8000:
                raise _StatusError(GROQ_413.replace("Requested 8604", f"Requested {requested}"), 413)
            return Out(ok=True)
        return RunnableLambda(run)

    monkeypatch.setattr(type(real), "with_structured_output", fake_structured)
    obj, errs = llm.generate(real, Out, "s", "x" * 9000, lambda o: [], 0)
    assert obj.ok and len(tried) == 2 and true_input + tried[1] <= 8000
