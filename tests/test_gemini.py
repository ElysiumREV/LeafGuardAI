from types import SimpleNamespace

import pytest

from leafguardai.ai import gemini
from leafguardai.ai.gemini import GuidanceError, discover_models, request_guidance


class ApiError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message or f"error {code}")
        self.code = code


def listing(*names, actions=("generateContent",)):
    return [SimpleNamespace(name=f"models/{n}", supported_actions=list(actions)) for n in names]


class FakeClient:
    """Cliente falso: `outcomes` mapeia modelo -> texto de resposta ou exceção a levantar."""

    def __init__(self, outcomes, models=()):
        self.calls = []
        self.models = SimpleNamespace(list=lambda: list(models))
        self.interactions = SimpleNamespace(create=self._create)
        self._outcomes = outcomes

    def _create(self, model, input):
        self.calls.append(model)
        outcome = self._outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(output_text=outcome)


def test_discover_models_filters_and_sorts_newest_first():
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: listing(
        "gemini-2.5-flash",
        "gemini-3.8-flash-lite",
        "gemini-3.8-flash-preview-05",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash-tts",
        "gemini-2.5-flash-image",
        "gemini-3.8-pro",
        "text-embedding-004",
    )))
    assert discover_models(client) == [
        "gemini-3.8-flash",
        "gemini-3.8-flash-preview-05",
        "gemini-3.8-flash-lite",
        "gemini-3.7-flash",
        "gemini-2.5-flash",
    ]


def test_discover_models_skips_models_without_generate_content():
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: listing("gemini-3.8-flash", actions=("embedContent",))))
    assert discover_models(client) == []


@pytest.mark.parametrize("code", [404, 429, 500, 503, 504])
def test_retryable_codes(code):
    assert gemini.is_retryable(ApiError(code))


@pytest.mark.parametrize("error", [ApiError(400, "API key not valid"), ApiError(403), ValueError("boom")])
def test_non_retryable_errors(error):
    assert not gemini.is_retryable(error)


def test_auto_mode_switches_to_next_model_when_busy():
    client = FakeClient(
        {"gemini-3.8-flash": ApiError(503, "overloaded"), "gemini-3.7-flash": "resposta"},
        models=listing("gemini-3.7-flash", "gemini-3.8-flash"),
    )
    guidance = request_guidance(client, "prompt")
    assert guidance.text == "resposta"
    assert guidance.model == "gemini-3.7-flash"
    assert guidance.failed == ["gemini-3.8-flash"]
    assert client.calls == ["gemini-3.8-flash", "gemini-3.7-flash"]


def test_auto_mode_stops_on_invalid_key_without_trying_other_models():
    client = FakeClient(
        {"gemini-3.8-flash": ApiError(400, "API key not valid"), "gemini-3.7-flash": "x"},
        models=listing("gemini-3.8-flash", "gemini-3.7-flash"),
    )
    with pytest.raises(GuidanceError) as info:
        request_guidance(client, "prompt")
    assert client.calls == ["gemini-3.8-flash"]
    assert info.value.failed == []
    assert "Chave da API inválida" in gemini.describe_error(info.value.error)


def test_explicit_model_is_never_substituted():
    client = FakeClient({"gemini-2.5-flash": ApiError(429, "quota")}, models=listing("gemini-3.8-flash"))
    with pytest.raises(GuidanceError) as info:
        request_guidance(client, "prompt", model="gemini-2.5-flash")
    assert client.calls == ["gemini-2.5-flash"]
    assert info.value.failed == ["gemini-2.5-flash"]


def test_all_models_busy_raises_with_every_failure():
    # lista vazia da API -> usa FALLBACK_MODELS; todos falham por disponibilidade
    client = FakeClient({name: ApiError(503) for name in gemini.FALLBACK_MODELS})
    with pytest.raises(GuidanceError) as info:
        request_guidance(client, "prompt")
    assert info.value.failed == gemini.FALLBACK_MODELS


def test_listing_failure_falls_back_to_static_list():
    def broken():
        raise ApiError(500, "listing down")

    client = FakeClient({name: "ok" for name in gemini.FALLBACK_MODELS})
    client.models = SimpleNamespace(list=broken)
    guidance = request_guidance(client, "prompt")
    assert guidance.model == gemini.FALLBACK_MODELS[0]


@pytest.mark.parametrize(
    "error, fragment",
    [
        (ApiError(403, "permission denied"), "Chave da API inválida"),
        (ApiError(404, "model not found"), "Modelo não encontrado"),
        (ApiError(429, "quota exceeded"), "cota"),
        (ConnectionError("connect failed"), "conexão"),
        (RuntimeError("???"), "Não foi possível"),
    ],
)
def test_describe_error(error, fragment):
    assert fragment in gemini.describe_error(error)
