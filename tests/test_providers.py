from __future__ import annotations

import asyncio
import importlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers.base import LLMProvider, LLMResponse
from app.providers.factory import MOCK_MODEL_PROFILES, ProviderFactory
from app.providers.deepseek import DEEPSEEK_CHAT_COMPLETIONS_URL, DeepSeekProvider
from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.providers.gemini import GEMINI_GENERATE_CONTENT_URL, GeminiProvider
from app.providers.mock import MockProvider
from app.providers.model_registry import GEMINI_25_FLASH, GEMINI_38_FLASH_API
from app.schemas import RouteRequest
from app.services.inference_service import InferenceService
from app.services.router_service import RouterService


def test_mock_provider_is_async_deterministic_and_returns_normalized_response():
    provider = MockProvider("mock-test", 0.8, 1.0, 2.0, 12)
    first = asyncio.run(provider.generate("hello world", max_tokens=5))
    second = asyncio.run(provider.generate("hello world", max_tokens=5))
    assert isinstance(provider, LLMProvider)
    assert isinstance(first, LLMResponse)
    assert first == second
    assert first.model_name == "mock-test"
    assert first.response == "[mock-test] Simulated response for: hello world"
    assert first.input_tokens == 2
    assert first.output_tokens == 5
    assert first.latency_ms >= 12
    assert first.estimated_cost_usd == pytest.approx(12 / 1_000_000)


@pytest.mark.parametrize("model_name", list(MOCK_MODEL_PROFILES))
def test_factory_resolves_all_canonical_models_to_mock(model_name: str):
    provider = ProviderFactory(mode="mock").get_provider(model_name)
    assert isinstance(provider, MockProvider)
    assert provider.model_name == model_name


def test_factory_rejects_unsupported_model_with_clear_message():
    with pytest.raises(ValueError, match="Unsupported model 'unknown-model'"):
        ProviderFactory(mode="mock").get_provider("unknown-model")


def test_mock_mode_needs_no_api_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = ProviderFactory(mode="mock").get_provider("gemini-2.5-flash")
    result = asyncio.run(provider.generate("offline prompt"))
    assert result.response.startswith("[gemini-2.5-flash]")


def test_provider_cost_and_latency_are_numeric_and_simulated():
    provider = ProviderFactory(mode="mock").get_provider("deepseek-v3-0324")
    result = asyncio.run(provider.generate("count my tokens"))
    assert isinstance(result.estimated_cost_usd, float)
    assert result.estimated_cost_usd >= 0
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms > 0


def test_inference_service_returns_normalized_llmresponse():
    response = asyncio.run(
        InferenceService(ProviderFactory(mode="mock")).generate(
            "gemini-2.5-pro", "Explain a simple idea", max_tokens=12
        )
    )
    assert isinstance(response, LLMResponse)
    assert response.model_name == "gemini-2.5-pro"
    assert response.output_tokens == 12
    assert response.estimated_cost_usd > 0


def test_router_service_smoke_uses_inference_layer_and_frozen_model():
    result = asyncio.run(
        RouterService().handle_async(
            RouteRequest(prompt="Rewrite this sentence politely: Please send the report.")
        )
    )
    assert result.selected_model == "gemini-2.5-pro"
    assert result.response.startswith("[gemini-2.5-pro] Simulated response")
    assert result.estimated_cost_usd > 0
    assert result.latency_ms == 900


def test_fastapi_route_and_application_import_smoke():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post("/v1/chat", json={"prompt": "A short local mock request"})
    assert response.status_code == 200
    assert response.json()["selected_model"] == "gemini-2.5-pro"
    assert response.json()["response"].startswith("[gemini-2.5-pro]")


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_deepseek_success_usage_cost_and_request_parameters():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello from deepseek"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4},
        })

    async def run():
        async with _client(handler) as client:
            provider = DeepSeekProvider(
                api_key="test-key", api_model_name="deepseek-v4-pro",
                input_price_per_million=2, output_price_per_million=5,
                http_client=client,
            )
            return await provider.generate("hello", max_tokens=9, temperature=0.3)

    result = asyncio.run(run())
    assert seen["url"] == DEEPSEEK_CHAT_COMPLETIONS_URL
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "deepseek-v4-pro"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["body"]["max_tokens"] == 9 and seen["body"]["temperature"] == 0.3
    assert result.model_name == "deepseek-v4-pro"
    assert result.response == "hello from deepseek"
    assert result.input_tokens == 7 and result.output_tokens == 4
    assert result.estimated_cost_usd == pytest.approx((7 * 2 + 4 * 5) / 1_000_000)
    assert result.latency_ms >= 0


@pytest.mark.parametrize(("status", "exception"), [
    (401, ProviderAuthenticationError), (429, ProviderRateLimitError),
])
def test_deepseek_auth_and_rate_limit_failures_are_sanitized(status, exception):
    def handler(request):
        return httpx.Response(status, text="response included test-key")

    async def run():
        async with _client(handler) as client:
            provider = DeepSeekProvider(api_key="test-key", api_model_name="deepseek-v4-pro", http_client=client)
            await provider.generate("hello")

    with pytest.raises(exception) as caught:
        asyncio.run(run())
    assert "test-key" not in str(caught.value)


def test_deepseek_timeout_and_malformed_response():
    def timeout_handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    def malformed_handler(request):
        return httpx.Response(200, json={"choices": []})

    async def request(handler):
        async with _client(handler) as client:
            provider = DeepSeekProvider(api_key="key", api_model_name="deepseek-flash", http_client=client)
            await provider.generate("hello")

    with pytest.raises(ProviderTimeoutError):
        asyncio.run(request(timeout_handler))
    with pytest.raises(ProviderResponseError):
        asyncio.run(request(malformed_handler))


def test_deepseek_absent_usage_and_unconfigured_pricing_are_explicit():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        async with _client(handler) as client:
            provider = DeepSeekProvider(api_key="key", api_model_name="deepseek-flash", http_client=client)
            return await provider.generate("hello")

    result = asyncio.run(run())
    assert result.input_tokens is None and result.output_tokens is None
    assert result.estimated_cost_usd is None


def test_gemini_success_usage_cost_and_request_parameters():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["content_type"] = request.headers.get("content-type")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "hello "}, {"text": "gemini"}]}}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 3},
        })

    async def run():
        async with _client(handler) as client:
            provider = GeminiProvider(
                GEMINI_25_FLASH, api_model_name=GEMINI_38_FLASH_API, api_key="test-key", input_price_per_million=1,
                output_price_per_million=2, http_client=client,
            )
            return provider, await provider.generate("hello", max_tokens=17, temperature=0.0)

    provider, result = asyncio.run(run())
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    assert seen["url"] == GEMINI_GENERATE_CONTENT_URL.format(model=GEMINI_38_FLASH_API)
    assert seen["method"] == "POST"
    assert seen["key"] == "test-key"
    assert seen["content_type"] == "application/json"
    assert seen["body"]["contents"] == [{"parts": [{"text": "hello"}]}]
    assert seen["body"]["generationConfig"] == {"maxOutputTokens": 17}
    assert provider.model_name == GEMINI_25_FLASH
    assert provider.api_model_name == GEMINI_38_FLASH_API
    assert result.model_name == GEMINI_38_FLASH_API
    assert result.response == "hello gemini"
    assert result.input_tokens == 11 and result.output_tokens == 8
    assert result.estimated_cost_usd == pytest.approx((11 + 16) / 1_000_000)
    assert result.latency_ms >= 0


@pytest.mark.parametrize(("status", "exception"), [
    (401, ProviderAuthenticationError), (403, ProviderAuthenticationError),
    (429, ProviderRateLimitError),
])
def test_gemini_auth_and_rate_limit_failures_are_sanitized(status, exception):
    def handler(request):
        return httpx.Response(status, text="response included test-key")

    async def run():
        async with _client(handler) as client:
            provider = GeminiProvider("gemini-2.5-pro", api_model_name="gemini-pro-live", api_key="test-key", http_client=client)
            await provider.generate("hello")

    with pytest.raises(exception) as caught:
        asyncio.run(run())
    assert "test-key" not in str(caught.value)


def test_gemini_404_includes_safe_endpoint_and_sanitized_provider_message():
    def handler(request):
        return httpx.Response(404, json={
            "error": {
                "code": 404,
                "message": "models/gemini-2.5-flash unavailable; x-goog-api-key: test-key",
            }
        })

    async def run():
        async with _client(handler) as client:
            provider = GeminiProvider(
                "gemini-2.5-flash", api_model_name="gemini-2.5-flash",
                api_key="test-key", http_client=client,
            )
            await provider.generate("hello")

    with pytest.raises(ProviderError) as caught:
        asyncio.run(run())
    message = str(caught.value)
    assert "HTTP 404" in message
    assert "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" in message
    assert "models/gemini-2.5-flash unavailable" in message
    assert "test-key" not in message
    assert "[REDACTED]" in message


@pytest.mark.parametrize("status", [500, 503])
def test_gemini_5xx_includes_bounded_sanitized_provider_message(status):
    prompt = "private prompt content"

    def handler(request):
        return httpx.Response(status, json={
            "error": {
                "code": status,
                "message": (
                    f"temporary backend failure; x-goog-api-key: test-key; "
                    f"Authorization: Bearer test-key; {prompt}; "
                    + "detail " * 100
                ),
            }
        })

    async def run():
        async with _client(handler) as client:
            provider = GeminiProvider(
                "gemini-2.5-flash",
                api_model_name="gemini-3.8-flash",
                api_key="test-key",
                http_client=client,
            )
            await provider.generate(prompt)

    with pytest.raises(ProviderError) as caught:
        asyncio.run(run())
    message = str(caught.value)
    assert f"HTTP {status}" in message
    assert "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent" in message
    assert "temporary backend failure" in message
    assert "test-key" not in message
    assert prompt not in message
    provider_detail = message.split("Provider message: ", 1)[1]
    assert len(provider_detail) <= 400


def test_gemini_timeout_and_malformed_response():
    def timeout_handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    def malformed_handler(request):
        return httpx.Response(200, json={"candidates": []})

    async def request(handler):
        async with _client(handler) as client:
            provider = GeminiProvider("gemini-2.5-flash", api_model_name="gemini-flash-live", api_key="key", http_client=client)
            await provider.generate("hello")

    with pytest.raises(ProviderTimeoutError):
        asyncio.run(request(timeout_handler))
    with pytest.raises(ProviderResponseError):
        asyncio.run(request(malformed_handler))


def test_gemini_38_rejects_unsupported_temperature_without_http_request():
    async def run():
        async with _client(lambda request: pytest.fail("unexpected HTTP request")) as client:
            provider = GeminiProvider(
                GEMINI_25_FLASH,
                api_model_name=GEMINI_38_FLASH_API,
                api_key="test-key",
                http_client=client,
            )
            await provider.generate("hello", temperature=0.2)

    with pytest.raises(ProviderConfigurationError, match="does not accept temperature"):
        asyncio.run(run())


def test_gemini_missing_usage_or_pricing_yields_unavailable_cost():
    def handler(request):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    async def run():
        async with _client(handler) as client:
            provider = GeminiProvider("gemini-2.5-pro", api_model_name="gemini-pro-live", api_key="key", http_client=client)
            return await provider.generate("hello")

    result = asyncio.run(run())
    assert result.input_tokens is None and result.output_tokens is None
    assert result.estimated_cost_usd is None


@pytest.mark.parametrize(("model", "provider_type"), [
    ("deepseek-v3-0324", DeepSeekProvider),
    ("gemini-2.5-flash", GeminiProvider),
    ("gemini-2.5-pro", GeminiProvider),
])
def test_real_factory_resolves_provider_types_and_checks_keys(monkeypatch, model, provider_type):
    factory_module = importlib.import_module("app.providers.factory")

    monkeypatch.setattr(factory_module.settings, "deepseek_api_key", "ds-test-key")
    monkeypatch.setattr(factory_module.settings, "deepseek_api_model", "deepseek-flash")
    monkeypatch.setattr(factory_module.settings, "gemini_api_key", "gm-test-key")
    monkeypatch.setattr(factory_module.settings, "gemini_flash_api_model", "gemini-flash-live")
    monkeypatch.setattr(factory_module.settings, "gemini_pro_api_model", "gemini-pro-live")
    provider = ProviderFactory(mode="real").get_provider(model)
    assert isinstance(provider, provider_type)


def test_explicit_gemini_38_mapping_preserves_canonical_identity(monkeypatch):
    factory_module = importlib.import_module("app.providers.factory")
    monkeypatch.setattr(factory_module.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(factory_module.settings, "gemini_flash_api_model", GEMINI_38_FLASH_API)
    provider = ProviderFactory(mode="real").get_provider(GEMINI_25_FLASH)
    assert isinstance(provider, GeminiProvider)
    assert provider.model_name == GEMINI_25_FLASH
    assert provider.api_model_name == GEMINI_38_FLASH_API


@pytest.mark.parametrize(("model", "key_name", "extra"), [
    ("deepseek-v3-0324", "deepseek_api_key", {"deepseek_api_model": "deepseek-flash"}),
    ("gemini-2.5-flash", "gemini_api_key", {"gemini_flash_api_model": "gemini-flash-live"}),
    ("gemini-2.5-pro", "gemini_api_key", {"gemini_pro_api_model": "gemini-pro-live"}),
])
def test_real_factory_missing_provider_key_is_actionable(monkeypatch, model, key_name, extra):
    factory_module = importlib.import_module("app.providers.factory")

    monkeypatch.setattr(factory_module.settings, key_name, None)
    for name, value in extra.items():
        monkeypatch.setattr(factory_module.settings, name, value)
    with pytest.raises(ProviderConfigurationError, match="API_KEY is required"):
        ProviderFactory(mode="real").get_provider(model)


def test_factory_rejects_missing_deepseek_live_model_mapping(monkeypatch):
    factory_module = importlib.import_module("app.providers.factory")

    monkeypatch.setattr(factory_module.settings, "deepseek_api_key", "ds-test-key")
    monkeypatch.setattr(factory_module.settings, "deepseek_api_model", None)
    with pytest.raises(ProviderConfigurationError, match="DEEPSEEK_API_MODEL"):
        ProviderFactory(mode="real").get_provider("deepseek-v3-0324")


def test_real_mode_factory_without_key_does_not_make_http_request():
    with pytest.raises(ProviderConfigurationError, match="DEEPSEEK_API_KEY"):
        ProviderFactory(mode="real").get_provider("deepseek-v3-0324")


@pytest.mark.parametrize(("canonical", "setting"), [
    ("gemini-2.5-flash", "gemini_flash_api_model"),
    ("gemini-2.5-pro", "gemini_pro_api_model"),
])
def test_gemini_real_factory_requires_explicit_api_model_id(monkeypatch, canonical, setting):
    factory_module = importlib.import_module("app.providers.factory")
    monkeypatch.setattr(factory_module.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(factory_module.settings, setting, None)
    with pytest.raises(ProviderConfigurationError, match=setting.upper()):
        ProviderFactory(mode="real").get_provider(canonical)
