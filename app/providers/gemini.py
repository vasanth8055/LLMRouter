from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.config import settings
from app.providers.base import LLMProvider, LLMResponse
from app.providers.errors import (
    ProviderConfigurationError,
    ProviderResponseError,
)
from app.providers.http_utils import (
    async_with_retries,
    raise_for_provider_status,
    translate_request_error,
)
from app.providers.pricing import estimated_cost, usage_count
from app.providers.model_registry import GEMINI_38_FLASH_API, MODEL_MAPPINGS

GEMINI_GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODELS = {
    model_id for model_id, mapping in MODEL_MAPPINGS.items() if mapping.provider == "gemini"
}


class GeminiProvider(LLMProvider):
    """Async adapter for Google's Gemini generateContent REST API."""

    def __init__(
        self,
        model_name: str,
        api_model_name: str | None = None,
        api_key: str | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
        timeout_s: float | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
        retry_base_delay_s: float | None = None,
        retry_max_delay_s: float | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if model_name not in GEMINI_MODELS:
            raise ProviderConfigurationError(
                f"Unsupported Gemini model '{model_name}'. Supported models: {', '.join(sorted(GEMINI_MODELS))}."
            )
        self.model_name = model_name
        self.api_model_name = api_model_name
        if not self.api_model_name:
            raise ProviderConfigurationError(
                f"API model ID is required for {model_name}; configure its live model mapping."
            )
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        if not self.api_key:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY is required when requesting a Gemini provider in real mode."
            )
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.timeout_s = timeout_s if timeout_s is not None else settings.llm_request_timeout_s
        self.http_client = http_client
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries
        self.retry_base_delay_s = (
            settings.llm_retry_base_delay_s if retry_base_delay_s is None else retry_base_delay_s
        )
        self.retry_max_delay_s = (
            settings.llm_retry_max_delay_s if retry_max_delay_s is None else retry_max_delay_s
        )
        self._sleep_fn = sleep_fn

    async def _post_once(
        self,
        url: str,
        payload: dict[str, object],
        safe_endpoint: str,
        prompt: str,
    ) -> tuple[httpx.Response, float]:
        started = time.perf_counter()
        try:
            if self.http_client is None:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(
                        url,
                        headers={
                            "x-goog-api-key": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            else:
                response = await self.http_client.post(
                    url,
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_s,
                )
        except httpx.RequestError as exc:
            raise translate_request_error(exc, "Gemini") from None
        latency_ms = (time.perf_counter() - started) * 1000
        raise_for_provider_status(
            response,
            "Gemini",
            safe_context=safe_endpoint,
            sensitive_values=(prompt,),
        )
        return response, latency_ms

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        generation_config: dict[str, object] = {}
        if self.api_model_name == GEMINI_38_FLASH_API:
            if temperature != 0.0:
                raise ProviderConfigurationError(
                    "Gemini 3.8 Flash does not accept temperature; use temperature=0.0."
                )
        else:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        payload: dict[str, object] = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if generation_config:
            payload["generationConfig"] = generation_config
        url = GEMINI_GENERATE_CONTENT_URL.format(model=self.api_model_name)
        safe_endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.api_model_name}:generateContent"
        ).replace(self.api_key, "[REDACTED]")
        retry_kwargs: dict[str, Any] = {
            "max_retries": self.max_retries,
            "base_delay_s": self.retry_base_delay_s,
            "max_delay_s": self.retry_max_delay_s,
        }
        if self._sleep_fn is not None:
            retry_kwargs["sleep_fn"] = self._sleep_fn
        response, latency_ms = await async_with_retries(
            lambda: self._post_once(url, payload, safe_endpoint, prompt),
            **retry_kwargs,
        )
        try:
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            texts = [part["text"] for part in parts if isinstance(part.get("text"), str)]
            if not texts:
                raise TypeError("response has no text parts")
            usage = data.get("usageMetadata") or {}
            input_tokens = usage_count(usage.get("promptTokenCount"))
            candidates_tokens = usage_count(usage.get("candidatesTokenCount"))
            thoughts_tokens = usage_count(usage.get("thoughtsTokenCount")) or 0
            output_tokens = (
                candidates_tokens + thoughts_tokens if candidates_tokens is not None else None
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ProviderResponseError("Gemini returned a malformed generateContent response.") from None
        cost = estimated_cost(
            input_tokens, output_tokens,
            self.input_price_per_million, self.output_price_per_million,
        )
        return LLMResponse(
            model_name=self.api_model_name,
            response="".join(texts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )
