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

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekProvider(LLMProvider):
    """Async adapter for DeepSeek's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_model_name: str | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
        timeout_s: float | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
        retry_base_delay_s: float | None = None,
        retry_max_delay_s: float | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.deepseek_api_key
        self.api_model_name = api_model_name if api_model_name is not None else settings.deepseek_api_model
        if not self.api_key:
            raise ProviderConfigurationError(
                "DEEPSEEK_API_KEY is required when requesting the DeepSeek provider in real mode."
            )
        if not self.api_model_name:
            raise ProviderConfigurationError(
                "Set DEEPSEEK_API_MODEL to a currently supported DeepSeek API model ID. "
                "The benchmark alias deepseek-v3-0324 is not assumed to be a live API model."
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

    async def _post_once(self, payload: dict[str, object]) -> tuple[httpx.Response, float]:
        started = time.perf_counter()
        try:
            if self.http_client is None:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(
                        DEEPSEEK_CHAT_COMPLETIONS_URL,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
            else:
                response = await self.http_client.post(
                    DEEPSEEK_CHAT_COMPLETIONS_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout_s,
                )
        except httpx.RequestError as exc:
            raise translate_request_error(exc, "DeepSeek") from None
        latency_ms = (time.perf_counter() - started) * 1000
        raise_for_provider_status(response, "DeepSeek")
        return response, latency_ms

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.api_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        retry_kwargs: dict[str, Any] = {
            "max_retries": self.max_retries,
            "base_delay_s": self.retry_base_delay_s,
            "max_delay_s": self.retry_max_delay_s,
        }
        if self._sleep_fn is not None:
            retry_kwargs["sleep_fn"] = self._sleep_fn
        response, latency_ms = await async_with_retries(
            lambda: self._post_once(payload),
            **retry_kwargs,
        )
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("message content is not text")
            usage = data.get("usage") or {}
            input_tokens = usage_count(usage.get("prompt_tokens"))
            output_tokens = usage_count(usage.get("completion_tokens"))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ProviderResponseError("DeepSeek returned a malformed chat completion response.") from None
        cost = estimated_cost(
            input_tokens, output_tokens,
            self.input_price_per_million, self.output_price_per_million,
        )
        return LLMResponse(
            model_name=self.api_model_name,
            response=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )
