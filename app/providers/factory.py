from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import settings
from app.providers.base import LLMProvider
from app.providers.deepseek import DeepSeekProvider
from app.providers.errors import ProviderConfigurationError
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.providers.model_registry import (
    CANONICAL_MODEL_IDS,
    DEEPSEEK_V3_0324,
    GEMINI_25_FLASH,
    GEMINI_25_PRO,
    MODEL_MAPPINGS,
)


@dataclass(frozen=True)
class MockModelProfile:
    quality: float
    input_price_per_million: float
    output_price_per_million: float
    latency_ms: float


# All values are simulated placeholders for local development, not current
# provider prices or measured model quality.
MOCK_MODEL_PROFILES = {
    DEEPSEEK_V3_0324: MockModelProfile(0.82, 0.20, 1.20, 250),
    GEMINI_25_FLASH: MockModelProfile(0.90, 2.00, 12.00, 500),
    GEMINI_25_PRO: MockModelProfile(0.96, 4.00, 20.00, 900),
}


class ProviderFactory:
    """Construct configured providers without leaking provider details upstream."""

    def __init__(
        self,
        mode: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        selected_mode = settings.llm_provider_mode if mode is None else mode
        self.mode = selected_mode.strip().lower()
        if self.mode not in {"mock", "real"}:
            raise ProviderConfigurationError("LLM_PROVIDER_MODE must be either 'mock' or 'real'.")
        self._http_client = http_client
        self._owns_client = http_client is None

    @property
    def http_client(self) -> httpx.AsyncClient | None:
        return self._http_client

    def _get_http_client(self) -> httpx.AsyncClient | None:
        if self.mode != "real":
            return None
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=settings.llm_request_timeout_s)
            self._owns_client = True
        return self._http_client

    async def startup(self) -> None:
        if self.mode == "real":
            self._get_http_client()

    async def close(self, force: bool = False) -> None:
        if self._http_client is not None:
            if (self._owns_client or force) and not self._http_client.is_closed:
                await self._http_client.aclose()
            self._http_client = None

    def get_provider(self, model_name: str) -> LLMProvider:
        mapping = MODEL_MAPPINGS.get(model_name)
        if mapping is None:
            supported = ", ".join(CANONICAL_MODEL_IDS)
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
        if self.mode == "real":
            return self._get_real_provider(mapping)
        profile = MOCK_MODEL_PROFILES[model_name]
        return MockProvider(
            model_name=model_name,
            quality=profile.quality,
            input_price_per_million=profile.input_price_per_million,
            output_price_per_million=profile.output_price_per_million,
            latency_ms=profile.latency_ms,
        )

    def _get_real_provider(self, mapping) -> LLMProvider:
        api_key = getattr(settings, mapping.api_key_setting)
        if not isinstance(api_key, str) or not api_key.strip():
            key_name = mapping.api_key_setting.upper()
            raise ProviderConfigurationError(
                f"{key_name} is required when requesting {mapping.canonical_id} in real mode."
            )
        api_model = getattr(settings, mapping.api_model_setting)
        if not isinstance(api_model, str) or not api_model.strip():
            variable = mapping.api_model_setting.upper()
            raise ProviderConfigurationError(
                f"{mapping.display_name} API model is not configured. Set {variable}."
            )
        api_model = api_model.strip()
        if any(char.isspace() for char in api_model) or api_model.startswith("/"):
            raise ProviderConfigurationError(
                f"{variable} must be a model ID without whitespace or a leading slash."
            )
        rates = settings.live_model_pricing.get(api_model)
        prices = {
            "input_price_per_million": rates.input_price_per_million if rates else None,
            "output_price_per_million": rates.output_price_per_million if rates else None,
        }
        client = self._get_http_client()
        if mapping.provider == "deepseek":
            return DeepSeekProvider(
                api_key=api_key,
                api_model_name=api_model,
                http_client=client,
                **prices,
            )
        return GeminiProvider(
            model_name=mapping.canonical_id,
            api_model_name=api_model,
            api_key=api_key,
            http_client=client,
            **prices,
        )


factory = ProviderFactory()
