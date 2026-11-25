from __future__ import annotations

from app.providers.base import LLMResponse
from app.providers.factory import ProviderFactory, factory


class InferenceService:
    """Resolve a selected model to a provider and return normalized output."""

    def __init__(self, provider_factory: ProviderFactory | None = None) -> None:
        self.provider_factory = provider_factory or factory

    async def startup(self) -> None:
        if hasattr(self.provider_factory, "startup"):
            await self.provider_factory.startup()

    async def generate(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        provider = self.provider_factory.get_provider(model_name)
        return await provider.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    async def close(self) -> None:
        if hasattr(self.provider_factory, "close"):
            await self.provider_factory.close()
