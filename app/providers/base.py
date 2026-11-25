from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral result returned by every inference provider."""

    model_name: str
    response: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    estimated_cost_usd: float | None


class LLMProvider(ABC):
    """Asynchronous interface implemented by all model providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError
