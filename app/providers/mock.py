from __future__ import annotations

import time

from app.providers.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """Deterministic, offline provider with explicitly simulated metrics.

    The quality and price settings are development placeholders. They do not
    represent measured model quality or current real provider pricing.
    """

    def __init__(
        self,
        model_name: str,
        quality: float,
        input_price_per_million: float,
        output_price_per_million: float,
        latency_ms: float,
    ) -> None:
        self.model_name = model_name
        self.quality = float(quality)
        self.input_price_per_million = float(input_price_per_million)
        self.output_price_per_million = float(output_price_per_million)
        self.latency_ms = float(latency_ms)

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Deliberately deterministic and network-free. The reported latency is
        # simulated; sleeping would make local development needlessly slow.
        del temperature  # Sampling is intentionally not simulated.
        started = time.perf_counter()
        input_tokens = max(1, len(prompt.split()))
        output_tokens = min(32, max_tokens) if max_tokens is not None else 32
        response = f"[{self.model_name}] Simulated response for: {prompt[:120]}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        cost = (
            input_tokens / 1_000_000 * self.input_price_per_million
            + output_tokens / 1_000_000 * self.output_price_per_million
        )
        return LLMResponse(
            model_name=self.model_name,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=max(self.latency_ms, elapsed_ms),
            estimated_cost_usd=cost,
        )
