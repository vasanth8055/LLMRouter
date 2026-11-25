from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceTelemetry:
    request_id: str
    prompt_hash: str
    prompt_length: int
    canonical_model: str | None
    provider_model: str | None
    predicted_quality: float | None
    model_mapping_warning: str | None
    routing_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    actual_latency_ms: float | None
    status: str
    error_type: str | None = None
    error_message: str | None = None
    attempted_models: list[str] | None = None
    failover: bool | None = None
    policy_fallback: bool | None = None
    retry_count: int | None = 0
