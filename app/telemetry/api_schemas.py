from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class AnalyticsSummary(BaseModel):
    total_requests: int
    average_cost_usd: float | None
    average_latency_ms: float | None
    error_rate: float
    fallback_rate: float | None = None
    failover_rate: float | None = None


class ModelAnalytics(BaseModel):
    canonical_model: str
    request_count: int
    average_cost_usd: float | None
    total_cost_usd: float | None
    average_latency_ms: float | None
    average_predicted_quality: float | None


class InferenceRequestRecord(BaseModel):
    request_id: str
    created_at: datetime
    prompt_hash: str
    canonical_model: str | None
    provider_model: str | None
    predicted_quality: float | None
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    actual_latency_ms: float | None
    status: Literal["success", "failed"]
    attempted_models: list[str] | None = None
    failover: bool | None = None
    policy_fallback: bool | None = None
    retry_count: int | None = None


class RequestPage(BaseModel):
    requests: list[InferenceRequestRecord]
    total: int
    limit: int
    offset: int
