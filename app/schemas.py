from pydantic import BaseModel, Field

class RouteRequest(BaseModel):
    request_id: str | None = None
    prompt: str = Field(min_length=1)
    max_latency_ms: float | None = Field(default=None, gt=0)
    min_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum predicted quality floor. Defaults to the active artifact's validated policy floor (0.90 for V2.1, 0.85 for V5 Platt); values below the active floor are rejected with HTTP 422.",
    )
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    
class RouteResponse(BaseModel):
    request_id: str
    selected_model: str
    canonical_model: str
    predicted_quality: float
    routing_reason: str
    response: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    estimated_cost_usd: float | None
    provider_model: str
    model_mapping_warning: str | None = None
    attempted_models: list[str] = Field(default_factory=list)
    failover: bool = False
    policy_fallback: bool = False
    retry_count: int = 0


class RoutingModelConfig(BaseModel):
    canonical_model: str
    display_name: str
    provider: str
    training_average_cost_usd: float
    relative_cost_vs_cheapest: float
    benchmark_input_price_per_million: float
    benchmark_output_price_per_million: float
    provider_model: str | None = None


class RoutingConfigResponse(BaseModel):
    policy_id: str
    quality_floor: float
    fallback_model: str
    provider_mode: str
    telemetry_table: str
    models: list[RoutingModelConfig]

class RouteExplainRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    min_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum predicted quality floor. Defaults to the active artifact's validated policy floor (0.90 for V2.1, 0.85 for V5 Platt); values below the active floor are rejected with HTTP 422.",
    )

class RouteCandidateExplanation(BaseModel):
    canonical_model: str
    predicted_quality: float
    eligible: bool

class RouteExplainResponse(BaseModel):
    prompt_length: int
    quality_floor: float
    candidates: list[RouteCandidateExplanation]
    selected_model: str
    fallback: bool
    routing_reason: str

class RuntimeStatus(BaseModel):
    provider_mode: str
