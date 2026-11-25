from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveModelPricing(BaseModel):
    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)

class Settings(BaseSettings):
    app_name: str = "LLMRouter"
    router_mode: str = "v5-platt"
    llm_provider_mode: str = "mock"
    database_url: str | None = None
    frontend_origin: str = "http://127.0.0.1:5173"
    deepseek_api_key: str | None = None
    gemini_api_key: str | None = None
    # The benchmark's DeepSeek V3 identifier is not necessarily a current
    # live API model ID; require an explicit API model mapping in real mode.
    deepseek_api_model: str | None = None
    gemini_flash_api_model: str | None = None
    gemini_pro_api_model: str | None = None
    live_model_pricing: dict[str, LiveModelPricing] = Field(default_factory=dict)
    deepseek_expected_latency_ms: float | None = Field(default=None, ge=0)
    gemini_25_flash_expected_latency_ms: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "GEMINI_FLASH_EXPECTED_LATENCY_MS", "GEMINI_25_FLASH_EXPECTED_LATENCY_MS"
        ),
    )
    gemini_25_pro_expected_latency_ms: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "GEMINI_PRO_EXPECTED_LATENCY_MS", "GEMINI_25_PRO_EXPECTED_LATENCY_MS"
        ),
    )
    llm_request_timeout_s: float = Field(default=60.0, gt=0)
    llm_request_deadline_s: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_retry_base_delay_s: float = Field(default=0.25, ge=0.0)
    llm_retry_max_delay_s: float = Field(default=2.0, ge=0.0)
    small_model_name: str = "mock-small"
    medium_model_name: str = "mock-medium"
    large_model_name: str = "mock-large"
    openai_api_key: str | None = None
    aws_region: str | None = None
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

settings = Settings()
