from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DEEPSEEK_V3_0324 = "deepseek-v3-0324"
GEMINI_25_FLASH = "gemini-2.5-flash"
GEMINI_25_PRO = "gemini-2.5-pro"
# Provider API identifier only; it is deliberately excluded from the frozen
# canonical research model list below.
GEMINI_38_FLASH_API = "gemini-3.8-flash"

CANONICAL_MODEL_IDS = (DEEPSEEK_V3_0324, GEMINI_25_FLASH, GEMINI_25_PRO)


@dataclass(frozen=True)
class ProviderModelMapping:
    canonical_id: str
    provider: Literal["deepseek", "gemini"]
    api_model_setting: str
    api_key_setting: str
    display_name: str
    expected_latency_setting: str
    benchmark_input_price_per_million: float
    benchmark_output_price_per_million: float


# This is the sole runtime mapping between research identities and the names
# used by provider APIs. Actual API IDs are supplied through configuration.
MODEL_MAPPINGS = {
    DEEPSEEK_V3_0324: ProviderModelMapping(
        canonical_id=DEEPSEEK_V3_0324,
        provider="deepseek",
        api_model_setting="deepseek_api_model",
        api_key_setting="deepseek_api_key",
        display_name="DeepSeek",
        expected_latency_setting="deepseek_expected_latency_ms",
        benchmark_input_price_per_million=0.25,
        benchmark_output_price_per_million=0.88,
    ),
    GEMINI_25_FLASH: ProviderModelMapping(
        canonical_id=GEMINI_25_FLASH,
        provider="gemini",
        api_model_setting="gemini_flash_api_model",
        api_key_setting="gemini_api_key",
        display_name="Gemini Flash",
        expected_latency_setting="gemini_25_flash_expected_latency_ms",
        benchmark_input_price_per_million=0.30,
        benchmark_output_price_per_million=2.50,
    ),
    GEMINI_25_PRO: ProviderModelMapping(
        canonical_id=GEMINI_25_PRO,
        provider="gemini",
        api_model_setting="gemini_pro_api_model",
        api_key_setting="gemini_api_key",
        display_name="Gemini Pro",
        expected_latency_setting="gemini_25_pro_expected_latency_ms",
        benchmark_input_price_per_million=1.25,
        benchmark_output_price_per_million=10.00,
    ),
}
