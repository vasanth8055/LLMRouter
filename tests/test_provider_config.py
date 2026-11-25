import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.config import LiveModelPricing
from app.providers.errors import ProviderConfigurationError
from app.providers.factory import ProviderFactory
from app.providers.model_registry import (
    CANONICAL_MODEL_IDS,
    GEMINI_25_FLASH,
    GEMINI_38_FLASH_API,
    MODEL_MAPPINGS,
)


def test_registry_has_explicit_mapping_for_every_canonical_model():
    assert tuple(MODEL_MAPPINGS) == CANONICAL_MODEL_IDS
    assert MODEL_MAPPINGS["deepseek-v3-0324"].api_model_setting == "deepseek_api_model"
    assert MODEL_MAPPINGS["gemini-2.5-flash"].api_model_setting == "gemini_flash_api_model"
    assert MODEL_MAPPINGS["gemini-2.5-pro"].api_model_setting == "gemini_pro_api_model"
    assert GEMINI_38_FLASH_API == "gemini-3.8-flash"
    assert GEMINI_38_FLASH_API not in CANONICAL_MODEL_IDS
    assert GEMINI_25_FLASH in CANONICAL_MODEL_IDS


def test_invalid_provider_mode_is_rejected(monkeypatch):
    module = importlib.import_module("app.providers.factory")
    monkeypatch.setattr(module.settings, "llm_provider_mode", "production-ish")
    with pytest.raises(ProviderConfigurationError, match="LLM_PROVIDER_MODE"):
        ProviderFactory()


def test_real_factory_uses_pricing_keyed_by_actual_api_model(monkeypatch):
    module = importlib.import_module("app.providers.factory")
    monkeypatch.setattr(module.settings, "deepseek_api_key", "secret-test-value")
    monkeypatch.setattr(module.settings, "deepseek_api_model", "ds-live-id")
    monkeypatch.setattr(module.settings, "live_model_pricing", {
        "ds-live-id": LiveModelPricing(input_price_per_million=3, output_price_per_million=7)
    })
    provider = ProviderFactory(mode="real").get_provider("deepseek-v3-0324")
    assert provider.api_model_name == "ds-live-id"
    assert provider.input_price_per_million == 3
    assert provider.output_price_per_million == 7
    assert "secret-test-value" not in repr(provider)


def test_missing_actual_model_pricing_keeps_live_cost_unavailable(monkeypatch):
    module = importlib.import_module("app.providers.factory")
    monkeypatch.setattr(module.settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(module.settings, "deepseek_api_model", "ds-live-unpriced")
    monkeypatch.setattr(module.settings, "live_model_pricing", {})
    provider = ProviderFactory(mode="real").get_provider("deepseek-v3-0324")
    assert provider.input_price_per_million is None
    assert provider.output_price_per_million is None


def test_config_validator_default_is_safe_and_never_prints_keys(monkeypatch, capsys):
    script = importlib.import_module("scripts.validate_provider_config")
    monkeypatch.setattr(script.settings, "llm_provider_mode", "real")
    monkeypatch.setattr(script.settings, "deepseek_api_key", "super-secret-key")
    monkeypatch.setattr(script.settings, "deepseek_api_model", "ds-live-id")
    monkeypatch.setattr(script.settings, "gemini_api_key", None)
    monkeypatch.setattr(script.settings, "gemini_flash_api_model", "gm-flash")
    monkeypatch.setattr(script.settings, "gemini_pro_api_model", "gm-pro")
    code = script.validate(["deepseek-v3-0324", "gemini-2.5-flash", "gemini-2.5-pro"], live=False)
    output = capsys.readouterr().out
    assert code == 1
    assert "super-secret-key" not in output
    assert "no network requests were made" not in output
    assert "GEMINI_API_KEY" in output


def test_config_validator_mock_requires_no_credentials_or_mapping(monkeypatch, capsys):
    script = importlib.import_module("scripts.validate_provider_config")
    monkeypatch.setattr(script.settings, "llm_provider_mode", "mock")
    monkeypatch.setattr(script.settings, "deepseek_api_key", None)
    monkeypatch.setattr(script.settings, "deepseek_api_model", None)
    assert script.validate(["deepseek-v3-0324"], live=False) == 0
    assert "no API required" in capsys.readouterr().out


def test_negative_live_pricing_is_invalid():
    with pytest.raises(ValueError):
        LiveModelPricing(input_price_per_million=-1, output_price_per_million=2)


def test_negative_expected_latency_is_invalid():
    from app.config import Settings

    with pytest.raises(ValueError):
        Settings(deepseek_expected_latency_ms=-1)


def test_step_five_gemini_latency_variable_names_are_accepted():
    from app.config import Settings

    configured = Settings(
        _env_file=None,
        GEMINI_FLASH_EXPECTED_LATENCY_MS=145,
        GEMINI_PRO_EXPECTED_LATENCY_MS=380,
    )
    assert configured.gemini_25_flash_expected_latency_ms == 145
    assert configured.gemini_25_pro_expected_latency_ms == 380


def test_blank_optional_latency_environment_values_are_ignored(monkeypatch):
    from app.config import Settings

    for name in (
        "DEEPSEEK_EXPECTED_LATENCY_MS",
        "GEMINI_FLASH_EXPECTED_LATENCY_MS",
        "GEMINI_PRO_EXPECTED_LATENCY_MS",
    ):
        monkeypatch.setenv(name, "")
    configured = Settings(_env_file=None)
    assert configured.deepseek_expected_latency_ms is None
    assert configured.gemini_25_flash_expected_latency_ms is None
    assert configured.gemini_25_pro_expected_latency_ms is None


def test_validator_mode_override_checks_real_mode_without_env_change(monkeypatch, capsys):
    script = importlib.import_module("scripts.validate_provider_config")
    monkeypatch.setattr(script.settings, "llm_provider_mode", "mock")
    monkeypatch.setattr(script.settings, "gemini_api_key", None)
    monkeypatch.setattr(script.settings, "gemini_flash_api_model", "gemini-2.5-flash")

    # Mode override allows validating real requirements while env is mock
    code = script.validate(["gemini-2.5-flash"], live=False, mode="real")
    output = capsys.readouterr().out
    assert code == 1
    assert "Provider mode: real" in output
    assert "ERROR: set GEMINI_API_KEY" in output


def test_validator_rejects_models_prefix_for_gemini_api_model(monkeypatch, capsys):
    script = importlib.import_module("scripts.validate_provider_config")
    monkeypatch.setattr(script.settings, "llm_provider_mode", "real")
    monkeypatch.setattr(script.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(script.settings, "gemini_flash_api_model", "models/gemini-2.5-flash")

    code = script.validate(["gemini-2.5-flash"], live=False)
    output = capsys.readouterr().out
    assert code == 1
    assert "must not include a leading 'models/' prefix" in output


def test_validator_warns_when_frozen_pro_fallback_is_not_configured(monkeypatch, capsys):
    script = importlib.import_module("scripts.validate_provider_config")
    monkeypatch.setattr(script.settings, "llm_provider_mode", "real")
    monkeypatch.setattr(script.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(script.settings, "gemini_flash_api_model", "gemini-2.5-flash")
    monkeypatch.setattr(script.settings, "gemini_pro_api_model", None)

    code = script.validate(["gemini-2.5-flash"], live=False)
    output = capsys.readouterr().out
    assert code == 0  # Flash is valid, but warning on fallback
    assert "WARNING: Frozen router fallback gemini-2.5-pro is not fully configured" in output


def test_validator_identifies_frozen_pro_fallback_role(monkeypatch, capsys):
    script = importlib.import_module("scripts.validate_provider_config")
    monkeypatch.setattr(script.settings, "llm_provider_mode", "mock")

    script.validate(["gemini-2.5-pro"], live=False)
    output = capsys.readouterr().out
    assert "gemini-2.5-pro [Frozen V2.1 Fallback]" in output
