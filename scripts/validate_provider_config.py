"""Validate provider configuration; --live performs metadata-only API checks."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.providers.model_registry import CANONICAL_MODEL_IDS, MODEL_MAPPINGS  # noqa: E402

DEEPSEEK_MODELS_URL = "https://api.deepseek.com/models"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


FROZEN_FALLBACK_MODEL = "gemini-2.5-pro"


def validate(selected_models: list[str], live: bool, mode: str | None = None) -> int:
    effective_mode = (mode or settings.llm_provider_mode).strip().lower()
    if effective_mode not in {"mock", "real"}:
        print("ERROR: LLM_PROVIDER_MODE must be 'mock' or 'real'.")
        return 1
    print(f"Provider mode: {effective_mode}")
    has_errors = False
    for canonical in selected_models:
        mapping = MODEL_MAPPINGS[canonical]
        api_model = getattr(settings, mapping.api_model_setting)
        api_key = getattr(settings, mapping.api_key_setting)
        api_model = api_model.strip() if isinstance(api_model, str) else ""
        role_label = " [Frozen V2.1 Fallback]" if canonical == FROZEN_FALLBACK_MODEL else ""
        if effective_mode == "mock":
            print(f"{canonical}{role_label} -> {api_model or '<not configured>'} (mock mode; no API required)")
        else:
            print(f"{canonical}{role_label} -> {api_model or '<not configured>'}")
            if not api_key:
                print(f"  ERROR: set {mapping.api_key_setting.upper()}.")
                has_errors = True
            if not api_model:
                print(f"  ERROR: set {mapping.api_model_setting.upper()} to the exact provider API model ID.")
                has_errors = True
            elif api_model.startswith("models/"):
                print(f"  ERROR: {mapping.api_model_setting.upper()} must not include a leading 'models/' prefix (use '{api_model.removeprefix('models/')}', not '{api_model}').")
                has_errors = True
            elif any(char.isspace() for char in api_model) or api_model.startswith("/"):
                print(f"  ERROR: {mapping.api_model_setting.upper()} must be a non-empty model ID without whitespace or a leading slash.")
                has_errors = True
            elif api_model == canonical:
                print("  Mapping explicitly matches canonical identity.")
            else:
                print("  Mapping differs from canonical identity; live quality transfer is not established.")

        pricing = settings.live_model_pricing.get(api_model) if api_model else None
        expected_latency = getattr(settings, mapping.expected_latency_setting)
        print(f"  Live pricing: {'configured' if pricing else 'unset'}")
        print(f"  Expected latency: {'configured' if expected_latency is not None else 'unset'}")

    if effective_mode == "real" and FROZEN_FALLBACK_MODEL not in selected_models:
        fallback_mapping = MODEL_MAPPINGS[FROZEN_FALLBACK_MODEL]
        fb_key = getattr(settings, fallback_mapping.api_key_setting)
        fb_model = getattr(settings, fallback_mapping.api_model_setting)
        if not fb_key or not fb_model:
            print(f"  WARNING: Frozen router fallback {FROZEN_FALLBACK_MODEL} is not fully configured; unroutable prompts will fail at runtime unless configured.")

    if effective_mode == "mock":
        print("Mock mode requires no provider keys or live model mappings.")
        if live:
            print("ERROR: --live requires real mode (set LLM_PROVIDER_MODE=real or use --mode real).")
            return 1
        return 0
    if has_errors:
        return 1
    if not live:
        print("Configuration is structurally valid. No network requests were made.")
        return 0
    return asyncio.run(_validate_live(selected_models))


async def _validate_live(selected_models: list[str]) -> int:
    failed = False
    timeout = httpx.Timeout(settings.llm_request_timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for canonical in selected_models:
            mapping = MODEL_MAPPINGS[canonical]
            api_model = getattr(settings, mapping.api_model_setting).strip()
            api_key = getattr(settings, mapping.api_key_setting)
            try:
                if mapping.provider == "deepseek":
                    response = await client.get(
                        DEEPSEEK_MODELS_URL,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if response.status_code >= 400:
                        print(f"ERROR: DeepSeek model-list validation failed (HTTP {response.status_code}).")
                        failed = True
                        continue
                    payload = response.json()
                    found = any(item.get("id") == api_model for item in payload.get("data", []))
                else:
                    response = await client.get(
                        GEMINI_MODELS_URL,
                        headers={"x-goog-api-key": api_key},
                    )
                    if response.status_code >= 400:
                        print(f"ERROR: Gemini model-list validation failed (HTTP {response.status_code}).")
                        failed = True
                        continue
                    payload = response.json()
                    found = any(
                        item.get("name") == f"models/{api_model}"
                        and "generateContent" in item.get("supportedGenerationMethods", [])
                        for item in payload.get("models", [])
                    )
                if found:
                    print(f"OK: {canonical} maps to an available {mapping.display_name} generate-capable API model.")
                else:
                    print(f"ERROR: configured API model for {canonical} was not found as a supported model.")
                    failed = True
            except (httpx.RequestError, ValueError, AttributeError, TypeError):
                print(f"ERROR: {mapping.display_name} model-list validation could not be completed.")
                failed = True
    print("Live validation used metadata-only GET requests; no inference request was sent.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CANONICAL_MODEL_IDS, help="Validate one canonical model; default checks all.")
    parser.add_argument("--mode", choices=["mock", "real"], default=None, help="Validate configuration as mock or real mode (default: configured LLM_PROVIDER_MODE).")
    parser.add_argument("--live", action="store_true", help="Explicitly query provider model metadata endpoints.")
    args = parser.parse_args()
    selected = [args.model] if args.model else list(CANONICAL_MODEL_IDS)
    return validate(selected, args.live, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
