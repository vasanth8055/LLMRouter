from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.config import LiveModelPricing, Settings
from app.providers.base import LLMProvider, LLMResponse
from scripts.live_validation import (
    estimate_prompt_costs,
    extract_choice_letter,
    inspect_env_capabilities,
    load_non_test_pool,
    run_mocked_deepseek_failover_check,
    run_probe,
    sample_non_test_prompts,
)


def test_non_test_pool_never_contains_burned_test_split():
    split_report = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "router_v21_split_report.json").read_text(encoding="utf-8")
    )
    burned = set(map(int, split_report["test_question_ids"]))
    pool_df, _ = load_non_test_pool()
    assert len(pool_df) == 2550
    assert set(pool_df["question_id"]).isdisjoint(burned)

    sampled = sample_non_test_prompts(pool_df, 150, seed=42)
    assert len(sampled) == 150
    assert set(sampled["question_id"]).isdisjoint(burned)


def test_inspect_env_capabilities_reports_gemini_only_vs_blocked_without_leaking_secrets():
    blocked = Settings(deepseek_api_key=None, gemini_api_key=None, gemini_flash_api_model="gemini-3.8-flash")
    out_blocked = inspect_env_capabilities(blocked)
    assert out_blocked["callable_models"] == []
    assert "BLOCKED" in out_blocked["validation_label"]

    gem_only = Settings(
        deepseek_api_key=None,
        gemini_api_key="super-secret-gemini-value-xyz",
        gemini_flash_api_model="gemini-2.5-flash",
        gemini_pro_api_model="gemini-2.5-pro",
        live_model_pricing={
            "gemini-2.5-flash": LiveModelPricing(input_price_per_million=0.30, output_price_per_million=2.50)
        },
    )
    out_gem = inspect_env_capabilities(gem_only)
    assert out_gem["validation_label"] == "Gemini-only live validation"
    assert out_gem["callable_models"] == ["gemini-2.5-flash", "gemini-2.5-pro"]
    assert "super-secret-gemini-value-xyz" not in json.dumps(out_gem)


def test_extract_choice_letter_handles_standard_formats():
    assert extract_choice_letter("Let's think step by step. The answer is (D).") == "D"
    assert extract_choice_letter("Final Answer: H") == "H"
    assert extract_choice_letter("Reasoning...\n**J**") == "J"
    assert extract_choice_letter("No letter option here 123") is None


def test_probe_runs_with_mocked_gemini_and_reports_mocked_deepseek_failover(tmp_path):
    class _FakeGem(LLMProvider):
        async def generate(self, prompt: str, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
            return LLMResponse(
                model_name="gemini-2.5-flash",
                response="Reasoning. The answer is (B).",
                input_tokens=100,
                output_tokens=20,
                latency_ms=42.0,
                estimated_cost_usd=0.00008,
            )

    failover_res = asyncio.run(run_mocked_deepseek_failover_check("Sample question", gemini_provider=_FakeGem()))
    assert failover_res["failover"] is True
    assert failover_res["attempted_models"] == ["deepseek-v3-0324", "gemini-2.5-flash"]
    assert failover_res["selected_model"] == "gemini-2.5-flash"

    probe_json = tmp_path / "probe.json"
    probe_jsonl = tmp_path / "probe.jsonl"
    out = asyncio.run(run_probe(probe_count=3, min_interval_s=0.0, probe_json=probe_json, probe_jsonl=probe_jsonl))
    assert out["probe_prompt_count"] == 3
    assert probe_json.exists()
    assert out["cost_estimate_n60"]["n_prompts"] == 60
    assert out["cost_estimate_n150"]["n_prompts"] == 150


def test_resumable_validation_checkpoints_skips_completed_and_tracks_cost_gap(tmp_path):
    from app.providers.factory import ProviderFactory
    from scripts.live_validation import run_validation

    gem_settings = Settings(
        deepseek_api_key=None,
        gemini_api_key="gm-dummy",
        gemini_flash_api_model="gemini-2.5-flash",
        gemini_pro_api_model="gemini-2.5-pro",
    )

    calls = {"n": 0}

    class _CountingGem(LLMProvider):
        def __init__(self, model_name: str, cost: float) -> None:
            self.model_name = model_name
            self.cost = cost

        async def generate(self, prompt: str, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
            calls["n"] += 1
            return LLMResponse(
                model_name=self.model_name,
                response="The answer is (B).",
                input_tokens=150,
                output_tokens=100,
                latency_ms=35.0,
                estimated_cost_usd=self.cost,
            )

    class _StubFactory(ProviderFactory):
        def __init__(self) -> None:
            super().__init__(mode="mock")

        def get_provider(self, model_name: str) -> LLMProvider:
            return _CountingGem(model_name, 0.005)

    jsonl_path = tmp_path / "val.jsonl"
    report_path = tmp_path / "val_report.json"

    with pytest.raises(RuntimeError, match="--approve-spend"):
        asyncio.run(
            run_validation(
                limit=4,
                min_interval_s=0.0,
                approve_spend=False,
                jsonl_path=jsonl_path,
                report_path=report_path,
                active_settings=gem_settings,
                factory=_StubFactory(),
            )
        )

    summary1 = asyncio.run(
        run_validation(
            limit=4,
            min_interval_s=0.0,
            approve_spend=True,
            mock_deepseek_failure=True,
            jsonl_path=jsonl_path,
            report_path=report_path,
            active_settings=gem_settings,
            factory=_StubFactory(),
        )
    )
    assert calls["n"] == 4
    assert summary1["n_recorded"] == 4
    assert summary1["total_billed_cost_usd"] == pytest.approx(0.02)
    assert summary1["cost_gap_usd"] is not None

    # Second run on same JSONL resumes without making new provider calls
    calls["n"] = 0
    summary2 = asyncio.run(
        run_validation(
            limit=4,
            min_interval_s=0.0,
            approve_spend=True,
            mock_deepseek_failure=True,
            jsonl_path=jsonl_path,
            report_path=report_path,
            active_settings=gem_settings,
            factory=_StubFactory(),
        )
    )
    assert calls["n"] == 0
    assert summary2["n_recorded"] == 4
