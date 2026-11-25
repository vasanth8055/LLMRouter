"""Phase 3 Resumable Live Validation & Probe Script for LLMRouter.

Rules enforced by design:
1. Never prints or writes raw API keys or raw prompt text (stores SHA-256 prompt_hash + question_id).
2. Excludes all 450 burned MMLU-Pro test_question_ids; samples strictly from train/validation splits.
3. Supports:
   - `--mode probe`: inspects .env key names, reports live-callable models, runs up to 3 prompts per
     available model, reports errors/rate limits, prints N=60 and N=150 cost estimates, and runs a
     mocked-DeepSeek failover smoke check when DeepSeek is unavailable.
   - `--mode validate`: resumable per-prompt JSONL checkpointing with request-count caps (60 default,
     up to 150) and inter-request throttling. Refuses paid live calls unless `--approve-spend` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings, settings  # noqa: E402
from app.providers.base import LLMProvider, LLMResponse  # noqa: E402
from app.providers.errors import ProviderConfigurationError, ProviderError, ProviderRateLimitError  # noqa: E402
from app.providers.factory import ProviderFactory  # noqa: E402
from app.providers.http_utils import REQUEST_RETRY_COUNT  # noqa: E402
from app.providers.model_registry import CANONICAL_MODEL_IDS, MODEL_MAPPINGS  # noqa: E402
from app.schemas import RouteRequest  # noqa: E402
from app.services.inference_service import InferenceService  # noqa: E402
from app.services.router_service import RouterService  # noqa: E402
from app.telemetry.repository import NoopTelemetryRepository  # noqa: E402
from app.telemetry.service import TelemetryService, build_prompt_hash  # noqa: E402

DATASET_PATH = ROOT / "data" / "llmrouter_mmlupro_test3000_long.csv"
SPLIT_REPORT_PATH = ROOT / "data" / "router_v21_split_report.json"
DEFAULT_PROBE_JSON = ROOT / "results" / "v5" / "live_probe.json"
DEFAULT_PROBE_JSONL = ROOT / "results" / "v5" / "live_probe.jsonl"
DEFAULT_VALIDATE_JSONL = ROOT / "results" / "v5" / "live_validation.jsonl"
DEFAULT_VALIDATE_REPORT = ROOT / "reports" / "phase3_live_validation.md"

PROVIDER_KEY_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "AWS_REGION",
)
MODEL_ID_ENV_VARS = (
    "DEEPSEEK_API_MODEL",
    "GEMINI_FLASH_API_MODEL",
    "GEMINI_PRO_API_MODEL",
)
PLACEHOLDER_PREFIXES = ("<", "your-", "changeme")


@dataclass(frozen=True)
class ModelCapability:
    canonical_model: str
    provider: str
    api_key_env: str
    api_key_present: bool
    api_model_env: str
    api_model_id: str | None
    live_callable: bool
    reason: str
    benchmark_input_price_per_million: float
    benchmark_output_price_per_million: float
    live_input_price_per_million: float | None
    live_output_price_per_million: float | None


def _is_non_placeholder_secret(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if stripped.startswith("<") and stripped.endswith(">"):
        return False
    return not any(lowered.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def inspect_dotenv_keys(dotenv_path: Path = ROOT / ".env") -> dict[str, str]:
    """Return status ('SET' or 'EMPTY_OR_UNSET') for known provider keys in .env without exposing values."""
    raw_map: dict[str, str] = {}
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            raw_map[key.strip()] = val.strip().strip('"').strip("'")
    statuses: dict[str, str] = {}
    for key_name in (*PROVIDER_KEY_ENV_VARS, *MODEL_ID_ENV_VARS):
        if key_name not in raw_map:
            statuses[key_name] = "NOT_IN_DOTENV"
        elif _is_non_placeholder_secret(raw_map[key_name]):
            statuses[key_name] = "SET"
        else:
            statuses[key_name] = "EMPTY_OR_PLACEHOLDER"
    return statuses


def inspect_env_capabilities(active_settings: Settings = settings) -> dict[str, Any]:
    """Inspect active settings and return key availability + callable canonical models (no secret values)."""
    dotenv_statuses = inspect_dotenv_keys()
    models_info: list[dict[str, Any]] = []
    callable_models: list[str] = []

    for canonical in CANONICAL_MODEL_IDS:
        mapping = MODEL_MAPPINGS[canonical]
        raw_key = getattr(active_settings, mapping.api_key_setting, None)
        raw_model = getattr(active_settings, mapping.api_model_setting, None)
        has_key = _is_non_placeholder_secret(raw_key)
        api_model_id = raw_model.strip() if _is_non_placeholder_secret(raw_model) else None

        reasons: list[str] = []
        if not has_key:
            reasons.append(f"missing {mapping.api_key_setting.upper()}")
        if not api_model_id:
            reasons.append(f"missing {mapping.api_model_setting.upper()}")
        live_callable = has_key and bool(api_model_id)
        if live_callable:
            callable_models.append(canonical)

        live_rate = active_settings.live_model_pricing.get(api_model_id) if api_model_id else None
        models_info.append(
            ModelCapability(
                canonical_model=canonical,
                provider=mapping.provider,
                api_key_env=mapping.api_key_setting.upper(),
                api_key_present=has_key,
                api_model_env=mapping.api_model_setting.upper(),
                api_model_id=api_model_id,
                live_callable=live_callable,
                reason="ready" if live_callable else "; ".join(reasons),
                benchmark_input_price_per_million=mapping.benchmark_input_price_per_million,
                benchmark_output_price_per_million=mapping.benchmark_output_price_per_million,
                live_input_price_per_million=(
                    float(live_rate.input_price_per_million) if live_rate else None
                ),
                live_output_price_per_million=(
                    float(live_rate.output_price_per_million) if live_rate else None
                ),
            ).__dict__
        )

    ds_ready = "deepseek-v3-0324" in callable_models
    gemini_ready = [m for m in callable_models if MODEL_MAPPINGS[m].provider == "gemini"]
    if len(callable_models) == len(CANONICAL_MODEL_IDS):
        validation_label = "3-model live validation"
    elif not ds_ready and gemini_ready:
        validation_label = "Gemini-only live validation"
    elif ds_ready and not gemini_ready:
        validation_label = "DeepSeek-only live validation"
    else:
        validation_label = "BLOCKED (no live models fully configured)"

    return {
        "dotenv_key_statuses": dotenv_statuses,
        "provider_keys_present": [
            k for k in PROVIDER_KEY_ENV_VARS if _is_non_placeholder_secret(getattr(active_settings, k.lower(), None))
        ],
        "callable_models": callable_models,
        "validation_label": validation_label,
        "models": models_info,
    }


def load_non_test_pool(
    dataset_path: Path = DATASET_PATH,
    split_report_path: Path = SPLIT_REPORT_PATH,
) -> tuple[pd.DataFrame, dict[int, dict[str, dict[str, float]]]]:
    """Load strictly train+validation MMLU-Pro prompts, asserting zero burned test_question_ids."""
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    train_ids = set(map(int, split_report["train_question_ids"]))
    val_ids = set(map(int, split_report["validation_question_ids"]))
    burned_test_ids = set(map(int, split_report["test_question_ids"]))
    assert not (train_ids & burned_test_ids), "Train/test split overlap detected!"
    assert not (val_ids & burned_test_ids), "Validation/test split overlap detected!"

    allowed_ids = train_ids | val_ids
    prompt_records: dict[int, dict[str, Any]] = {}
    offline_by_qid: dict[int, dict[str, dict[str, float]]] = {}

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = int(row["question_id"])
            if qid in burned_test_ids:
                continue
            if qid not in allowed_ids:
                continue
            if qid not in prompt_records:
                prompt_records[qid] = {
                    "question_id": qid,
                    "split": "validation" if qid in val_ids else "train",
                    "prompt": row["prompt"],
                    "ground_truth": str(row["ground_truth"]).strip().upper(),
                }
            offline_by_qid.setdefault(qid, {})[row["model"]] = {
                "score": int(float(row["score"])),
                "cost": float(row["cost"]),
                "input_tokens": float(row["input_tokens"]),
                "output_tokens": float(row["output_tokens"]),
            }

    df = pd.DataFrame(list(prompt_records.values())).sort_values("question_id").reset_index(drop=True)
    assert set(df["question_id"]).isdisjoint(burned_test_ids), "Burned test split leaked into pool!"
    assert len(df) == len(allowed_ids), f"Expected {len(allowed_ids)} non-test prompts, got {len(df)}"
    return df, offline_by_qid


def sample_non_test_prompts(
    pool_df: pd.DataFrame,
    count: int,
    *,
    seed: int = 42,
    prefer_validation: bool = False,
) -> pd.DataFrame:
    """Deterministically sample prompts stratified by split and answer option without burned test prompts."""
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(pool_df):
        raise ValueError(f"Requested {count} prompts but non-test pool has {len(pool_df)}")
    work = pool_df.copy()
    if prefer_validation:
        val_only = work[work["split"] == "validation"].copy()
        if len(val_only) >= count:
            work = val_only
    work["stratum"] = work["split"] + ":" + work["ground_truth"]
    rng = np.random.default_rng(seed)
    # Stratified proportional allocation by (split, ground_truth)
    strata_counts = work["stratum"].value_counts().sort_index()
    raw_alloc = strata_counts / strata_counts.sum() * count
    alloc = np.floor(raw_alloc).astype(int)
    remainder = int(count - alloc.sum())
    if remainder > 0:
        remainders = (raw_alloc - alloc).sort_values(ascending=False)
        for key in remainders.index[:remainder]:
            alloc[key] += 1

    parts = []
    for key, n_take in alloc.items():
        if n_take <= 0:
            continue
        sub = work[work["stratum"] == key]
        chosen_idx = rng.choice(sub.index.to_numpy(), size=int(n_take), replace=False)
        parts.append(work.loc[chosen_idx])
    sampled = pd.concat(parts, axis=0).sort_values("question_id").reset_index(drop=True)
    return sampled.drop(columns=["stratum"])


def extract_choice_letter(text: str) -> str | None:
    """Extract MMLU-Pro answer choice letter A-J from free-form model response."""
    if not text:
        return None
    patterns = (
        r"(?i)the\s+answer\s+is\s*\(?([A-J])\)?\b",
        r"(?i)final\s+answer\s*[:\-]\s*\(?([A-J])\)?\b",
        r"(?i)answer\s*[:\-]\s*\(?([A-J])\)?\b",
        r"(?i)\*\*([A-J])\*\*",
        r"(?m)^\s*\(?([A-J])\)?[\.\)]?\s*$",
    )
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].upper()
    tail_matches = re.findall(r"\b([A-J])\b", text[-80:])
    if tail_matches:
        return tail_matches[-1].upper()
    return None


def estimate_prompt_costs(
    sampled_df: pd.DataFrame,
    offline_by_qid: dict[int, dict[str, dict[str, float]]],
    router: RouterService | None = None,
) -> dict[str, Any]:
    """Estimate live USD cost using MMLU-Pro historical tokens + benchmark pricing and routed model mix."""
    per_model_totals = {m: 0.0 for m in CANONICAL_MODEL_IDS}
    per_model_counts = {m: 0 for m in CANONICAL_MODEL_IDS}
    routed_offline_total = 0.0

    for _, row in sampled_df.iterrows():
        qid = int(row["question_id"])
        per_model_stats = offline_by_qid[qid]
        for m in CANONICAL_MODEL_IDS:
            per_model_totals[m] += float(per_model_stats[m]["cost"])
        if router is not None:
            probs = router.predict_qualities(str(row["prompt"]))
            selected, _, _ = router._choose_model(probs, router.threshold, None)
            per_model_counts[selected] += 1
            routed_offline_total += float(per_model_stats[selected]["cost"])

    return {
        "n_prompts": len(sampled_df),
        "estimated_always_model_usd_benchmark_rates": {
            m: round(per_model_totals[m], 6) for m in CANONICAL_MODEL_IDS
        },
        "estimated_v21_routed_usd_benchmark_rates": round(routed_offline_total, 6) if router else None,
        "v21_predicted_route_counts": per_model_counts if router else None,
    }


def _compute_cost_from_rates(
    input_tokens: int | None,
    output_tokens: int | None,
    in_price_per_m: float,
    out_price_per_m: float,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    return (input_tokens / 1_000_000.0) * in_price_per_m + (
        output_tokens / 1_000_000.0
    ) * out_price_per_m


def _load_existing_checkpoint(jsonl_path: Path) -> dict[int, dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    if not jsonl_path.exists():
        return seen
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        qid = int(item["question_id"])
        seen[qid] = item
    return seen


def _append_jsonl(jsonl_path: Path, record: dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


async def run_mocked_deepseek_failover_check(
    prompt: str,
    *,
    gemini_provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Test runtime failover when DeepSeek raises a mocked ProviderRateLimitError."""

    class _AlwaysFailDeepSeek(LLMProvider):
        async def generate(
            self,
            prompt: str,
            max_tokens: int | None = None,
            temperature: float = 0.0,
        ) -> LLMResponse:
            raise ProviderRateLimitError("Mocked DeepSeek 429 for Phase 3 failover verification.")

    class _DummyFlash(LLMProvider):
        async def generate(
            self,
            prompt: str,
            max_tokens: int | None = None,
            temperature: float = 0.0,
        ) -> LLMResponse:
            return LLMResponse(
                model_name="gemini-2.5-flash",
                response="The answer is (B).",
                input_tokens=max(1, len(prompt.split())),
                output_tokens=8,
                latency_ms=12.0,
                estimated_cost_usd=0.000025,
            )

    class _FailoverFactory(ProviderFactory):
        def __init__(self) -> None:
            super().__init__(mode="mock")

        def get_provider(self, model_name: str) -> LLMProvider:
            if model_name == "deepseek-v3-0324":
                return _AlwaysFailDeepSeek()
            if gemini_provider is not None and model_name in ("gemini-2.5-flash", "gemini-2.5-pro"):
                return gemini_provider
            return _DummyFlash()

    router = RouterService(
        inference_service=InferenceService(_FailoverFactory()),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
    )
    with patch.object(
        router,
        "predict_qualities",
        return_value={"deepseek-v3-0324": 0.95, "gemini-2.5-flash": 0.92, "gemini-2.5-pro": 0.96},
    ):
        resp = await router.handle_async(RouteRequest(prompt=prompt, max_tokens=64, temperature=0.0))
    return {
        "note": "Mocked DeepSeek failover simulation (NOT a live DeepSeek call).",
        "selected_model": resp.selected_model,
        "attempted_models": resp.attempted_models,
        "failover": resp.failover,
        "policy_fallback": resp.policy_fallback,
        "retry_count": resp.retry_count,
    }


async def run_probe(
    *,
    probe_count: int = 3,
    min_interval_s: float = 0.5,
    probe_json: Path = DEFAULT_PROBE_JSON,
    probe_jsonl: Path = DEFAULT_PROBE_JSONL,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a 3-prompt probe per live-callable model, plus cost estimates & mocked failover smoke check."""
    caps = inspect_env_capabilities()
    pool_df, offline_by_qid = load_non_test_pool()
    router = RouterService(telemetry_service=TelemetryService(NoopTelemetryRepository()))
    probe_prompts = sample_non_test_prompts(pool_df, probe_count, seed=seed)
    est_60 = estimate_prompt_costs(sample_non_test_prompts(pool_df, 60, seed=seed), offline_by_qid, router)
    est_150 = estimate_prompt_costs(sample_non_test_prompts(pool_df, 150, seed=seed), offline_by_qid, router)

    probe_results: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    factory = ProviderFactory(mode="real") if caps["callable_models"] else None
    if probe_jsonl.exists():
        probe_jsonl.unlink()

    try:
        for canonical in CANONICAL_MODEL_IDS:
            cap = next(m for m in caps["models"] if m["canonical_model"] == canonical)
            if not cap["live_callable"]:
                model_summaries[canonical] = {
                    "live_callable": False,
                    "reason": cap["reason"],
                    "attempted": 0,
                    "succeeded": 0,
                    "rate_limited": 0,
                    "errors": [],
                }
                continue

            provider = factory.get_provider(canonical)
            latencies: list[float] = []
            rate_limited = 0
            errors: list[dict[str, Any]] = []
            succeeded = 0

            for idx, row in probe_prompts.iterrows():
                if idx > 0 and min_interval_s > 0:
                    await asyncio.sleep(min_interval_s)
                retry_box = [0]
                token = REQUEST_RETRY_COUNT.set(retry_box)
                started = time.perf_counter()
                try:
                    res = await provider.generate(str(row["prompt"]), max_tokens=512, temperature=0.0)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    latencies.append(elapsed_ms)
                    succeeded += 1
                    mapping = MODEL_MAPPINGS[canonical]
                    billed = res.estimated_cost_usd
                    benchmark_cost = _compute_cost_from_rates(
                        res.input_tokens,
                        res.output_tokens,
                        mapping.benchmark_input_price_per_million,
                        mapping.benchmark_output_price_per_million,
                    )
                    item = {
                        "canonical_model": canonical,
                        "provider_model": res.model_name,
                        "question_id": int(row["question_id"]),
                        "split": str(row["split"]),
                        "prompt_hash": build_prompt_hash(str(row["prompt"])),
                        "status": "success",
                        "latency_ms": round(elapsed_ms, 2),
                        "input_tokens": res.input_tokens,
                        "output_tokens": res.output_tokens,
                        "billed_cost_usd": billed,
                        "benchmark_rate_cost_usd": benchmark_cost,
                        "offline_cost_usd": offline_by_qid[int(row["question_id"])][canonical]["cost"],
                        "retry_count": retry_box[0],
                        "extracted_answer": extract_choice_letter(res.response),
                        "ground_truth": str(row["ground_truth"]),
                        "correct": extract_choice_letter(res.response) == str(row["ground_truth"]),
                    }
                except ProviderError as exc:
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if isinstance(exc, ProviderRateLimitError) or exc.status_code == 429:
                        rate_limited += 1
                    err_info = {
                        "question_id": int(row["question_id"]),
                        "error_type": type(exc).__name__,
                        "status_code": exc.status_code,
                        "retryable": exc.retryable,
                        "retry_count": retry_box[0],
                    }
                    errors.append(err_info)
                    item = {
                        "canonical_model": canonical,
                        "provider_model": cap["api_model_id"],
                        "question_id": int(row["question_id"]),
                        "split": str(row["split"]),
                        "prompt_hash": build_prompt_hash(str(row["prompt"])),
                        "status": "failed",
                        "latency_ms": round(elapsed_ms, 2),
                        "error_type": type(exc).__name__,
                        "status_code": exc.status_code,
                        "retry_count": retry_box[0],
                    }
                finally:
                    REQUEST_RETRY_COUNT.reset(token)

                probe_results.append(item)
                _append_jsonl(probe_jsonl, item)

            model_summaries[canonical] = {
                "live_callable": True,
                "api_model_id": cap["api_model_id"],
                "attempted": len(probe_prompts),
                "succeeded": succeeded,
                "rate_limited": rate_limited,
                "errors": errors,
                "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
            }
    finally:
        if factory is not None:
            await factory.close()

    failover_smoke = await run_mocked_deepseek_failover_check(str(probe_prompts.iloc[0]["prompt"]))
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_label": caps["validation_label"],
        "dotenv_key_statuses": caps["dotenv_key_statuses"],
        "callable_models": caps["callable_models"],
        "models": caps["models"],
        "probe_prompt_count": len(probe_prompts),
        "probe_question_ids": [int(x) for x in probe_prompts["question_id"].tolist()],
        "probe_model_summaries": model_summaries,
        "cost_estimate_n60": est_60,
        "cost_estimate_n150": est_150,
        "mocked_deepseek_failover_check": failover_smoke,
    }
    probe_json.parent.mkdir(parents=True, exist_ok=True)
    probe_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


async def run_validation(
    *,
    limit: int = 60,
    min_interval_s: float = 1.0,
    approve_spend: bool = False,
    mock_deepseek_failure: bool = False,
    jsonl_path: Path = DEFAULT_VALIDATE_JSONL,
    report_path: Path = DEFAULT_VALIDATE_REPORT,
    seed: int = 42,
    active_settings: Settings = settings,
    factory: ProviderFactory | None = None,
) -> dict[str, Any]:
    """Run resumable live validation over /v1/chat pipeline (`RouterService.handle_async`)."""
    if limit > 150:
        raise ValueError("Hard cap exceeded: limit must be <= 150 prompts")
    caps = inspect_env_capabilities(active_settings)
    pool_df, offline_by_qid = load_non_test_pool()
    sampled = sample_non_test_prompts(pool_df, limit, seed=seed)
    base_router = RouterService(telemetry_service=TelemetryService(NoopTelemetryRepository()))
    estimate = estimate_prompt_costs(sampled, offline_by_qid, base_router)

    if not caps["callable_models"]:
        raise RuntimeError(
            "No live models are configured in .env; cannot run live validation."
        )
    if not approve_spend:
        raise RuntimeError(
            f"Refusing live API calls without --approve-spend. Estimated benchmark-rate cost for "
            f"N={limit}: ${estimate['estimated_v21_routed_usd_benchmark_rates']:.6f}."
        )

    existing = _load_existing_checkpoint(jsonl_path)
    if factory is None:
        factory = ProviderFactory(mode="real")
    if mock_deepseek_failure and "deepseek-v3-0324" not in caps["callable_models"]:
        original_get_provider = factory.get_provider

        class _FailingDeepSeek(LLMProvider):
            async def generate(self, prompt: str, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
                raise ProviderConfigurationError("DEEPSEEK_API_KEY not configured (mocked failure for Gemini-only validation).")

        def _patched_get_provider(model_name: str) -> LLMProvider:
            if model_name == "deepseek-v3-0324":
                return _FailingDeepSeek()
            return original_get_provider(model_name)

        factory.get_provider = _patched_get_provider  # type: ignore[method-assign]

    router = RouterService(
        inference_service=InferenceService(factory),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
    )

    try:
        processed_now = 0
        for _, row in sampled.iterrows():
            qid = int(row["question_id"])
            if qid in existing:
                continue
            if processed_now > 0 and min_interval_s > 0:
                await asyncio.sleep(min_interval_s)
            prompt = str(row["prompt"])
            ground_truth = str(row["ground_truth"])
            probs = router.predict_qualities(prompt)
            predicted_first, _, _ = router._choose_model(probs, router.threshold, None)
            offline_row = offline_by_qid[qid]

            try:
                resp = await router.handle_async(
                    RouteRequest(
                        request_id=f"v5-live-{qid}",
                        prompt=prompt,
                        max_tokens=1024,
                        temperature=0.0,
                    )
                )
                extracted = extract_choice_letter(resp.response)
                mapping = MODEL_MAPPINGS[resp.selected_model]
                benchmark_rate_cost = _compute_cost_from_rates(
                    resp.input_tokens,
                    resp.output_tokens,
                    mapping.benchmark_input_price_per_million,
                    mapping.benchmark_output_price_per_million,
                )
                offline_cost = float(offline_row[resp.selected_model]["cost"])
                billed = resp.estimated_cost_usd
                gap = (billed - offline_cost) if billed is not None else None
                item = {
                    "question_id": qid,
                    "split": str(row["split"]),
                    "prompt_hash": build_prompt_hash(prompt),
                    "ground_truth": ground_truth,
                    "predicted_first_model": predicted_first,
                    "selected_model": resp.selected_model,
                    "provider_model": resp.provider_model,
                    "attempted_models": resp.attempted_models,
                    "failover": resp.failover,
                    "policy_fallback": resp.policy_fallback,
                    "retry_count": resp.retry_count,
                    "latency_ms": resp.latency_ms,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "billed_cost_usd": billed,
                    "benchmark_rate_cost_usd": benchmark_rate_cost,
                    "offline_cost_usd": offline_cost,
                    "cost_gap_usd": gap,
                    "extracted_answer": extracted,
                    "correct": extracted == ground_truth,
                    "status": "success",
                    "error_type": None,
                }
            except ProviderError as exc:
                item = {
                    "question_id": qid,
                    "split": str(row["split"]),
                    "prompt_hash": build_prompt_hash(prompt),
                    "ground_truth": ground_truth,
                    "predicted_first_model": predicted_first,
                    "selected_model": None,
                    "provider_model": None,
                    "attempted_models": [],
                    "failover": False,
                    "policy_fallback": False,
                    "retry_count": 0,
                    "latency_ms": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "billed_cost_usd": None,
                    "benchmark_rate_cost_usd": None,
                    "offline_cost_usd": float(offline_row[predicted_first]["cost"]),
                    "cost_gap_usd": None,
                    "extracted_answer": None,
                    "correct": False,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            _append_jsonl(jsonl_path, item)
            existing[qid] = item
            processed_now += 1
    finally:
        await factory.close()

    records = [existing[int(qid)] for qid in sampled["question_id"].tolist() if int(qid) in existing]
    successes = [r for r in records if r["status"] == "success"]
    latencies = [float(r["latency_ms"]) for r in successes if r["latency_ms"] is not None]
    billed_costs = [float(r["billed_cost_usd"]) for r in successes if r["billed_cost_usd"] is not None]
    offline_costs = [float(r["offline_cost_usd"]) for r in successes if r["offline_cost_usd"] is not None]
    summary = {
        "validation_label": caps["validation_label"],
        "n_target": limit,
        "n_recorded": len(records),
        "n_success": len(successes),
        "n_failed": len(records) - len(successes),
        "accuracy": float(np.mean([bool(r["correct"]) for r in successes])) if successes else None,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "total_input_tokens": int(sum(int(r["input_tokens"] or 0) for r in successes)),
        "total_output_tokens": int(sum(int(r["output_tokens"] or 0) for r in successes)),
        "total_billed_cost_usd": float(sum(billed_costs)) if billed_costs else None,
        "total_offline_cost_usd": float(sum(offline_costs)) if offline_costs else None,
        "cost_gap_usd": (float(sum(billed_costs)) - float(sum(offline_costs))) if billed_costs and offline_costs else None,
        "total_retries": int(sum(int(r.get("retry_count") or 0) for r in records)),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "validate"], default="probe")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--probe-count", type=int, default=3)
    parser.add_argument("--min-interval-s", type=float, default=1.0)
    parser.add_argument("--approve-spend", action="store_true")
    parser.add_argument("--mock-deepseek-failure", action="store_true")
    args = parser.parse_args()

    if args.mode == "probe":
        out = asyncio.run(run_probe(probe_count=args.probe_count, min_interval_s=args.min_interval_s))
        print(json.dumps(out, indent=2))
        return 0

    out = asyncio.run(
        run_validation(
            limit=args.limit,
            min_interval_s=args.min_interval_s,
            approve_spend=args.approve_spend,
            mock_deepseek_failure=args.mock_deepseek_failure,
        )
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
