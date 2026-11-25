from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.providers.errors import ProviderError, ProviderTimeoutError
from app.providers.factory import MOCK_MODEL_PROFILES
from app.providers.http_utils import REQUEST_DEADLINE_AT, REQUEST_RETRY_COUNT
from app.providers.model_registry import CANONICAL_MODEL_IDS, MODEL_MAPPINGS
from app.schemas import RouteRequest, RouteResponse
from app.services.inference_service import InferenceService
from app.services.routing_policy import route_by_quality_and_cost
from app.telemetry.models import InferenceTelemetry
from app.telemetry.service import TelemetryService, build_prompt_hash, sanitize_error_message

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
V21_MODEL_FILE = ROOT / "models" / "router_v21_tfidf.joblib"
V5_PLATT_MODEL_FILE = ROOT / "models" / "router_v5_platt.joblib"
ROUTER_ARTIFACT_FILES = {
    "baseline": V21_MODEL_FILE,
    "v21": V21_MODEL_FILE,
    "v5-platt": V5_PLATT_MODEL_FILE,
    "platt": V5_PLATT_MODEL_FILE,
}
MODEL_IDS = CANONICAL_MODEL_IDS
FEATURE_COLUMNS = ("prompt", "model")
MODEL_PROVIDER_NAMES = {model: f"{mapping.provider.title()}Provider" for model, mapping in MODEL_MAPPINGS.items()}


class RouterModelError(RuntimeError):
    """The frozen V2.1 runtime artifact is missing or incompatible."""


class RoutingConstraintError(ValueError):
    """The requested runtime constraints cannot be evaluated safely."""


class RouterService:
    """Runtime routing with the frozen Router V2.1 quality predictor."""

    def __init__(
        self,
        inference_service: InferenceService | None = None,
        model_package: dict[str, Any] | None = None,
        telemetry_service: TelemetryService | None = None,
        request_deadline_s: float | None = None,
        router_artifact: str | None = None,
    ) -> None:
        self.inference_service = inference_service or InferenceService()
        self.telemetry_service = telemetry_service or TelemetryService.from_database_url(settings.database_url)
        self._request_deadline_s = request_deadline_s
        selected_key = (router_artifact if router_artifact is not None else settings.router_mode).strip().lower()
        artifact_path = ROUTER_ARTIFACT_FILES.get(selected_key)
        if model_package is None and artifact_path is None:
            raise RouterModelError(f"Unsupported router_mode '{selected_key}'.")
        chosen_path = artifact_path or V21_MODEL_FILE
        try:
            package = model_package if model_package is not None else joblib.load(chosen_path)
            predictor = package["predictor"]
            features = package["metadata"]["features"]
            threshold = float(package["selected_threshold"])
            costs = {model: float(value) for model, value in package["training_average_costs"].items()}
            fallback = str(package["fallback_model"])
            platt_params = package.get("platt_params")
            if features != list(FEATURE_COLUMNS):
                raise ValueError("unexpected feature schema")
            if set(costs) != set(MODEL_IDS) or fallback not in MODEL_IDS:
                raise ValueError("unexpected model cost or fallback metadata")
            if platt_params is None:
                if not np.isclose(threshold, 0.90):
                    raise ValueError("frozen threshold is not 0.90")
            else:
                if set(platt_params) != set(MODEL_IDS):
                    raise ValueError("platt_params must cover all canonical models")
                if not (0.50 <= threshold <= 1.0):
                    raise ValueError("calibrated threshold out of range")
            if set(predictor.named_steps["classifier"].classes_) != {0, 1}:
                raise ValueError("predictor does not expose binary correctness probabilities")
        except Exception as exc:
            raise RouterModelError(
                f"Could not load the router artifact at {chosen_path.name}."
            ) from exc
        self.predictor = predictor
        self.platt_params: dict[str, dict[str, float]] | None = platt_params
        self.artifact_version = "v5-platt" if platt_params is not None else "v21"
        self.artifact_label = "V5 Platt Router" if platt_params is not None else "frozen V2.1"
        self.artifact_path = chosen_path
        self.features = tuple(features)
        self.threshold = threshold
        self.training_average_costs = costs
        self.fallback_model = fallback

    def predict_qualities(self, prompt: str) -> dict[str, float]:
        """Apply the artifact's original (prompt, model) feature construction (+ optional Platt scaling)."""
        features = pd.DataFrame(
            [{"prompt": prompt, "model": model} for model in MODEL_IDS],
            columns=self.features,
        )
        classes = list(self.predictor.named_steps["classifier"].classes_)
        positive_index = classes.index(1)
        values = self.predictor.predict_proba(features)[:, positive_index]
        if len(values) != len(MODEL_IDS) or not np.isfinite(values).all():
            raise RouterModelError(f"{self.artifact_label} returned invalid quality probabilities.")
        if ((values < 0) | (values > 1)).any():
            raise RouterModelError(f"{self.artifact_label} returned probabilities outside [0, 1].")
        if self.platt_params is not None:
            calibrated = []
            for model, raw_p in zip(MODEL_IDS, values):
                p = float(np.clip(raw_p, 1e-6, 1.0 - 1e-6))
                logit = np.log(p / (1.0 - p))
                params = self.platt_params[model]
                z = float(params["coef"]) * logit + float(params["intercept"])
                calibrated.append(float(1.0 / (1.0 + np.exp(-z))))
            return dict(zip(MODEL_IDS, calibrated))
        return {model: float(probability) for model, probability in zip(MODEL_IDS, values)}

    def _latency_estimates(self) -> dict[str, float | None]:
        if settings.llm_provider_mode.strip().lower() == "mock":
            return {model: float(MOCK_MODEL_PROFILES[model].latency_ms) for model in MODEL_IDS}
        return {
            model: getattr(settings, mapping.expected_latency_setting)
            for model, mapping in MODEL_MAPPINGS.items()
        }

    def _prepare_routing(
        self,
        probabilities: dict[str, float],
        min_quality: float | None,
        max_latency_ms: float | None,
    ) -> tuple[list[str], str, bool]:
        # A request may raise the quality floor, or omit it to use the active
        # artifact's validated threshold, but cannot lower it below self.threshold.
        effective_min_quality = self.threshold if min_quality is None else float(min_quality)
        if effective_min_quality < self.threshold or effective_min_quality > 1.0:
            raise RoutingConstraintError(
                f"min_quality ({effective_min_quality:.2f}) must be between {self.threshold:.2f} and 1.00; "
                f"it cannot lower the {self.artifact_label} policy floor of {self.threshold:.2f}."
            )
        threshold = max(self.threshold, effective_min_quality)
        routing_probabilities = dict(probabilities)
        latency_estimates = self._latency_estimates()
        excluded_by_latency: set[str] = set()
        if max_latency_ms is not None:
            if any(latency_estimates[model] is None for model in MODEL_IDS):
                raise RoutingConstraintError(
                    "max_latency_ms requires configured expected latency values for every model in real mode."
                )
            for model in MODEL_IDS:
                if float(latency_estimates[model]) > max_latency_ms:
                    routing_probabilities[model] = 0.0
                    excluded_by_latency.add(model)

        decision = route_by_quality_and_cost(
            routing_probabilities,
            self.training_average_costs,
            threshold,
            self.fallback_model,
        )
        if (
            decision["fallback"]
            and max_latency_ms is not None
            and self.fallback_model in excluded_by_latency
        ):
            raise RoutingConstraintError(
                "No model met predicted quality and latency constraints, and the frozen V2.1 fallback exceeds max_latency_ms."
            )
        eligible = sorted(
            (model for model, prob in routing_probabilities.items() if float(prob) >= threshold),
            key=lambda model: (float(self.training_average_costs[model]), model),
        )
        policy_fallback = not bool(eligible)
        candidate_order = list(eligible)
        if self.fallback_model not in candidate_order and self.fallback_model not in excluded_by_latency:
            candidate_order.append(self.fallback_model)
        if policy_fallback:
            reason = (
                f"No eligible model met predicted quality {threshold:.2f}; "
                f"used frozen V2.1 fallback {self.fallback_model}."
            )
        else:
            reason = (
                f"Selected the lowest training-cost model with predicted quality "
                f"at least {threshold:.2f} under the frozen V2.1 policy."
            )
        return candidate_order, reason, policy_fallback

    def _choose_model(
        self,
        probabilities: dict[str, float],
        min_quality: float | None,
        max_latency_ms: float | None,
    ) -> tuple[str, str, float]:
        candidate_order, reason, _ = self._prepare_routing(probabilities, min_quality, max_latency_ms)
        selected = candidate_order[0]
        return selected, reason, float(probabilities[selected])

    async def handle_async(self, request: RouteRequest) -> RouteResponse:
        request_id = request.request_id or str(uuid.uuid4())
        probabilities = self.predict_qualities(request.prompt)
        candidate_order, base_reason, policy_fallback = self._prepare_routing(
            probabilities, request.min_quality, request.max_latency_ms
        )
        attempted_models: list[str] = []
        last_error: ProviderError | None = None
        last_failed_model = candidate_order[0]
        result = None
        selected_model = candidate_order[0]
        inference_started = time.perf_counter()
        effective_deadline_s = (
            float(settings.llm_request_deadline_s)
            if self._request_deadline_s is None
            else float(self._request_deadline_s)
        )
        deadline_at = time.monotonic() + effective_deadline_s
        retry_counter = [0]
        deadline_token = REQUEST_DEADLINE_AT.set(deadline_at)
        retry_token = REQUEST_RETRY_COUNT.set(retry_counter)
        try:
            for candidate in candidate_order:
                remaining = deadline_at - time.monotonic()
                if remaining <= 0:
                    last_error = ProviderTimeoutError(
                        f"Request exceeded overall deadline of {effective_deadline_s:.1f}s."
                    )
                    break
                attempted_models.append(candidate)
                try:
                    result = await asyncio.wait_for(
                        self.inference_service.generate(
                            candidate,
                            request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                        ),
                        timeout=remaining,
                    )
                    selected_model = candidate
                    break
                except asyncio.TimeoutError:
                    last_error = ProviderTimeoutError(
                        f"Request exceeded overall deadline of {effective_deadline_s:.1f}s."
                    )
                    last_failed_model = candidate
                    logger.warning(
                        "provider_attempt_failed",
                        extra={
                            "request_id": request_id,
                            "candidate_model": candidate,
                            "provider": MODEL_PROVIDER_NAMES[candidate],
                            "error_type": type(last_error).__name__,
                            "attempt_index": len(attempted_models),
                        },
                    )
                    break
                except ProviderError as exc:
                    last_error = exc
                    last_failed_model = candidate
                    logger.warning(
                        "provider_attempt_failed",
                        extra={
                            "request_id": request_id,
                            "candidate_model": candidate,
                            "provider": MODEL_PROVIDER_NAMES[candidate],
                            "error_type": type(exc).__name__,
                            "attempt_index": len(attempted_models),
                        },
                    )
                    if time.monotonic() >= deadline_at:
                        break
        finally:
            REQUEST_RETRY_COUNT.reset(retry_token)
            REQUEST_DEADLINE_AT.reset(deadline_token)

        retry_count = retry_counter[0]
        failover = len(attempted_models) > 1
        if result is None:
            assert last_error is not None
            predicted_quality = float(probabilities[last_failed_model])
            mapping = MODEL_MAPPINGS[last_failed_model]
            configured_provider_model = (
                last_failed_model
                if settings.llm_provider_mode.strip().lower() == "mock"
                else getattr(settings, mapping.api_model_setting, None)
            )
            failure_reason = (
                f"{base_reason} All attempted providers failed ({', '.join(attempted_models)})."
                if failover
                else base_reason
            )
            telemetry = InferenceTelemetry(
                request_id=request_id,
                prompt_hash=build_prompt_hash(request.prompt),
                prompt_length=len(request.prompt),
                canonical_model=last_failed_model,
                provider_model=configured_provider_model or None,
                predicted_quality=predicted_quality,
                model_mapping_warning=None,
                routing_reason=failure_reason,
                input_tokens=None,
                output_tokens=None,
                estimated_cost_usd=None,
                actual_latency_ms=(time.perf_counter() - inference_started) * 1000,
                status="failed",
                error_type=type(last_error).__name__,
                error_message=sanitize_error_message(
                    str(last_error),
                    request.prompt,
                    (settings.deepseek_api_key or "", settings.gemini_api_key or ""),
                ),
                attempted_models=list(attempted_models),
                failover=failover,
                policy_fallback=policy_fallback,
                retry_count=retry_count,
            )
            await self.telemetry_service.record(telemetry)
            logger.warning(
                "provider_inference_failed",
                extra={
                    "request_id": request_id,
                    "selected_model": last_failed_model,
                    "predicted_quality": predicted_quality,
                    "provider": MODEL_PROVIDER_NAMES[last_failed_model],
                    "error_type": type(last_error).__name__,
                    "attempted_models": attempted_models,
                    "failover": failover,
                    "retry_count": retry_count,
                },
            )
            raise last_error

        predicted_quality = float(probabilities[selected_model])
        provider_name = MODEL_PROVIDER_NAMES[selected_model]
        reason = (
            f"{base_reason} Runtime failover routed to {selected_model} after failure on "
            f"{', '.join(attempted_models[:-1])}."
            if failover
            else base_reason
        )
        logger.info(
            "inference_completed",
            extra={
                "request_id": request_id,
                "selected_model": selected_model,
                "predicted_quality": predicted_quality,
                "provider": provider_name,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
                "attempted_models": attempted_models,
                "failover": failover,
                "retry_count": retry_count,
            },
        )
        mapping_warning = None
        if result.model_name != selected_model:
            mapping_warning = (
                f"Live API model '{result.model_name}' differs from frozen benchmark identity "
                f"'{selected_model}'; benchmark quality estimates do not establish live-model quality."
            )
        telemetry = InferenceTelemetry(
            request_id=request_id,
            prompt_hash=build_prompt_hash(request.prompt),
            prompt_length=len(request.prompt),
            canonical_model=selected_model,
            provider_model=result.model_name,
            predicted_quality=predicted_quality,
            model_mapping_warning=mapping_warning,
            routing_reason=reason,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            actual_latency_ms=result.latency_ms,
            status="success",
            attempted_models=list(attempted_models),
            failover=failover,
            policy_fallback=policy_fallback,
            retry_count=retry_count,
        )
        await self.telemetry_service.record(telemetry)
        return RouteResponse(
            request_id=request_id,
            selected_model=selected_model,
            canonical_model=selected_model,
            predicted_quality=predicted_quality,
            routing_reason=reason,
            response=result.response,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            estimated_cost_usd=result.estimated_cost_usd,
            provider_model=result.model_name,
            model_mapping_warning=mapping_warning,
            attempted_models=list(attempted_models),
            failover=failover,
            policy_fallback=policy_fallback,
            retry_count=retry_count,
        )

    def handle(self, request: RouteRequest) -> RouteResponse:
        """Synchronous compatibility entry point for existing Python callers."""
        return asyncio.run(self.handle_async(request))

    async def startup(self) -> None:
        await self.telemetry_service.initialize()
        if hasattr(self.inference_service, "startup"):
            await self.inference_service.startup()

    async def close(self) -> None:
        try:
            await self.telemetry_service.close()
        finally:
            if hasattr(self.inference_service, "close"):
                await self.inference_service.close()
