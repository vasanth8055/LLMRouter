from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.providers.model_registry import CANONICAL_MODEL_IDS, MODEL_MAPPINGS
from app.schemas import (
    RouteExplainRequest,
    RouteExplainResponse,
    RouteRequest,
    RouteResponse,
    RoutingConfigResponse,
    RuntimeStatus,
)
from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.services.router_service import RouterService, RoutingConstraintError, RouterModelError
from app.telemetry.api_schemas import AnalyticsSummary, InferenceRequestRecord, ModelAnalytics, RequestPage
from app.telemetry.postgres import PostgresTelemetryRepository

logger = logging.getLogger(__name__)

router_service = RouterService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        await router_service.startup()
    except Exception as exc:
        logger.warning("application_startup_warning", extra={"error_type": type(exc).__name__})
    try:
        yield
    finally:
        await router_service.close()


app = FastAPI(
    title="LLMRouter",
    version="0.1.0",
    description="Cost-aware LLM routing gateway. Predicted quality is not actual correctness.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _analytics_repository() -> PostgresTelemetryRepository:
    repository = router_service.telemetry_service.repository
    if not isinstance(repository, PostgresTelemetryRepository):
        raise HTTPException(status_code=503, detail="Telemetry database is not configured.")
    if not repository.is_initialized:
        raise HTTPException(status_code=503, detail="Telemetry database is not initialized.")
    return repository


async def _read_telemetry(operation):
    try:
        return await operation()
    except HTTPException:
        raise
    except Exception as exc:
        # SQL/connection details can include infrastructure information.
        logger.warning("telemetry_read_failed", extra={"error_type": type(exc).__name__})
        raise HTTPException(status_code=503, detail="Telemetry data is temporarily unavailable.") from exc


@app.exception_handler(ProviderConfigurationError)
async def provider_configuration_error(_request: Request, exc: ProviderConfigurationError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ProviderAuthenticationError)
async def provider_authentication_error(_request: Request, exc: ProviderAuthenticationError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(ProviderRateLimitError)
async def provider_rate_limit_error(_request: Request, exc: ProviderRateLimitError):
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(ProviderTimeoutError)
async def provider_timeout_error(_request: Request, exc: ProviderTimeoutError):
    return JSONResponse(status_code=504, content={"detail": str(exc)})


@app.exception_handler(RoutingConstraintError)
async def routing_constraint_error(_request: Request, exc: RoutingConstraintError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RouterModelError)
async def router_model_error(_request: Request, exc: RouterModelError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ProviderError)
async def provider_error(_request: Request, exc: ProviderError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "service": "llmrouter"}


@app.get("/v1/runtime", response_model=RuntimeStatus)
def runtime_status():
    mode = settings.llm_provider_mode.strip().lower()
    return {"provider_mode": mode if mode in {"mock", "real"} else "unknown"}


@app.get("/v1/routing/config", response_model=RoutingConfigResponse)
def routing_config():
    mode = settings.llm_provider_mode.strip().lower()
    provider_mode = mode if mode in {"mock", "real"} else "unknown"
    costs = router_service.training_average_costs
    cheapest_cost = min(costs.values())
    models = []
    for canonical in CANONICAL_MODEL_IDS:
        mapping = MODEL_MAPPINGS[canonical]
        avg_cost = float(costs[canonical])
        configured_provider_model = (
            canonical
            if provider_mode == "mock"
            else getattr(settings, mapping.api_model_setting, None)
        )
        models.append(
            {
                "canonical_model": canonical,
                "display_name": mapping.display_name,
                "provider": mapping.provider,
                "training_average_cost_usd": avg_cost,
                "relative_cost_vs_cheapest": round(avg_cost / cheapest_cost, 2),
                "benchmark_input_price_per_million": mapping.benchmark_input_price_per_million,
                "benchmark_output_price_per_million": mapping.benchmark_output_price_per_million,
                "provider_model": configured_provider_model or None,
            }
        )
    return {
        "policy_id": (
            "V5 Platt Cost-Aware Router"
            if router_service.artifact_version == "v5-platt"
            else "V2.1 Cost-Aware Router"
        ),
        "quality_floor": router_service.threshold,
        "fallback_model": router_service.fallback_model,
        "provider_mode": provider_mode,
        "telemetry_table": "inference_requests",
        "models": models,
    }


@app.post("/v1/route/explain", response_model=RouteExplainResponse)
def explain_route(request: RouteExplainRequest):
    """Explain the route without persisting or returning prompt contents."""
    probabilities = router_service.predict_qualities(request.prompt)
    selected, reason, _ = router_service._choose_model(probabilities, request.min_quality, None)
    floor = (
        router_service.threshold
        if request.min_quality is None
        else max(router_service.threshold, request.min_quality)
    )
    return {
        "prompt_length": len(request.prompt),
        "quality_floor": floor,
        "candidates": [
            {
                "canonical_model": model,
                "predicted_quality": probabilities[model],
                "eligible": probabilities[model] >= floor,
            }
            for model in probabilities
        ],
        "selected_model": selected,
        "fallback": reason.startswith("No eligible model met predicted quality"),
        "routing_reason": reason,
    }


@app.get("/v1/analytics/summary", response_model=AnalyticsSummary, response_model_exclude_unset=True)
async def analytics_summary():
    repository = _analytics_repository()
    return await _read_telemetry(repository.analytics_summary)


@app.get("/v1/analytics/models", response_model=list[ModelAnalytics])
async def analytics_models():
    repository = _analytics_repository()
    return await _read_telemetry(repository.analytics_models)


@app.get("/v1/analytics/recent", response_model=list[InferenceRequestRecord], response_model_exclude_unset=True)
async def analytics_recent():
    repository = _analytics_repository()
    return await _read_telemetry(lambda: repository.recent_requests(limit=8))


@app.get("/v1/requests", response_model=RequestPage, response_model_exclude_unset=True)
async def requests(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    model: str | None = None,
    status: str | None = Query(default=None, pattern="^(success|failed)$"),
    search: str | None = Query(default=None, max_length=100),
):
    repository = _analytics_repository()
    total, rows = await _read_telemetry(
        lambda: repository.request_page(
            limit=limit, offset=offset, model=model, status=status, search=search
        )
    )
    return {"requests": rows, "total": total, "limit": limit, "offset": offset}

@app.post("/v1/chat", response_model=RouteResponse)
@app.post("/v1/route", response_model=RouteResponse, include_in_schema=False)
async def chat(request: RouteRequest) -> RouteResponse:
    return await router_service.handle_async(request)


