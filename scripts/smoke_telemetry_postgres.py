"""Exercise one mock /v1/chat request against a real PostgreSQL telemetry DB."""

from __future__ import annotations

import asyncio
import math
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _safe_database_diagnostic(error: BaseException) -> str:
    """Return driver metadata only; exception messages can include bind values."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        fields = [type(current).__name__]
        sqlstate = getattr(current, "sqlstate", None) or getattr(current, "pgcode", None)
        if sqlstate:
            fields.append(f"sqlstate={sqlstate}")
        diag = getattr(current, "diag", None)
        for field in ("schema_name", "table_name", "column_name", "constraint_name"):
            value = getattr(diag, field, None) if diag is not None else None
            if value and str(value).replace("_", "").isalnum():
                fields.append(f"{field}={value}")
        parts.append(" ".join(fields))
        current = getattr(current, "orig", None) or current.__cause__
    return " <- ".join(parts)


def main() -> int:
    database_url = os.environ.get("DATABASE_URL")
    provider_mode = os.environ.get("LLM_PROVIDER_MODE", "mock").strip().lower()
    if not database_url:
        print("Set DATABASE_URL to a local PostgreSQL database before running this smoke test.")
        return 2
    if provider_mode != "mock":
        print("Smoke test requires LLM_PROVIDER_MODE=mock; no request was sent.")
        return 2

    from fastapi.testclient import TestClient
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.main import app, router_service

    # Capture only exception class and PostgreSQL diagnostic identifiers for
    # this opt-in smoke run. Normal application logs remain message-free.
    repository = router_service.telemetry_service.repository
    original_save = repository.save

    async def save_with_safe_diagnostic(record):
        try:
            await original_save(record)
        except BaseException as exc:
            print(f"Safe telemetry diagnostic: {_safe_database_diagnostic(exc)}")
            raise

    repository.save = save_with_safe_diagnostic

    for method_name in ("analytics_summary", "analytics_models", "recent_requests", "request_page"):
        original_method = getattr(repository, method_name)

        async def method_with_safe_diagnostic(*args, _method=original_method, _name=method_name, **kwargs):
            try:
                return await _method(*args, **kwargs)
            except BaseException as exc:
                print(f"Safe {_name} diagnostic: {_safe_database_diagnostic(exc)}")
                raise

        setattr(repository, method_name, method_with_safe_diagnostic)

    request_id = f"telemetry-smoke-{uuid.uuid4()}"
    prompt = "Explain binary search in simple terms."
    with TestClient(app) as client:
        health_response = client.get("/health")
        response = client.post(
            "/v1/chat",
            json={"request_id": request_id, "prompt": prompt},
        )
        if health_response.status_code != 200 or health_response.json().get("status") != "ok":
            print("FastAPI health check did not report online.")
            return 1
        if response.status_code != 200:
            print(f"Mock /v1/chat failed with HTTP {response.status_code}.")
            return 1
        response_text = response.json().get("response")
        analytics_response = client.get("/v1/analytics/summary")
        models_response = client.get("/v1/analytics/models")
        recent_response = client.get("/v1/analytics/recent")
        history_response = client.get("/v1/requests", params={"search": request_id})
        results = (
            ("summary", analytics_response), ("models", models_response),
            ("recent", recent_response), ("history", history_response),
        )
        failed_results = [(name, result.status_code) for name, result in results if result.status_code != 200]
        if failed_results:
            print(f"Read-only analytics/history endpoint failed: {failed_results[0][0]} HTTP {failed_results[0][1]}.")
            return 1
        if analytics_response.json()["total_requests"] < 1:
            print("Analytics summary did not include the persisted request.")
            return 1
        if not any(row["request_id"] == request_id for row in recent_response.json()):
            print("Recent analytics did not include the persisted request.")
            return 1
        if not any(row["request_id"] == request_id for row in history_response.json()["requests"]):
            print("Request history did not include the persisted request.")
            return 1
        if not models_response.json():
            print("Model analytics returned no model data after a persisted inference.")
            return 1

    async def read_back() -> tuple[dict, set[str]]:
        engine = create_async_engine(database_url, pool_pre_ping=True)
        try:
            async with engine.connect() as connection:
                row = (
                    await connection.execute(
                        text(
                            "SELECT * "
                            "FROM inference_requests WHERE request_id = :request_id"
                        ),
                        {"request_id": request_id},
                    )
                ).mappings().one_or_none()
                columns = await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = 'inference_requests'"
                    )
                )
                return (dict(row) if row else {}, {item[0] for item in columns})
        finally:
            await engine.dispose()

    try:
        row, columns = asyncio.run(read_back())
    except Exception as exc:
        print(f"Telemetry readback failed ({type(exc).__name__}); connection details were omitted.")
        return 1

    required = {
        "request_id", "canonical_model", "provider_model", "predicted_quality",
        "input_tokens", "output_tokens", "estimated_cost_usd", "actual_latency_ms",
        "status", "prompt_hash",
    }
    if not row:
        print("No telemetry row was found for the mock request.")
        return 1
    if not required.issubset(row) or not required.issubset(columns):
        print("Telemetry row is missing one or more required fields.")
        return 1
    if "prompt" in columns or "response" in columns:
        print("Telemetry schema unexpectedly includes prompt/response text columns.")
        return 1
    expected = response.json()
    if row["request_id"] != expected["request_id"]:
        print("Persisted request ID does not match the API response.")
        return 1
    if row["canonical_model"] != expected["canonical_model"]:
        print("Persisted canonical model does not match the API response.")
        return 1
    if row["provider_model"] != expected["provider_model"]:
        print("Persisted provider model does not match the API response.")
        return 1
    if row["predicted_quality"] != expected["predicted_quality"]:
        print("Persisted predicted quality does not match the API response.")
        return 1
    if row["input_tokens"] != expected["input_tokens"] or row["output_tokens"] != expected["output_tokens"]:
        print("Persisted token counts do not match the API response.")
        return 1
    persisted_cost = row["estimated_cost_usd"]
    expected_cost = expected["estimated_cost_usd"]
    if persisted_cost is not None:
        persisted_cost = float(persisted_cost)
    if (persisted_cost is None) != (expected_cost is None) or (
        persisted_cost is not None
        and expected_cost is not None
        and not math.isclose(persisted_cost, expected_cost, rel_tol=1e-9, abs_tol=1e-10)
    ):
        print("Persisted estimated cost does not match the API response.")
        return 1
    if row["actual_latency_ms"] != expected["latency_ms"]:
        print("Persisted latency does not match the API response.")
        return 1
    unsafe_columns = {"prompt", "response", "api_key", "authorization", "password"}
    if unsafe_columns.intersection(columns):
        print("Telemetry schema unexpectedly includes sensitive text/credential columns.")
        return 1
    serialized = repr(row)
    if prompt in serialized or (response_text and response_text in serialized):
        print("Prompt or response text appeared in the selected telemetry fields.")
        return 1
    if not row["prompt_hash"] or len(row["prompt_hash"].strip()) != 64:
        print("Persisted prompt hash is invalid.")
        return 1
    if row["status"] != "success" or row["actual_latency_ms"] is None:
        print("Persisted status or latency is invalid.")
        return 1

    print("PostgreSQL telemetry smoke passed: one mock request persisted and read back.")
    print(
        "Demo route: "
        f"canonical_model={expected['canonical_model']}; provider_model={expected['provider_model']}; "
        f"predicted_quality={expected['predicted_quality']:.4f}; "
        f"latency_ms={expected['latency_ms']:.2f}."
    )
    print(
        "Verified request/model metadata, predicted quality, tokens, cost, latency, status, "
        "prompt hash, and absence of prompt/response/credential columns."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
