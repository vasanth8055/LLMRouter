from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app


class FakeReadRepository:
    async def analytics_summary(self):
        return {
            "total_requests": 1,
            "average_cost_usd": 0.001,
            "average_latency_ms": 23.5,
            "error_rate": 0.0,
        }

    async def analytics_models(self):
        return [{
            "canonical_model": "gemini-2.5-flash",
            "request_count": 1,
            "average_cost_usd": 0.001,
            "total_cost_usd": 0.001,
            "average_latency_ms": 23.5,
            "average_predicted_quality": 0.91,
        }]

    def _row(self):
        return {
            "request_id": "safe-request-id",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "prompt_hash": "a" * 64,
            "canonical_model": "gemini-2.5-flash",
            "provider_model": "gemini-3.8-flash",
            "predicted_quality": 0.91,
            "input_tokens": 5,
            "output_tokens": 8,
            "estimated_cost_usd": 0.001,
            "actual_latency_ms": 23.5,
            "status": "success",
        }

    async def recent_requests(self, limit=8):
        return [self._row()][:limit]

    async def request_page(self, **kwargs):
        return 1, [self._row()]


def test_telemetry_read_endpoints_return_aggregates_and_safe_request_fields(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_analytics_repository", lambda: FakeReadRepository())
    with TestClient(app) as client:
        summary = client.get("/v1/analytics/summary")
        models = client.get("/v1/analytics/models")
        recent = client.get("/v1/analytics/recent")
        page = client.get("/v1/requests", params={"model": "gemini-2.5-flash", "status": "success"})

    assert summary.status_code == models.status_code == recent.status_code == page.status_code == 200
    assert summary.json()["total_requests"] == 1
    assert models.json()[0]["canonical_model"] == "gemini-2.5-flash"
    assert recent.json()[0]["provider_model"] == "gemini-3.8-flash"
    assert page.json()["requests"][0]["prompt_hash"] == "a" * 64
    for response in (recent, page):
        record = response.json() if response is recent else response.json()["requests"]
        records = record if isinstance(record, list) else [record]
        for item in records:
            assert "prompt" not in item
            assert "response" not in item


def test_telemetry_reads_return_503_when_database_is_not_configured(monkeypatch):
    import app.main as main_module

    from fastapi import HTTPException

    def unavailable():
        raise HTTPException(status_code=503, detail="Telemetry database is not configured.")

    monkeypatch.setattr(main_module, "_analytics_repository", unavailable)
    with TestClient(app) as client:
        response = client.get("/v1/analytics/summary")
    assert response.status_code == 503
    assert response.json()["detail"] == "Telemetry database is not configured."


def test_requests_filters_are_validated():
    with TestClient(app) as client:
        response = client.get("/v1/requests", params={"status": "pending"})
    assert response.status_code == 422


def test_unconfigured_placeholder_database_url_uses_noop_and_returns_503(monkeypatch):
    import app.main as main_module
    from app.services.router_service import RouterService
    from app.telemetry.repository import NoopTelemetryRepository
    from app.telemetry.service import TelemetryService

    service = TelemetryService.from_database_url(
        "postgresql+asyncpg://<user>:<password>@localhost:<port>/<database>"
    )
    assert isinstance(service.repository, NoopTelemetryRepository)

    mock_router = RouterService(telemetry_service=service)
    monkeypatch.setattr(main_module, "router_service", mock_router)
    with TestClient(app) as client:
        response = client.get("/v1/analytics/summary")
    assert response.status_code == 503
    assert response.json()["detail"] == "Telemetry database is not configured."


def test_empty_telemetry_repository_returns_valid_empty_state(monkeypatch):
    import app.main as main_module

    class EmptyRepository:
        is_initialized = True

        async def analytics_summary(self):
            return {
                "total_requests": 0,
                "average_cost_usd": None,
                "average_latency_ms": None,
                "error_rate": 0.0,
            }

        async def analytics_models(self):
            return []

        async def recent_requests(self, limit=8):
            return []

        async def request_page(self, **kwargs):
            return 0, []

    monkeypatch.setattr(main_module, "_analytics_repository", lambda: EmptyRepository())
    with TestClient(app) as client:
        summary = client.get("/v1/analytics/summary")
        models = client.get("/v1/analytics/models")
        recent = client.get("/v1/analytics/recent")
        page = client.get("/v1/requests")

    assert summary.status_code == models.status_code == recent.status_code == page.status_code == 200
    assert summary.json() == {
        "total_requests": 0,
        "average_cost_usd": None,
        "average_latency_ms": None,
        "error_rate": 0.0,
    }
    assert models.json() == []
    assert recent.json() == []
    assert page.json() == {"requests": [], "total": 0, "limit": 10, "offset": 0}


def test_uninitialized_postgres_repository_returns_503_not_initialized(monkeypatch):
    import app.main as main_module
    from app.services.router_service import RouterService
    from app.telemetry.postgres import PostgresTelemetryRepository
    from app.telemetry.service import TelemetryService

    repo = PostgresTelemetryRepository("postgresql+asyncpg://mockuser:mockpass@localhost:5432/mockdb")
    assert not repo.is_initialized

    telemetry = TelemetryService(repo)
    mock_router = RouterService(telemetry_service=telemetry)
    monkeypatch.setattr(main_module, "router_service", mock_router)

    with TestClient(app) as client:
        response = client.get("/v1/analytics/summary")
    assert response.status_code == 503
    assert response.json()["detail"] == "Telemetry database is not initialized."


def test_database_unavailable_during_read_returns_503_temporarily_unavailable(monkeypatch):
    import app.main as main_module

    class FailingReadRepository:
        is_initialized = True

        async def analytics_summary(self):
            raise ConnectionRefusedError("Database connection refused: credentials=hidden")

    monkeypatch.setattr(main_module, "_analytics_repository", lambda: FailingReadRepository())
    with TestClient(app) as client:
        response = client.get("/v1/analytics/summary")
    assert response.status_code == 503
    assert response.json()["detail"] == "Telemetry data is temporarily unavailable."


