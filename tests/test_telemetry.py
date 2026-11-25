from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict

from fastapi.testclient import TestClient

from app.main import app
from app.providers.errors import ProviderError
from app.providers.factory import ProviderFactory
from app.services.inference_service import InferenceService
from app.services.router_service import MODEL_IDS, RouterService
from app.telemetry.models import InferenceTelemetry
from app.telemetry.postgres import CREATE_TABLE_SQL, INSERT_TELEMETRY_SQL, PostgresTelemetryRepository
from app.telemetry.repository import NoopTelemetryRepository
from app.telemetry.service import TelemetryService, build_prompt_hash


class FakeRepository:
    def __init__(self, *, fail: bool = False) -> None:
        self.records: list[InferenceTelemetry] = []
        self.fail = fail
        self.closed = False

    async def initialize(self) -> None:
        return None

    async def save(self, record: InferenceTelemetry) -> None:
        if self.fail:
            raise RuntimeError("database unavailable; password=do-not-log")
        if not any(saved.request_id == record.request_id for saved in self.records):
            self.records.append(record)

    async def close(self) -> None:
        self.closed = True


def _mock_router(repository: FakeRepository) -> RouterService:
    return RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(repository),
    )


def test_prompt_hash_is_sha256_and_only_safe_prompt_metadata_is_persisted(monkeypatch):
    import app.main as main_module

    repository = FakeRepository()
    prompt = "private text that should not be persisted"
    monkeypatch.setattr(main_module, "router_service", _mock_router(repository))
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"request_id": "telemetry-success", "prompt": prompt})

    assert response.status_code == 200
    record = repository.records[0]
    assert record.status == "success"
    assert record.request_id == "telemetry-success"
    assert record.prompt_hash == hashlib.sha256(prompt.encode()).hexdigest()
    assert record.prompt_hash == build_prompt_hash(prompt)
    assert record.prompt_length == len(prompt)
    assert record.canonical_model == response.json()["canonical_model"]
    assert record.provider_model == response.json()["provider_model"]
    assert record.predicted_quality == response.json()["predicted_quality"]
    assert record.input_tokens == response.json()["input_tokens"]
    assert record.output_tokens == response.json()["output_tokens"]
    assert record.actual_latency_ms == response.json()["latency_ms"]
    assert record.estimated_cost_usd == response.json()["estimated_cost_usd"]
    assert record.routing_reason == response.json()["routing_reason"]
    assert prompt not in str(asdict(record))
    assert response.json()["response"] not in str(asdict(record))


class FailingInference:
    async def generate(self, model_name, prompt, max_tokens=None, temperature=0.0):
        raise ProviderError(
            f"provider failed; authorization: Bearer dummy-secret; prompt={prompt}"
        )


def test_failed_provider_request_persists_sanitized_error_and_measured_duration(monkeypatch):
    import app.main as main_module

    repository = FakeRepository()
    router = RouterService(
        inference_service=FailingInference(),
        telemetry_service=TelemetryService(repository),
    )
    monkeypatch.setattr(main_module, "router_service", router)
    prompt = "private failure prompt"
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"request_id": "telemetry-failure", "prompt": prompt})

    assert response.status_code == 502
    record = repository.records[0]
    assert record.status == "failed"
    assert record.error_type == "ProviderError"
    assert record.error_message is not None
    assert "dummy-secret" not in record.error_message
    assert prompt not in record.error_message
    assert record.prompt_hash == build_prompt_hash(prompt)
    assert record.canonical_model in MODEL_IDS
    assert record.actual_latency_ms is not None and record.actual_latency_ms >= 0
    assert record.input_tokens is None and record.output_tokens is None
    assert record.estimated_cost_usd is None
    assert "error_message" not in response.json()


def test_database_unavailable_does_not_fail_successful_inference(monkeypatch, caplog):
    import app.main as main_module

    repository = FakeRepository(fail=True)
    monkeypatch.setattr(main_module, "router_service", _mock_router(repository))
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"prompt": "still respond if database is down"})

    assert response.status_code == 200
    assert "password=do-not-log" not in caplog.text


class SlowRepository(FakeRepository):
    async def save(self, record: InferenceTelemetry) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.write_cancelled = True
            raise


def test_slow_telemetry_write_is_bounded_and_does_not_fail_inference(monkeypatch):
    import app.main as main_module

    repository = SlowRepository()
    telemetry = TelemetryService(repository, write_timeout_s=0.01)
    router = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=telemetry,
    )
    monkeypatch.setattr(main_module, "router_service", router)
    repository.write_cancelled = False
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"prompt": "database write stalls"})
    assert response.status_code == 200
    assert repository.write_cancelled
    assert repository.records == []


def test_duplicate_request_ids_are_ignored_by_database_schema_and_insert():
    assert "request_id TEXT NOT NULL UNIQUE" in CREATE_TABLE_SQL
    assert "ON CONFLICT (request_id) DO NOTHING" in INSERT_TELEMETRY_SQL
    lowered = CREATE_TABLE_SQL.lower()
    assert "prompt text" not in lowered
    assert "response text" not in lowered
    assert "api_key" not in lowered
    assert "authorization" not in lowered


def test_postgres_repository_fake_ignores_duplicate_request_ids(monkeypatch):
    import app.main as main_module

    repository = FakeRepository()
    monkeypatch.setattr(main_module, "router_service", _mock_router(repository))
    with TestClient(app) as client:
        first = client.post("/v1/chat", json={"request_id": "duplicate-id", "prompt": "first"})
        second = client.post("/v1/chat", json={"request_id": "duplicate-id", "prompt": "second"})
    assert first.status_code == second.status_code == 200
    assert len(repository.records) == 1


def test_postgres_urls_use_asyncpg_driver_without_exposing_credentials():
    assert PostgresTelemetryRepository._normalize_url("postgres://user:pass@host/db") == (
        "postgresql+asyncpg://user:pass@host/db"
    )
    assert PostgresTelemetryRepository._normalize_url("postgresql://user:pass@host/db") == (
        "postgresql+asyncpg://user:pass@host/db"
    )


def test_missing_database_url_uses_noop_repository_and_stays_usable():
    service = TelemetryService.from_database_url(None)
    assert isinstance(service.repository, NoopTelemetryRepository)
    sample = InferenceTelemetry(
        request_id="no-db", prompt_hash=build_prompt_hash("p"), prompt_length=1,
        canonical_model="gemini-2.5-flash", provider_model="gemini-2.5-flash",
        predicted_quality=0.9, model_mapping_warning=None, routing_reason="mock",
        input_tokens=1, output_tokens=1, estimated_cost_usd=None,
        actual_latency_ms=1.0, status="success",
    )
    asyncio.run(service.record(sample))
