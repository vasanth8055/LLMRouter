from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app, _analytics_repository
from app.providers.deepseek import DeepSeekProvider
from app.providers.errors import ProviderConfigurationError
from app.providers.factory import ProviderFactory
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.schemas import RouteRequest
from app.services.inference_service import InferenceService
from app.services.router_service import RouterService
from app.telemetry.models import InferenceTelemetry
from app.telemetry.postgres import (
    CREATE_INDEXES_SQL,
    CREATE_TABLE_SQL,
    INSERT_TELEMETRY_SQL,
    PostgresTelemetryRepository,
)
from app.telemetry.repository import NoopTelemetryRepository
from app.telemetry.service import TelemetryService, build_prompt_hash


# ---------------------------------------------------------------------------
# Task 1 & 2: Telemetry Schema Initialization and Idempotent Indexes
# ---------------------------------------------------------------------------

def test_database_index_definitions_are_idempotent_and_match_queries():
    assert len(CREATE_INDEXES_SQL) == 3
    index_names = {
        "idx_inference_requests_created_at_id",
        "idx_inference_requests_canonical_model",
        "idx_inference_requests_status",
    }
    for sql in CREATE_INDEXES_SQL:
        assert sql.startswith("CREATE INDEX IF NOT EXISTS")
        assert any(name in sql for name in index_names)
    assert any("(created_at DESC, id DESC)" in sql for sql in CREATE_INDEXES_SQL)
    assert any("(canonical_model)" in sql for sql in CREATE_INDEXES_SQL)
    assert any("(status)" in sql for sql in CREATE_INDEXES_SQL)


def test_schema_initialization_not_repeated_on_save():
    async def run():
        repo = PostgresTelemetryRepository("postgresql://user:pass@localhost:5432/testdb")
        assert not repo.is_initialized

        mock_connection = AsyncMock()
        mock_begin = MagicMock()
        mock_begin.__aenter__.return_value = mock_connection
        mock_begin.__aexit__.return_value = None

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_begin
        mock_engine.dispose = AsyncMock()
        repo._engine = mock_engine

        with patch.object(repo, "_get_engine", return_value=mock_engine):
            # 1. initialize() executes CREATE TABLE and all CREATE INDEX statements
            await repo.initialize()
            assert repo.is_initialized

            initial_calls = mock_connection.execute.call_count
            # 1 CREATE TABLE + 3 CREATE INDEX = 4 calls
            assert initial_calls == 1 + len(CREATE_INDEXES_SQL)

            # 2. save() executes only INSERT, never CREATE TABLE or CREATE INDEX
            sample_record = InferenceTelemetry(
                request_id="test-save-1",
                prompt_hash=build_prompt_hash("p"),
                prompt_length=1,
                canonical_model="deepseek-v3-0324",
                provider_model="deepseek-flash",
                predicted_quality=0.95,
                model_mapping_warning=None,
                routing_reason="reason",
                input_tokens=10,
                output_tokens=20,
                estimated_cost_usd=0.0001,
                actual_latency_ms=250.0,
                status="success",
            )
            await repo.save(sample_record)
            await repo.save(sample_record)

            # Total calls = initial 4 + 2 INSERT calls = 6
            assert mock_connection.execute.call_count == initial_calls + 2
            for call_args in mock_connection.execute.call_args_list[initial_calls:]:
                executed_sql = str(call_args[0][0])
                assert "CREATE TABLE" not in executed_sql
                assert "CREATE INDEX" not in executed_sql
                assert "INSERT INTO inference_requests" in executed_sql

            # 3. close() resets initialized state and disposes engine
            await repo.close()
            assert not repo.is_initialized
            mock_engine.dispose.assert_awaited_once()

    asyncio.run(run())


def test_save_on_uninitialized_repository_raises_explicit_error():
    async def run():
        repo = PostgresTelemetryRepository("postgresql://user:pass@localhost:5432/testdb")
        assert not repo.is_initialized

        sample_record = InferenceTelemetry(
            request_id="uninit-test",
            prompt_hash=build_prompt_hash("p"),
            prompt_length=1,
            canonical_model="gemini-2.5-pro",
            provider_model="gemini-2.5-pro",
            predicted_quality=0.92,
            model_mapping_warning=None,
            routing_reason="reason",
            input_tokens=5,
            output_tokens=5,
            estimated_cost_usd=0.0001,
            actual_latency_ms=500.0,
            status="success",
        )
        with pytest.raises(RuntimeError, match="PostgresTelemetryRepository is not initialized"):
            await repo.save(sample_record)

    asyncio.run(run())


def test_telemetry_initialization_failure_is_safe_and_explicit(caplog):
    async def run():
        repo = PostgresTelemetryRepository("postgresql://user:pass@localhost:5432/testdb")
        mock_engine = MagicMock()
        mock_engine.begin.side_effect = ConnectionRefusedError("Connection refused")

        with patch.object(repo, "_get_engine", return_value=mock_engine):
            service = TelemetryService(repo)
            assert not service.is_initialized

            # initialize() catches exception, logs warning, and does NOT claim persistence is ready
            await service.initialize()
            assert not service.is_initialized
            assert not repo.is_initialized
            assert "inference_telemetry_initialization_failed" in caplog.text

            # record() gracefully handles persistence failure without failing inference
            sample_record = InferenceTelemetry(
                request_id="record-uninit",
                prompt_hash=build_prompt_hash("p"),
                prompt_length=1,
                canonical_model="gemini-2.5-flash",
                provider_model="gemini-2.5-flash",
                predicted_quality=0.91,
                model_mapping_warning=None,
                routing_reason="reason",
                input_tokens=5,
                output_tokens=5,
                estimated_cost_usd=0.0001,
                actual_latency_ms=200.0,
                status="success",
            )
            await service.record(sample_record)
            assert "inference_telemetry_persistence_failed" in caplog.text

    asyncio.run(run())


def test_analytics_repository_checks_initialization_state(monkeypatch):
    import app.main as main_module
    from fastapi import HTTPException

    # Case A: Postgres repository configured but not initialized -> 503 Not Initialized
    uninit_repo = PostgresTelemetryRepository("postgresql://user:pass@localhost:5432/testdb")
    assert not uninit_repo.is_initialized
    mock_router = RouterService(telemetry_service=TelemetryService(uninit_repo))
    monkeypatch.setattr(main_module, "router_service", mock_router)

    with pytest.raises(HTTPException) as exc_info:
        _analytics_repository()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Telemetry database is not initialized."

    # Case B: Noop repository (unconfigured) -> 503 Not Configured
    noop_router = RouterService(telemetry_service=TelemetryService(NoopTelemetryRepository()))
    monkeypatch.setattr(main_module, "router_service", noop_router)

    with pytest.raises(HTTPException) as exc_info2:
        _analytics_repository()
    assert exc_info2.value.status_code == 503
    assert exc_info2.value.detail == "Telemetry database is not configured."


# ---------------------------------------------------------------------------
# Task 3: Reusable HTTP Clients for Real Providers
# ---------------------------------------------------------------------------

def test_mock_factory_does_not_create_http_client():
    factory = ProviderFactory(mode="mock")
    assert factory.http_client is None
    provider = factory.get_provider("gemini-2.5-flash")
    assert isinstance(provider, MockProvider)
    assert factory.http_client is None


def test_real_factory_shares_single_http_client_across_multiple_providers(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-ds-key")
    monkeypatch.setattr(settings, "deepseek_api_model", "deepseek-flash")
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "gemini_flash_api_model", "gemini-flash-live")
    monkeypatch.setattr(settings, "gemini_pro_api_model", "gemini-pro-live")

    factory = ProviderFactory(mode="real")
    assert factory.http_client is None

    p1 = factory.get_provider("deepseek-v3-0324")
    assert isinstance(p1, DeepSeekProvider)
    shared_client = factory.http_client
    assert isinstance(shared_client, httpx.AsyncClient)
    assert not shared_client.is_closed
    assert p1.http_client is shared_client

    p2 = factory.get_provider("gemini-2.5-flash")
    assert isinstance(p2, GeminiProvider)
    assert p2.http_client is shared_client

    p3 = factory.get_provider("gemini-2.5-pro")
    assert isinstance(p3, GeminiProvider)
    assert p3.http_client is shared_client

    # Verify all providers reference the exact same HTTP client instance
    assert p1.http_client is p2.http_client is p3.http_client is shared_client


def test_shared_http_client_closed_during_shutdown(monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "deepseek_api_key", "test-ds-key")
        monkeypatch.setattr(settings, "deepseek_api_model", "deepseek-flash")

        factory = ProviderFactory(mode="real")
        _provider = factory.get_provider("deepseek-v3-0324")
        client = factory.http_client
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed

        await factory.close()
        assert client.is_closed
        assert factory.http_client is None

    asyncio.run(run())


def test_router_service_startup_and_close_lifecycle(monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "deepseek_api_key", "test-ds-key")
        monkeypatch.setattr(settings, "deepseek_api_model", "deepseek-flash")

        factory = ProviderFactory(mode="real")
        inference = InferenceService(factory)
        noop_telemetry = TelemetryService(NoopTelemetryRepository())
        router = RouterService(inference_service=inference, telemetry_service=noop_telemetry)

        # Startup creates the shared client
        await router.startup()
        client = factory.http_client
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed

        # Close cleans up all resources
        await router.close()
        assert client.is_closed
        assert factory.http_client is None

    asyncio.run(run())


def test_lifespan_manages_resources_cleanly():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
