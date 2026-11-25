from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers.deepseek import DeepSeekProvider
from app.providers.errors import ProviderAuthenticationError, ProviderError, ProviderRateLimitError, ProviderTimeoutError
from app.providers.factory import ProviderFactory
from app.providers.gemini import GeminiProvider
from app.providers.model_registry import DEEPSEEK_V3_0324, GEMINI_25_FLASH, GEMINI_25_PRO
from app.services.inference_service import InferenceService
from app.services.router_service import RouterService
from app.telemetry.models import InferenceTelemetry
from app.telemetry.postgres import (
    MIGRATION_SQL,
    PostgresTelemetryRepository,
    _decode_attempted_models,
    _normalize_record_row,
)
from app.telemetry.service import TelemetryService


class RecordingTelemetryRepo:
    is_initialized = True

    def __init__(self) -> None:
        self.records: list[InferenceTelemetry] = []

    async def initialize(self) -> None:
        return None

    async def save(self, record: InferenceTelemetry) -> None:
        self.records.append(record)

    async def close(self) -> None:
        return None


def _mock_transport(handlers):
    idx = {"i": 0}

    def dispatch(request: httpx.Request) -> httpx.Response:
        pos = idx["i"]
        idx["i"] += 1
        fn = handlers[min(pos, len(handlers) - 1)]
        return fn(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(dispatch))


@pytest.mark.parametrize("transient_status", [429, 500, 503])
def test_deepseek_retries_on_429_and_5xx_then_succeeds(transient_status: int):
    attempts = {"count": 0}
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            return httpx.Response(transient_status, json={"error": {"message": "transient"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "recovered"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DeepSeekProvider(
                api_key="ds-key",
                api_model_name="deepseek-chat",
                http_client=client,
                max_retries=2,
                retry_base_delay_s=0.1,
                retry_max_delay_s=1.0,
                sleep_fn=fake_sleep,
            )
            return await provider.generate("hello")

    res = asyncio.run(run())
    assert res.response == "recovered"
    assert attempts["count"] == 3
    assert len(delays) == 2
    assert 0.05 <= delays[0] <= 0.10
    assert 0.10 <= delays[1] <= 0.20


def test_gemini_retries_on_timeout_then_succeeds():
    attempts = {"count": 0}
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "gemini recovered"}]}}],
                "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 4},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = GeminiProvider(
                GEMINI_25_FLASH,
                api_model_name="gemini-2.5-flash",
                api_key="gm-key",
                http_client=client,
                max_retries=2,
                retry_base_delay_s=0.1,
                retry_max_delay_s=1.0,
                sleep_fn=fake_sleep,
            )
            return await provider.generate("hello")

    res = asyncio.run(run())
    assert res.response == "gemini recovered"
    assert attempts["count"] == 2
    assert len(delays) == 1


@pytest.mark.parametrize(
    ("status", "exc_cls"),
    [
        (400, ProviderError),
        (401, ProviderAuthenticationError),
        (403, ProviderAuthenticationError),
        (404, ProviderError),
    ],
)
def test_providers_do_not_retry_on_non_429_4xx(status: int, exc_cls: type[ProviderError]):
    ds_attempts = {"count": 0}
    gm_attempts = {"count": 0}
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def ds_handler(request: httpx.Request) -> httpx.Response:
        ds_attempts["count"] += 1
        return httpx.Response(status, json={"error": {"message": "client error"}})

    def gm_handler(request: httpx.Request) -> httpx.Response:
        gm_attempts["count"] += 1
        return httpx.Response(status, json={"error": {"message": "client error"}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(ds_handler)) as ds_client:
            ds = DeepSeekProvider(
                api_key="ds-key",
                api_model_name="deepseek-chat",
                http_client=ds_client,
                max_retries=2,
                sleep_fn=fake_sleep,
            )
            with pytest.raises(exc_cls):
                await ds.generate("hello")

        async with httpx.AsyncClient(transport=httpx.MockTransport(gm_handler)) as gm_client:
            gm = GeminiProvider(
                GEMINI_25_FLASH,
                api_model_name="gemini-2.5-flash",
                api_key="gm-key",
                http_client=gm_client,
                max_retries=2,
                sleep_fn=fake_sleep,
            )
            with pytest.raises(exc_cls):
                await gm.generate("hello")

    asyncio.run(run())
    assert ds_attempts["count"] == 1
    assert gm_attempts["count"] == 1
    assert delays == []


class _MockHttpFactory(ProviderFactory):
    def __init__(self, providers_by_model: dict[str, object]) -> None:
        super().__init__(mode="mock")
        self._providers = providers_by_model

    def get_provider(self, model_name: str):
        return self._providers[model_name]


def test_runtime_failover_chain_through_api_records_attempts_and_telemetry(monkeypatch):
    import app.main as main_module

    async def no_sleep(_delay: float) -> None:
        return None

    ds_calls = {"n": 0}
    fl_calls = {"n": 0}
    pr_calls = {"n": 0}

    def ds_500(_req):
        ds_calls["n"] += 1
        return httpx.Response(500, json={"error": {"message": "ds down"}})

    def fl_503(_req):
        fl_calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "flash overloaded"}})

    def pr_200(_req):
        pr_calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "pro failover answer"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            },
        )

    async def run_test():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(ds_500)) as ds_c,
            httpx.AsyncClient(transport=httpx.MockTransport(fl_503)) as fl_c,
            httpx.AsyncClient(transport=httpx.MockTransport(pr_200)) as pr_c,
        ):
            factory = _MockHttpFactory(
                {
                    DEEPSEEK_V3_0324: DeepSeekProvider(
                        api_key="k",
                        api_model_name="deepseek-chat",
                        http_client=ds_c,
                        max_retries=2,
                        retry_base_delay_s=0.0,
                        sleep_fn=no_sleep,
                    ),
                    GEMINI_25_FLASH: GeminiProvider(
                        GEMINI_25_FLASH,
                        api_model_name="gemini-2.5-flash",
                        api_key="k",
                        http_client=fl_c,
                        max_retries=1,
                        retry_base_delay_s=0.0,
                        sleep_fn=no_sleep,
                    ),
                    GEMINI_25_PRO: GeminiProvider(
                        GEMINI_25_PRO,
                        api_model_name="gemini-2.5-pro",
                        api_key="k",
                        http_client=pr_c,
                        max_retries=2,
                        retry_base_delay_s=0.0,
                        sleep_fn=no_sleep,
                    ),
                }
            )
            repo = RecordingTelemetryRepo()
            service = RouterService(
                inference_service=InferenceService(factory),
                telemetry_service=TelemetryService(repo),
            )
            monkeypatch.setattr(
                service,
                "predict_qualities",
                lambda _prompt: {DEEPSEEK_V3_0324: 0.95, GEMINI_25_FLASH: 0.92, GEMINI_25_PRO: 0.97},
            )
            monkeypatch.setattr(main_module, "router_service", service)
            with TestClient(app) as client:
                resp = client.post("/v1/chat", json={"request_id": "failover-1", "prompt": "test prompt"})
            return resp, repo

    resp, repo = asyncio.run(run_test())
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_model"] == GEMINI_25_PRO
    assert body["attempted_models"] == [DEEPSEEK_V3_0324, GEMINI_25_FLASH, GEMINI_25_PRO]
    assert body["failover"] is True
    assert body["policy_fallback"] is False
    assert body["retry_count"] == 3
    assert body["response"] == "pro failover answer"
    assert ds_calls["n"] == 3  # 1 initial + 2 retries
    assert fl_calls["n"] == 2  # 1 initial + 1 retry
    assert pr_calls["n"] == 1
    assert repo.records[0].attempted_models == [DEEPSEEK_V3_0324, GEMINI_25_FLASH, GEMINI_25_PRO]
    assert repo.records[0].failover is True
    assert repo.records[0].retry_count == 3
    assert repo.records[0].status == "success"


def test_all_providers_failing_returns_clean_http_error_and_records_attempts(monkeypatch):
    import app.main as main_module

    async def no_sleep(_delay: float) -> None:
        return None

    def fail_429(_req):
        return httpx.Response(429, json={"error": {"message": "rate limit"}})

    def fail_502(_req):
        return httpx.Response(502, json={"error": {"message": "upstream bad gateway"}})

    async def run_test():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(fail_429)) as ds_c,
            httpx.AsyncClient(transport=httpx.MockTransport(fail_502)) as pr_c,
        ):
            factory = _MockHttpFactory(
                {
                    DEEPSEEK_V3_0324: DeepSeekProvider(
                        api_key="k",
                        api_model_name="deepseek-chat",
                        http_client=ds_c,
                        max_retries=1,
                        retry_base_delay_s=0.0,
                        sleep_fn=no_sleep,
                    ),
                    GEMINI_25_FLASH: GeminiProvider(
                        GEMINI_25_FLASH,
                        api_model_name="gemini-2.5-flash",
                        api_key="k",
                        http_client=pr_c,
                        max_retries=1,
                        retry_base_delay_s=0.0,
                        sleep_fn=no_sleep,
                    ),
                    GEMINI_25_PRO: GeminiProvider(
                        GEMINI_25_PRO,
                        api_model_name="gemini-2.5-pro",
                        api_key="k",
                        http_client=pr_c,
                        max_retries=1,
                        retry_base_delay_s=0.0,
                        sleep_fn=no_sleep,
                    ),
                }
            )
            repo = RecordingTelemetryRepo()
            service = RouterService(
                inference_service=InferenceService(factory),
                telemetry_service=TelemetryService(repo),
            )
            monkeypatch.setattr(
                service,
                "predict_qualities",
                lambda _prompt: {DEEPSEEK_V3_0324: 0.94, GEMINI_25_FLASH: 0.80, GEMINI_25_PRO: 0.85},
            )
            monkeypatch.setattr(main_module, "router_service", service)
            with TestClient(app) as client:
                resp = client.post("/v1/chat", json={"request_id": "all-fail-1", "prompt": "test prompt"})
            return resp, repo

    resp, repo = asyncio.run(run_test())
    assert resp.status_code == 502
    assert repo.records[0].status == "failed"
    assert repo.records[0].attempted_models == [DEEPSEEK_V3_0324, GEMINI_25_PRO]
    assert repo.records[0].failover is True
    assert repo.records[0].policy_fallback is False


def test_postgres_migration_is_idempotent_and_decodes_legacy_rows():
    assert len(MIGRATION_SQL) == 4
    for stmt in MIGRATION_SQL:
        assert "ADD COLUMN IF NOT EXISTS" in stmt

    async def run():
        repo = PostgresTelemetryRepository("postgresql://user:pass@localhost:5432/testdb")
        mock_connection = AsyncMock()
        mock_begin = MagicMock()
        mock_begin.__aenter__.return_value = mock_connection
        mock_begin.__aexit__.return_value = None
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_begin

        with patch.object(repo, "_get_engine", return_value=mock_engine):
            await repo.migrate()
            assert mock_connection.execute.call_count == len(MIGRATION_SQL)

    asyncio.run(run())
    assert _decode_attempted_models(None) is None
    assert _decode_attempted_models(json.dumps(["a", "b"])) == ["a", "b"]
    legacy_row = _normalize_record_row(
        {
            "request_id": "old-1",
            "attempted_models": None,
            "failover": None,
            "policy_fallback": None,
            "retry_count": None,
        }
    )
    assert legacy_row["attempted_models"] is None
    assert legacy_row["retry_count"] is None


def test_routing_config_endpoint_exposes_real_v21_cost_multipliers(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(
        main_module,
        "router_service",
        RouterService(
            inference_service=InferenceService(ProviderFactory(mode="mock")),
            telemetry_service=TelemetryService(RecordingTelemetryRepo()),
            router_artifact="v21",
        ),
    )
    with TestClient(app) as client:
        resp = client.get("/v1/routing/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["quality_floor"] == pytest.approx(0.90)
    assert data["fallback_model"] == GEMINI_25_PRO
    assert data["telemetry_table"] == "inference_requests"
    by_id = {m["canonical_model"]: m for m in data["models"]}
    assert by_id[DEEPSEEK_V3_0324]["relative_cost_vs_cheapest"] == 1.0
    assert by_id[GEMINI_25_FLASH]["relative_cost_vs_cheapest"] == pytest.approx(6.16, abs=0.05)
    assert by_id[GEMINI_25_PRO]["relative_cost_vs_cheapest"] == pytest.approx(66.3, abs=0.2)


def test_overall_request_deadline_aborts_retries_and_failover(monkeypatch):
    import app.main as main_module
    from app.config import settings

    assert settings.llm_request_deadline_s == 120.0

    ds_calls = {"n": 0}
    pr_calls = {"n": 0}

    async def slow_ds_handler(_req):
        ds_calls["n"] += 1
        if ds_calls["n"] == 1:
            await asyncio.sleep(0.015)
            return httpx.Response(503, json={"error": {"message": "slow retryable"}})
        await asyncio.sleep(0.20)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "too late"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    async def pr_handler(_req):
        pr_calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "should not reach"}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
            },
        )

    async def slow_sleep(delay: float) -> None:
        await asyncio.sleep(delay)

    async def run_test():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(slow_ds_handler)) as ds_c,
            httpx.AsyncClient(transport=httpx.MockTransport(pr_handler)) as pr_c,
        ):
            factory = _MockHttpFactory(
                {
                    DEEPSEEK_V3_0324: DeepSeekProvider(
                        api_key="k",
                        api_model_name="deepseek-chat",
                        http_client=ds_c,
                        max_retries=3,
                        retry_base_delay_s=0.01,
                        retry_max_delay_s=0.01,
                        sleep_fn=slow_sleep,
                    ),
                    GEMINI_25_FLASH: GeminiProvider(
                        GEMINI_25_FLASH,
                        api_model_name="gemini-2.5-flash",
                        api_key="k",
                        http_client=pr_c,
                        max_retries=1,
                        retry_base_delay_s=0.0,
                    ),
                    GEMINI_25_PRO: GeminiProvider(
                        GEMINI_25_PRO,
                        api_model_name="gemini-2.5-pro",
                        api_key="k",
                        http_client=pr_c,
                        max_retries=1,
                        retry_base_delay_s=0.0,
                    ),
                }
            )
            repo = RecordingTelemetryRepo()
            service = RouterService(
                inference_service=InferenceService(factory),
                telemetry_service=TelemetryService(repo),
                request_deadline_s=0.06,
            )
            monkeypatch.setattr(
                service,
                "predict_qualities",
                lambda _prompt: {DEEPSEEK_V3_0324: 0.95, GEMINI_25_FLASH: 0.80, GEMINI_25_PRO: 0.90},
            )
            monkeypatch.setattr(main_module, "router_service", service)
            with TestClient(app) as client:
                resp = client.post("/v1/chat", json={"request_id": "deadline-1", "prompt": "deadline test"})
            return resp, repo

    resp, repo = asyncio.run(run_test())
    assert resp.status_code == 504
    assert ds_calls["n"] == 2
    assert pr_calls["n"] == 0
    assert repo.records[0].status == "failed"
    assert repo.records[0].error_type == "ProviderTimeoutError"
    assert repo.records[0].attempted_models == [DEEPSEEK_V3_0324]
    assert repo.records[0].retry_count == 1
