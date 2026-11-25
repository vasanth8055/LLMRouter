from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.providers.base import LLMResponse
from app.providers.errors import ProviderTimeoutError
from app.providers.model_registry import GEMINI_38_FLASH_API
from app.schemas import RouteRequest
from app.services.router_service import (
    MODEL_IDS,
    ROOT,
    RouterService,
    RoutingConstraintError,
)
from app.services.inference_service import InferenceService
from app.providers.factory import ProviderFactory
from app.services.routing_policy import route_by_quality_and_cost


class RecordingInference:
    def __init__(self, result: LLMResponse | None = None, error: Exception | None = None):
        self.result = result or LLMResponse("mock-provider-model", "mock answer", 4, 6, 7.5, 0.0002)
        self.error = error
        self.calls = []

    async def generate(self, model_name, prompt, max_tokens=None, temperature=0.0):
        self.calls.append((model_name, prompt, max_tokens, temperature))
        if self.error:
            raise self.error
        return self.result


def test_frozen_v21_artifact_loads_unchanged_and_exposes_expected_metadata():
    service = RouterService(router_artifact="v21")
    integrity = json.loads((ROOT / "data/router_v22_frontier_integrity.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / "models/router_v21_tfidf.joblib").read_bytes()).hexdigest()
    assert digest == integrity["loaded_v21_model_sha256_before"]
    assert service.threshold == 0.90
    assert service.fallback_model == "gemini-2.5-pro"
    assert service.features == ("prompt", "model")
    assert set(service.training_average_costs) == set(MODEL_IDS)


def test_runtime_prediction_uses_only_prompt_and_canonical_model_features(monkeypatch):
    service = RouterService(router_artifact="v21")
    captured = {}

    def fake_predict_proba(frame):
        captured["columns"] = list(frame.columns)
        captured["records"] = frame.to_dict("records")
        return np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])

    monkeypatch.setattr(service.predictor, "predict_proba", fake_predict_proba)
    values = service.predict_qualities("runtime prompt")
    assert captured["columns"] == ["prompt", "model"]
    assert captured["records"] == [
        {"prompt": "runtime prompt", "model": model} for model in MODEL_IDS
    ]
    assert values == dict(zip(MODEL_IDS, [0.9, 0.8, 0.7]))


def test_v21_policy_selects_using_frozen_threshold_costs_and_fallback():
    service = RouterService(router_artifact="v21")
    prompt = "Explain binary search in simple terms."
    probabilities = service.predict_qualities(prompt)
    selected, _reason, selected_quality = service._choose_model(probabilities, 0.90, None)
    expected = route_by_quality_and_cost(
        probabilities, service.training_average_costs, 0.90, service.fallback_model
    )
    assert selected == expected["selected_model"]
    assert selected_quality == probabilities[selected]
    assert selected in MODEL_IDS


def test_runtime_min_quality_cannot_lower_frozen_threshold_but_can_raise_it():
    service = RouterService(router_artifact="v21")
    probabilities = {MODEL_IDS[0]: 0.94, MODEL_IDS[1]: 0.96, MODEL_IDS[2]: 0.99}
    # Values below 0.90 cannot lower the floor and are rejected with RoutingConstraintError.
    with pytest.raises(RoutingConstraintError, match="cannot lower the frozen V2.1 policy floor"):
        service._choose_model(probabilities, 0.80, None)
    with pytest.raises(RoutingConstraintError, match="cannot lower the frozen V2.1 policy floor"):
        service._choose_model(probabilities, 0.89, None)
    # Stricter values raise the floor above 0.90 and affect model selection.
    floor_choice, _, _ = service._choose_model(probabilities, 0.90, None)
    assert floor_choice == MODEL_IDS[0]
    stricter_choice, _, _ = service._choose_model(probabilities, 0.95, None)
    assert stricter_choice == MODEL_IDS[1]


def test_inference_service_receives_exact_selected_model_and_request_parameters():
    inference = RecordingInference()
    service = RouterService(inference_service=inference)
    response = asyncio.run(service.handle_async(RouteRequest(
        request_id="req-123", prompt="Explain binary search simply.",
        min_quality=0.90, max_tokens=19, temperature=0.25,
    )))
    assert len(inference.calls) == 1
    selected, prompt, max_tokens, temperature = inference.calls[0]
    assert selected == response.selected_model
    assert prompt == "Explain binary search simply."
    assert max_tokens == 19 and temperature == 0.25
    assert response.provider_model == "mock-provider-model"


def test_response_distinguishes_predicted_quality_from_actual_quality():
    response = asyncio.run(RouterService(inference_service=RecordingInference()).handle_async(
        RouteRequest(prompt="A sample prompt")
    ))
    data = response.model_dump()
    assert 0 <= data["predicted_quality"] <= 1
    assert "actual_quality" not in data
    assert data["input_tokens"] == 4 and data["output_tokens"] == 6
    assert data["latency_ms"] == 7.5 and data["estimated_cost_usd"] == 0.0002


def test_mock_routing_respects_latency_constraint_or_reports_no_valid_fallback():
    service = RouterService(router_artifact="v21")
    probabilities = {MODEL_IDS[0]: 0.99, MODEL_IDS[1]: 0.1, MODEL_IDS[2]: 0.1}
    selected, _, _ = service._choose_model(probabilities, 0.90, 500)
    assert selected == MODEL_IDS[0]
    with pytest.raises(RoutingConstraintError, match="frozen V2.1 fallback"):
        service._choose_model({model: 0.1 for model in MODEL_IDS}, 0.90, 500)


def test_real_routing_requires_and_uses_expected_latency_configuration(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_mode", "real")
    service = RouterService()
    probabilities = {MODEL_IDS[0]: 0.99, MODEL_IDS[1]: 0.95, MODEL_IDS[2]: 0.1}
    with pytest.raises(RoutingConstraintError, match="expected latency values"):
        service._choose_model(probabilities, 0.90, 250)
    monkeypatch.setattr(settings, "deepseek_expected_latency_ms", 100)
    monkeypatch.setattr(settings, "gemini_25_flash_expected_latency_ms", 200)
    monkeypatch.setattr(settings, "gemini_25_pro_expected_latency_ms", 300)
    selected, _, _ = service._choose_model(probabilities, 0.90, 250)
    assert selected == MODEL_IDS[0]


def test_chat_endpoint_runs_frozen_router_through_mock_provider():
    client = TestClient(app)
    response = client.post("/v1/chat", json={
        "prompt": "Explain binary search in simple terms.",
        "request_id": "e2e-123", "min_quality": 0.90,
        "max_tokens": 20, "temperature": 0,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "e2e-123"
    assert body["selected_model"] in MODEL_IDS
    assert body["canonical_model"] == body["selected_model"]
    assert 0 <= body["predicted_quality"] <= 1
    assert body["response"].startswith(f"[{body['selected_model']}]")
    assert body["input_tokens"] is not None and body["output_tokens"] == 20
    assert body["latency_ms"] > 0 and body["estimated_cost_usd"] > 0
    assert body["provider_model"] == body["selected_model"]
    assert body["model_mapping_warning"] is None


def test_route_explain_returns_real_frozen_candidate_predictions_without_prompt(monkeypatch):
    import app.main as main_module

    service = RouterService(router_artifact="v21")
    prompt = "private OOD prompt text should not be returned"
    expected = service.predict_qualities(prompt)
    monkeypatch.setattr(main_module, "router_service", service)
    response = TestClient(app).post("/v1/route/explain", json={"prompt": prompt, "min_quality": 0.90})
    assert response.status_code == 200
    body = response.json()
    assert body["prompt_length"] == len(prompt)
    assert body["quality_floor"] == 0.90
    assert {row["canonical_model"]: row["predicted_quality"] for row in body["candidates"]} == expected
    assert body["selected_model"] in MODEL_IDS
    assert body["fallback"] == body["routing_reason"].startswith("No eligible model met predicted quality")
    assert prompt not in response.text
    assert "response" not in body


def test_runtime_metadata_is_safe_and_explain_enforces_frozen_floor(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "router_service", RouterService(router_artifact="v21"))
    client = TestClient(app)
    runtime = client.get("/v1/runtime")
    assert runtime.status_code == 200
    assert runtime.json() == {"provider_mode": "mock"}
    assert client.post("/v1/route/explain", json={"prompt": "valid", "min_quality": 0.89}).status_code == 422


@pytest.mark.parametrize("payload", [
    {"prompt": ""},
    {"prompt": "valid", "min_quality": 0.89},
    {"prompt": "valid", "min_quality": 0.80},
    {"prompt": "valid", "min_quality": 0.0},
    {"prompt": "valid", "min_quality": -0.1},
    {"prompt": "valid", "min_quality": 1.1},
    {"prompt": "valid", "max_latency_ms": 0},
    {"prompt": "valid", "max_tokens": 0},
    {"prompt": "valid", "temperature": 2.1},
])
def test_chat_request_validation(payload, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "router_service", RouterService(router_artifact="v21"))
    assert TestClient(app).post("/v1/chat", json=payload).status_code == 422


def test_chat_min_quality_contract_explicit_cases(monkeypatch):
    import app.main as main_module

    service = RouterService(router_artifact="v21")
    monkeypatch.setattr(main_module, "router_service", service)
    client = TestClient(app)
    # 1. Omitted min_quality defaults to 0.90 and succeeds
    res_omitted = client.post("/v1/chat", json={"prompt": "Explain binary search in simple terms."})
    assert res_omitted.status_code == 200
    assert 0.0 <= res_omitted.json()["predicted_quality"] <= 1.0

    # 2. Exact quality floor 0.90 succeeds
    res_floor = client.post("/v1/chat", json={"prompt": "Explain binary search in simple terms.", "min_quality": 0.90})
    assert res_floor.status_code == 200

    # 3. Maximum valid quality floor 1.0 succeeds
    res_max = client.post("/v1/chat", json={"prompt": "Explain binary search in simple terms.", "min_quality": 1.0})
    assert res_max.status_code == 200

    # 4. Values below 0.90 are rejected by API validation with HTTP 422
    for low in [0.89, 0.80, 0.50, 0.0]:
        res_low = client.post("/v1/chat", json={"prompt": "Explain binary search in simple terms.", "min_quality": low})
        assert res_low.status_code == 422

    # 5. Out-of-range values are rejected by API validation with HTTP 422
    for ood in [-0.5, -0.01, 1.01, 2.0]:
        res_ood = client.post("/v1/chat", json={"prompt": "Explain binary search in simple terms.", "min_quality": ood})
        assert res_ood.status_code == 422

    # 6. Service-level checks for programmatic calls
    probs = service.predict_qualities("Explain binary search in simple terms.")
    _, _, q90 = service._choose_model(probs, 0.90, None)
    assert 0.0 <= q90 <= 1.0
    _, _, q100 = service._choose_model(probs, 1.0, None)
    assert 0.0 <= q100 <= 1.0
    with pytest.raises(RoutingConstraintError, match="cannot lower the frozen V2.1 policy floor"):
        service._choose_model(probs, 0.89, None)
    with pytest.raises(RoutingConstraintError, match="must be between 0.90 and 1.00"):
        service._choose_model(probs, 1.01, None)

    # 7. RouteRequest Pydantic + artifact floor contract
    assert RouteRequest(prompt="valid").min_quality is None
    assert RouteRequest(prompt="valid", min_quality=0.90).min_quality == 0.90
    assert RouteRequest(prompt="valid", min_quality=1.0).min_quality == 1.0
    assert RouteRequest(prompt="valid", min_quality=0.89).min_quality == 0.89
    with pytest.raises(RoutingConstraintError, match="cannot lower the frozen V2.1 policy floor"):
        service._choose_model(probs, 0.89, None)
    with pytest.raises(ValueError):
        RouteRequest(prompt="valid", min_quality=1.01)


def test_provider_failure_maps_to_http_error_without_fallback(monkeypatch):
    import app.main as main_module

    inference = RecordingInference(error=ProviderTimeoutError("provider timed out"))
    monkeypatch.setattr(main_module, "router_service", RouterService(inference_service=inference))
    response = TestClient(app).post("/v1/chat", json={"prompt": "a valid prompt"})
    assert response.status_code == 504
    assert response.json()["detail"] == "provider timed out"
    assert len(inference.calls) == 1


def test_mock_mode_does_not_construct_real_providers(monkeypatch):
    assert settings.llm_provider_mode == "mock"
    factory_module = importlib.import_module("app.providers.factory")
    monkeypatch.setattr(factory_module, "DeepSeekProvider", lambda **_kwargs: pytest.fail("real provider instantiated"))
    monkeypatch.setattr(factory_module, "GeminiProvider", lambda **_kwargs: pytest.fail("real provider instantiated"))
    result = TestClient(app).post("/v1/chat", json={"prompt": "mock-only request"})
    assert result.status_code == 200


@pytest.mark.parametrize(("model_name", "probabilities", "provider_class_name"), [
    ("deepseek-v3-0324", [0.99, 0.50, 0.40], "DeepSeekProvider"),
    ("gemini-2.5-flash", [0.50, 0.99, 0.40], "GeminiProvider"),
])
def test_real_mode_chat_uses_factory_with_mocked_provider_calls(
    monkeypatch, model_name, probabilities, provider_class_name,
):
    import app.main as main_module

    class FixedPredictor:
        named_steps = {"classifier": SimpleNamespace(classes_=np.array([0, 1]))}

        def predict_proba(self, frame):
            positive = dict(zip(MODEL_IDS, probabilities))
            return np.array([[1 - positive[model], positive[model]] for model in frame["model"]])

    package = {
        "predictor": FixedPredictor(),
        "metadata": {"features": ["prompt", "model"]},
        "selected_threshold": 0.90,
        "training_average_costs": {
            "deepseek-v3-0324": 0.001,
            "gemini-2.5-flash": 0.004,
            "gemini-2.5-pro": 0.04,
        },
        "fallback_model": "gemini-2.5-pro",
    }
    calls = []

    class FakeProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def generate(self, prompt, max_tokens=None, temperature=0.0):
            calls.append((self.kwargs, prompt, max_tokens, temperature))
            actual_model = self.kwargs.get("api_model_name", self.kwargs.get("model_name"))
            return LLMResponse(actual_model, "real-provider-mocked", 2, 3, 12.0, None)

    provider_factory_module = importlib.import_module("app.providers.factory")
    monkeypatch.setattr(provider_factory_module.settings, "deepseek_api_key", "fake-ds-key")
    monkeypatch.setattr(provider_factory_module.settings, "deepseek_api_model", "deepseek-flash")
    monkeypatch.setattr(provider_factory_module.settings, "gemini_api_key", "fake-gemini-key")
    monkeypatch.setattr(provider_factory_module.settings, "gemini_flash_api_model", GEMINI_38_FLASH_API)
    monkeypatch.setattr(provider_factory_module, provider_class_name, FakeProvider)
    service = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="real")),
        model_package=package,
    )
    monkeypatch.setattr(main_module, "router_service", service)
    response = TestClient(app).post("/v1/chat", json={
        "prompt": "real path fixture", "max_tokens": 13, "temperature": 0.2,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["selected_model"] == model_name
    assert body["canonical_model"] == model_name
    assert body["provider_model"] == ("deepseek-flash" if model_name.startswith("deepseek") else GEMINI_38_FLASH_API)
    assert model_name in body["model_mapping_warning"]
    assert body["provider_model"] in body["model_mapping_warning"]
    assert body["response"] == "real-provider-mocked"
    expected_kwargs = {
        "api_key": "fake-ds-key",
        "api_model_name": "deepseek-flash",
        "input_price_per_million": None,
        "output_price_per_million": None,
        "http_client": service.inference_service.provider_factory.http_client,
    } if model_name.startswith("deepseek") else {
        "model_name": model_name,
        "api_model_name": GEMINI_38_FLASH_API,
        "api_key": "fake-gemini-key",
        "input_price_per_million": None,
        "output_price_per_million": None,
        "http_client": service.inference_service.provider_factory.http_client,
    }
    assert isinstance(service.inference_service.provider_factory.http_client, httpx.AsyncClient)
    assert calls == [(expected_kwargs, "real path fixture", 13, 0.2)]


def test_chat_logging_has_routing_metadata_without_prompt(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.router_service")
    prompt = "private prompt that must not be logged"
    asyncio.run(RouterService().handle_async(RouteRequest(prompt=prompt, request_id="logging-test")))
    record = next(record for record in caplog.records if record.message == "inference_completed")
    assert record.request_id == "logging-test"
    assert record.selected_model in MODEL_IDS
    assert record.provider
    assert record.latency_ms > 0
    assert record.input_tokens is not None and record.output_tokens is not None
    assert record.estimated_cost_usd is not None
    assert prompt not in caplog.text


def test_research_artifacts_and_integrity_records_are_unchanged():
    expected = {
        "models/router_v21_tfidf.joblib": "f2041eddfff07cebb0ab62cb2c6496e6558adfc7c97a8bf2670880cd145ab4bc",
        "data/router_v21_split_report.json": "cc526bfddc2c5ed67a51e97697edc478eaafc983948db2dd800913c8d83cf88f",
        "data/router_v21_validation_results.csv": "711c8485f232db43c750ad9434a8de13e9043af76dd1e0ecb386127672844512",
        "data/router_v21_final_test_results.csv": "694a29ba27bd9ea5d9a7e8d9e236c8ff0c9eb3964645ae754121d9f671eb643d",
        "data/router_v21_final_test_integrity.json": "32cc0b050c6b556558dc90beba1b04ffd102707d561c74e7a23d322e642a0672",
        "data/router_v22_frontier_results.csv": "756a7518853e7df702f853cde4a40ffbb187d80d00c52339d469b139fc9ab09f",
        "data/router_v22_frontier_integrity.json": "2ffe8518ad0b480ddc7e03812debe82355a195413adf79873bb96c6069803167",
    }
    actual = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in expected
    }
    assert actual == expected
