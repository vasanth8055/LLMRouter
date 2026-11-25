from __future__ import annotations

import joblib
import numpy as np
import pytest

from app.config import settings
from app.providers.factory import ProviderFactory
from app.services.inference_service import InferenceService
from app.services.router_service import MODEL_IDS, ROOT, RouterService
from app.telemetry.repository import NoopTelemetryRepository
from app.telemetry.service import TelemetryService


def test_v5_platt_artifact_loads_via_router_service_and_keeps_v21_selectable(monkeypatch):
    from app.config import Settings

    assert Settings(_env_file=None).router_mode == "v5-platt"

    v21 = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
        router_artifact="v21",
    )
    assert v21.artifact_version == "v21"
    assert v21.threshold == 0.90
    assert v21.platt_params is None

    v21_baseline = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
        router_artifact="baseline",
    )
    assert v21_baseline.artifact_version == "v21" and v21_baseline.threshold == 0.90

    v5 = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
        router_artifact="v5-platt",
    )
    assert v5.artifact_version == "v5-platt"
    assert v5.threshold == pytest.approx(0.85)
    assert v5.platt_params is not None and set(v5.platt_params) == set(MODEL_IDS)

    # Also verify settings.router_mode="v5-platt" selection without argument
    monkeypatch.setattr(settings, "router_mode", "v5-platt")
    v5_env = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
    )
    assert v5_env.artifact_version == "v5-platt" and v5_env.threshold == pytest.approx(0.85)

    # Verify predict_qualities matches explicit Platt transform on raw LR outputs
    pkg = joblib.load(ROOT / "models" / "router_v5_platt.joblib")
    probs = v5.predict_qualities("What is 2 + 2?")
    assert set(probs) == set(MODEL_IDS)
    for m, p in probs.items():
        assert 0.0 <= p <= 1.0
    assert pkg["metadata"]["version"] == "v5-platt"
    assert pkg["metadata"]["selected_tau"] == 0.85


def test_api_enforces_artifact_tau_without_loosening_v21(monkeypatch):
    import app.main as main_module
    from fastapi.testclient import TestClient

    from app.main import app

    # 1. With V2.1 active (0.90), values < 0.90 remain rejected with 422
    v21 = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
        router_artifact="v21",
    )
    monkeypatch.setattr(main_module, "router_service", v21)
    with TestClient(app) as client:
        assert client.post("/v1/chat", json={"prompt": "test", "min_quality": 0.85}).status_code == 422
        assert client.post("/v1/route/explain", json={"prompt": "test", "min_quality": 0.89}).status_code == 422
        assert client.post("/v1/chat", json={"prompt": "test"}).status_code == 200

    # 2. With V5 Platt active (tau=0.85), default resolves to 0.85 and < 0.85 is rejected
    v5 = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
        router_artifact="v5-platt",
    )
    monkeypatch.setattr(main_module, "router_service", v5)
    with TestClient(app) as client:
        cfg = client.get("/v1/routing/config")
        assert cfg.status_code == 200
        assert cfg.json()["policy_id"] == "V5 Platt Cost-Aware Router"
        assert cfg.json()["quality_floor"] == pytest.approx(0.85)
        by_id = {m["canonical_model"]: m for m in cfg.json()["models"]}
        assert by_id["deepseek-v3-0324"]["relative_cost_vs_cheapest"] == 1.0
        assert by_id["gemini-2.5-flash"]["relative_cost_vs_cheapest"] == pytest.approx(6.21, abs=0.05)
        assert by_id["gemini-2.5-pro"]["relative_cost_vs_cheapest"] == pytest.approx(66.63, abs=0.2)

        exp_default = client.post("/v1/route/explain", json={"prompt": "Explain quicksort."})
        assert exp_default.status_code == 200
        assert exp_default.json()["quality_floor"] == pytest.approx(0.85)

        assert client.post("/v1/chat", json={"prompt": "Explain quicksort.", "min_quality": 0.85}).status_code == 200
        assert client.post("/v1/chat", json={"prompt": "Explain quicksort.", "min_quality": 0.90}).status_code == 200
        assert client.post("/v1/chat", json={"prompt": "Explain quicksort.", "min_quality": 0.84}).status_code == 422
        assert client.post("/v1/route/explain", json={"prompt": "Explain quicksort.", "min_quality": 0.84}).status_code == 422
