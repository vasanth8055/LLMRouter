from fastapi.testclient import TestClient
from app.main import app
from app.schemas import RouteRequest
from app.services.router_service import RouterService

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_easy_request():
    r = RouterService().handle(RouteRequest(prompt="Rewrite this sentence politely: Please send the report."))
    assert r.selected_model == "gemini-2.5-pro"
    assert 0 <= r.predicted_quality <= 1

def test_hard_request():
    r = RouterService().handle(RouteRequest(prompt="Design and analyze a distributed system architecture for multi-region failover, explain the trade-offs, and optimize consistency and latency."))
    assert r.selected_model == "gemini-2.5-pro"
    assert r.provider_model == r.selected_model
