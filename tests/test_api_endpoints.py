"""
End-to-end tests for API endpoints.

Tests the full FastAPI app using httpx.AsyncClient with ASGITransport.
"""

import pytest
from httpx import ASGITransport, AsyncClient

import server.app as app_module
from server.app import (
    app, model_registry, circuit_breaker, token_accounting,
    rate_limiter, input_guard, audit_logger,
)
from middleware.router import Router


@pytest.fixture
async def client():
    # Manually initialize the router_engine since ASGI test transport
    # doesn't trigger FastAPI lifespan events
    router = Router(
        model_registry=model_registry,
        circuit_breaker=circuit_breaker,
        token_accounting=token_accounting,
        rate_limiter=rate_limiter,
        input_guard=input_guard,
        audit_logger=audit_logger,
    )
    app_module.router_engine = router
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await router.close()
    app_module.router_engine = None


class TestAPIEndpoints:
    """End-to-end tests for all API endpoints."""

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """GET /api/models returns all models."""
        response = await client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 4
        model = data["models"][0]
        assert "id" in model
        assert "name" in model
        assert "description" in model
        assert "cost_ratio" in model
        assert "capabilities" in model
        assert "available" in model

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """GET /api/health returns system health."""
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models" in data
        assert "total_requests_served" in data
        assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_usage_empty_user(self, client):
        """GET /api/usage/{user_id} returns empty usage for new user."""
        response = await client.get("/api/usage/new-user")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "new-user"
        assert data["total_requests"] == 0
        assert data["total_system_tokens"] == 0

    @pytest.mark.asyncio
    async def test_chat_missing_fields(self, client):
        """POST /api/chat with missing fields returns 422."""
        response = await client.post("/api/chat", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_empty_prompt(self, client):
        """POST /api/chat with empty prompt returns 400."""
        response = await client.post("/api/chat", json={
            "user_id": "user-1",
            "prompt": "   ",
        })
        # Pydantic might catch it (422) or input guard (400)
        assert response.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_chat_invalid_model(self, client):
        """POST /api/chat with invalid model_id returns 400."""
        response = await client.post("/api/chat", json={
            "user_id": "user-1",
            "prompt": "Hello",
            "model_id": "fake-model",
        })
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_admin_metrics(self, client):
        """GET /api/admin/metrics returns aggregate metrics."""
        response = await client.get("/api/admin/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "total_errors" in data
        assert "error_rate" in data
        assert "per_model_metrics" in data

    @pytest.mark.asyncio
    async def test_chat_with_model_backend_unavailable(self, client):
        """POST /api/chat when backends are unavailable returns 503."""
        # Since no actual mock backend is running on port 8100,
        # the router should fail to reach any backend and return 503
        response = await client.post("/api/chat", json={
            "user_id": "user-1",
            "prompt": "Hello, how are you?",
        })
        # Should get 503 because mock backend is not available
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_models_have_circuit_state(self, client):
        """Model listing includes circuit breaker state."""
        response = await client.get("/api/models")
        data = response.json()
        for model in data["models"]:
            assert "circuit_state" in model
            assert model["circuit_state"] in ("closed", "open", "half_open")

    @pytest.mark.asyncio
    async def test_health_models_have_failure_rate(self, client):
        """Health endpoint includes failure rate per model."""
        response = await client.get("/api/health")
        data = response.json()
        for model in data["models"]:
            assert "failure_rate" in model
            assert isinstance(model["failure_rate"], (int, float))
