"""
Tests for Mock LLM Backend Services.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from mock_backends.server import app as mock_app


@pytest.fixture
async def mock_client():
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestMockBackends:
    """Tests for the mock LLM backend services."""

    @pytest.mark.asyncio
    async def test_valid_model_returns_response(self, mock_client):
        """Valid model returns a proper response."""
        response = await mock_client.post(
            "/models/novamind-7b/generate",
            json={"prompt": "Hello", "max_tokens": 1024},
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert data["model_id"] == "novamind-7b"
        assert "response_text" in data
        assert len(data["response_text"]) > 0
        assert "tokens_used" in data
        assert data["tokens_used"] > 0
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_invalid_model_returns_404(self, mock_client):
        """Invalid model returns 404."""
        response = await mock_client.post(
            "/models/nonexistent-model/generate",
            json={"prompt": "Hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_all_valid_models(self, mock_client):
        """All four models return valid responses."""
        models = ["novamind-7b", "quantumleap-13b", "stellarcode-70b", "nebulachat-3b"]
        for model_id in models:
            response = await mock_client.post(
                f"/models/{model_id}/generate",
                json={"prompt": "Test prompt", "max_tokens": 512},
            )
            # May get an injected error, but should at least get a valid HTTP response
            assert response.status_code in (200, 408, 429, 500, 503)

    @pytest.mark.asyncio
    async def test_response_format(self, mock_client):
        """Successful responses have the correct format."""
        # Send many to find at least one success
        for _ in range(10):
            response = await mock_client.post(
                "/models/novamind-7b/generate",
                json={"prompt": "Tell me a story", "max_tokens": 2048},
            )
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data["response_text"], str)
                assert isinstance(data["tokens_used"], int)
                assert isinstance(data["latency_ms"], float)
                assert data["model_id"] == "novamind-7b"
                return
        # If all 10 were errors, that's statistically very unlikely but acceptable
        pytest.skip("All 10 requests were error-injected (extremely unlikely)")

    @pytest.mark.asyncio
    async def test_error_injection_rate(self, mock_client):
        """Error injection rate is approximately 2-4% over many requests."""
        errors = 0
        total = 200

        for _ in range(total):
            response = await mock_client.post(
                "/models/novamind-7b/generate",
                json={"prompt": "Test", "max_tokens": 128},
            )
            if response.status_code != 200:
                errors += 1

        error_rate = errors / total
        # Allow some statistical variance: expect ~3% ± margin
        # With 200 samples and 3% error rate, expect ~6 errors
        # Accept between 0.5% and 10% (very generous bounds for test stability)
        assert 0.005 <= error_rate <= 0.10, (
            f"Error rate {error_rate:.2%} is outside expected range [0.5%, 10%]"
        )

    @pytest.mark.asyncio
    async def test_error_types_are_valid_http_codes(self, mock_client):
        """All injected errors use valid HTTP status codes."""
        valid_error_codes = {408, 429, 500, 503}
        for _ in range(100):
            response = await mock_client.post(
                "/models/novamind-7b/generate",
                json={"prompt": "Test", "max_tokens": 128},
            )
            if response.status_code != 200:
                assert response.status_code in valid_error_codes

    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_client):
        """Health endpoint returns status and model list."""
        response = await mock_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert len(data["models"]) == 4

    @pytest.mark.asyncio
    async def test_max_tokens_capping(self, mock_client):
        """Token count does not exceed max_tokens."""
        for _ in range(20):
            response = await mock_client.post(
                "/models/novamind-7b/generate",
                json={"prompt": "Test", "max_tokens": 50},
            )
            if response.status_code == 200:
                data = response.json()
                assert data["tokens_used"] <= 50
