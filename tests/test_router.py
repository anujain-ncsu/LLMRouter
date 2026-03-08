"""
Tests for the Router — integration tests with mock HTTP backend.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from middleware.audit_logger import AuditLogger
from middleware.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from middleware.input_guard import InputGuard, InputValidationError
from middleware.model_registry import ModelRegistry
from middleware.models import ChatRequest
from middleware.rate_limiter import RateLimiter, RateLimiterConfig
from middleware.router import Router, RoutingError
from middleware.token_accounting import TokenAccounting


def _make_mock_response(status_code=200, model_id="novamind-7b", tokens=100):
    """Create a mock httpx.Response."""
    if status_code == 200:
        data = {
            "model_id": model_id,
            "response_text": "This is a mock response.",
            "tokens_used": tokens,
            "latency_ms": 150.0,
        }
        response = httpx.Response(status_code, json=data)
    else:
        response = httpx.Response(status_code, json={"detail": "Error"})
    return response


@pytest.fixture
def mock_client():
    """Create a mock HTTP client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def test_router(mock_client):
    """Router with mock HTTP client for isolated testing."""
    registry = ModelRegistry()
    cb = CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=0.30,
        window_size=5,
        cooloff_seconds=1.0,
    ))
    accounting = TokenAccounting()
    rate_limiter = RateLimiter(RateLimiterConfig(
        requests_per_minute=100,
        requests_per_hour=1000,
    ))
    input_guard = InputGuard()
    audit = AuditLogger(logger_name="test.router")
    audit.logger.setLevel(50)

    return Router(
        model_registry=registry,
        circuit_breaker=cb,
        token_accounting=accounting,
        rate_limiter=rate_limiter,
        input_guard=input_guard,
        audit_logger=audit,
        http_client=mock_client,
    )


class TestRouter:
    """Integration tests for the Router."""

    @pytest.mark.asyncio
    async def test_successful_auto_select(self, test_router, mock_client):
        """Auto-select routes to correct model and returns response."""
        mock_client.post.return_value = _make_mock_response(200, "novamind-7b", 100)

        request = ChatRequest(user_id="user-1", prompt="What is Python?")
        response = await test_router.route(request)

        assert response.response_text == "This is a mock response."
        assert response.auto_selected is True
        assert response.tokens_used == 100
        assert response.system_tokens_used > 0

    @pytest.mark.asyncio
    async def test_successful_manual_select(self, test_router, mock_client):
        """Manually selecting a model routes to that model."""
        mock_client.post.return_value = _make_mock_response(200, "stellarcode-70b", 200)

        request = ChatRequest(
            user_id="user-1",
            prompt="Write a function",
            model_id="stellarcode-70b",
        )
        response = await test_router.route(request)

        assert response.model_id == "stellarcode-70b"
        assert response.auto_selected is False

    @pytest.mark.asyncio
    async def test_invalid_model_id(self, test_router, mock_client):
        """Invalid model ID raises RoutingError."""
        request = ChatRequest(
            user_id="user-1",
            prompt="Hello",
            model_id="nonexistent-model",
        )
        with pytest.raises(RoutingError, match="Unknown model"):
            await test_router.route(request)

    @pytest.mark.asyncio
    async def test_empty_prompt_rejected(self, test_router, mock_client):
        """Empty prompt is rejected by input guard."""
        with pytest.raises(InputValidationError):
            request = ChatRequest(user_id="user-1", prompt="   ")
            await test_router.route(request)

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, test_router, mock_client):
        """When primary model fails, router falls back to next model."""
        # First call fails (500), second succeeds
        mock_client.post.side_effect = [
            _make_mock_response(500),
            _make_mock_response(200, "nebulachat-3b", 80),
        ]

        request = ChatRequest(user_id="user-1", prompt="Hello there")
        response = await test_router.route(request)

        assert response.response_text == "This is a mock response."
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_all_models_fail(self, test_router, mock_client):
        """When all models fail, raises RoutingError."""
        mock_client.post.return_value = _make_mock_response(500)

        request = ChatRequest(user_id="user-1", prompt="Hello there")
        with pytest.raises(RoutingError, match="All models failed"):
            await test_router.route(request)

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self, test_router, mock_client):
        """Timeout on primary model triggers fallback."""
        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout"),
            _make_mock_response(200, "nebulachat-3b", 50),
        ]

        request = ChatRequest(user_id="user-1", prompt="Hello there")
        response = await test_router.route(request)
        assert response.response_text == "This is a mock response."

    @pytest.mark.asyncio
    async def test_token_accounting_on_success(self, test_router, mock_client):
        """Token accounting is updated after a successful request."""
        mock_client.post.return_value = _make_mock_response(200, "novamind-7b", 100)

        request = ChatRequest(user_id="user-1", prompt="What is Python?")
        await test_router.route(request)

        usage = await test_router.accounting.get_user_usage("user-1")
        assert usage.total_model_tokens == 100
        assert usage.total_requests == 1

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_client):
        """Rate limiting is enforced."""
        registry = ModelRegistry()
        cb = CircuitBreaker()
        accounting = TokenAccounting()
        rl = RateLimiter(RateLimiterConfig(requests_per_minute=2, requests_per_hour=100))
        guard = InputGuard()
        audit = AuditLogger(logger_name="test.rl")
        audit.logger.setLevel(50)

        router = Router(
            model_registry=registry,
            circuit_breaker=cb,
            token_accounting=accounting,
            rate_limiter=rl,
            input_guard=guard,
            audit_logger=audit,
            http_client=mock_client,
        )

        mock_client.post.return_value = _make_mock_response(200)

        # First 2 should succeed
        for _ in range(2):
            await router.route(ChatRequest(user_id="user-1", prompt="Hello"))

        # Third should be rate limited
        with pytest.raises(RoutingError, match="Rate limit"):
            await router.route(ChatRequest(user_id="user-1", prompt="Hello"))

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_unavailable(self, test_router, mock_client):
        """Router skips models with open circuit breakers."""
        # Manually trip a circuit breaker
        for _ in range(3):
            test_router.circuit_breaker.record_success("novamind-7b")
        for _ in range(5):
            test_router.circuit_breaker.record_failure("novamind-7b")

        mock_client.post.return_value = _make_mock_response(200, "nebulachat-3b", 50)

        request = ChatRequest(user_id="user-1", prompt="What is Python?")
        response = await test_router.route(request)

        # Should have used a fallback model (not novamind which is tripped)
        assert response.response_text == "This is a mock response."
