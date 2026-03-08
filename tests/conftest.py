"""
Shared fixtures for the LLM Router test suite.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from middleware.audit_logger import AuditLogger
from middleware.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from middleware.input_guard import InputGuard
from middleware.model_registry import ModelRegistry
from middleware.rate_limiter import RateLimiter, RateLimiterConfig
from middleware.router import Router
from middleware.token_accounting import TokenAccounting


@pytest.fixture
def model_registry():
    return ModelRegistry()


@pytest.fixture
def circuit_breaker():
    return CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=0.30,
        window_size=10,
        window_duration_seconds=60.0,
        cooloff_seconds=2.0,  # Short for tests
    ))


@pytest.fixture
def token_accounting():
    return TokenAccounting()


@pytest.fixture
def rate_limiter():
    return RateLimiter(RateLimiterConfig(
        requests_per_minute=60,  # Generous for tests
        requests_per_hour=1000,
    ))


@pytest.fixture
def input_guard():
    return InputGuard()


@pytest.fixture
def audit_logger():
    logger = AuditLogger(logger_name="test.audit")
    logger.logger.setLevel(50)  # Suppress output during tests
    return logger


@pytest_asyncio.fixture
async def router(model_registry, circuit_breaker, token_accounting, rate_limiter, input_guard, audit_logger):
    r = Router(
        model_registry=model_registry,
        circuit_breaker=circuit_breaker,
        token_accounting=token_accounting,
        rate_limiter=rate_limiter,
        input_guard=input_guard,
        audit_logger=audit_logger,
    )
    yield r
    await r.close()


@pytest_asyncio.fixture
async def app_client():
    """Create an async test client for the FastAPI app."""
    from server.app import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
