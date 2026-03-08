"""
Tests for Circuit Breaker.
"""

import time
from unittest.mock import patch

import pytest

from middleware.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from middleware.models import CircuitState


@pytest.fixture
def cb():
    return CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=0.30,
        window_size=10,
        window_duration_seconds=60.0,
        cooloff_seconds=1.0,  # Short for tests
        half_open_max_probes=1,
    ))


class TestCircuitBreaker:
    """Tests for per-model circuit breaker."""

    def test_starts_closed(self, cb):
        """New model starts in CLOSED state."""
        status = cb.get_status("model-a")
        assert status.state == CircuitState.CLOSED

    def test_available_when_closed(self, cb):
        """Model is available when circuit is CLOSED."""
        assert cb.is_available("model-a") is True

    def test_stays_closed_below_threshold(self, cb):
        """Circuit stays CLOSED when failure rate is below threshold."""
        # 2 failures out of 10 = 20%, below 30% threshold
        for _ in range(8):
            cb.record_success("model-a")
        for _ in range(2):
            cb.record_failure("model-a")

        assert cb.is_available("model-a") is True

    def test_opens_above_threshold(self, cb):
        """Circuit OPENS when failure rate exceeds threshold."""
        # 4 failures out of 10 = 40%, above 30% threshold
        for _ in range(6):
            cb.record_success("model-a")
        for _ in range(4):
            cb.record_failure("model-a")

        assert cb.is_available("model-a") is False
        status = cb.get_status("model-a")
        assert status.state == CircuitState.OPEN

    def test_rejects_when_open(self, cb):
        """Requests are rejected when circuit is OPEN."""
        # Trip the breaker
        for _ in range(6):
            cb.record_success("model-a")
        for _ in range(4):
            cb.record_failure("model-a")

        assert cb.is_available("model-a") is False

    def test_transitions_to_half_open_after_cooloff(self, cb):
        """Circuit transitions from OPEN to HALF_OPEN after cooloff expires."""
        # Trip the breaker
        for _ in range(6):
            cb.record_success("model-a")
        for _ in range(4):
            cb.record_failure("model-a")

        assert cb.is_available("model-a") is False

        # Wait for cooloff to expire
        time.sleep(1.1)

        # Should now be HALF_OPEN and allow a probe
        assert cb.is_available("model-a") is True

    def test_half_open_recovery_on_success(self, cb):
        """HALF_OPEN → CLOSED when probe succeeds."""
        # Trip the breaker
        for _ in range(6):
            cb.record_success("model-a")
        for _ in range(4):
            cb.record_failure("model-a")

        time.sleep(1.1)

        # Probe succeeds
        assert cb.is_available("model-a") is True
        cb.record_success("model-a")

        status = cb.get_status("model-a")
        assert status.state == CircuitState.CLOSED
        assert cb.is_available("model-a") is True

    def test_half_open_retrip_on_failure(self, cb):
        """HALF_OPEN → OPEN when probe fails."""
        # Trip the breaker
        for _ in range(6):
            cb.record_success("model-a")
        for _ in range(4):
            cb.record_failure("model-a")

        time.sleep(1.1)

        # Probe fails
        assert cb.is_available("model-a") is True
        cb.record_failure("model-a")

        status = cb.get_status("model-a")
        assert status.state == CircuitState.OPEN
        assert cb.is_available("model-a") is False

    def test_independent_models(self, cb):
        """Circuit breakers are independent per model."""
        # Trip model-a
        for _ in range(6):
            cb.record_success("model-a")
        for _ in range(4):
            cb.record_failure("model-a")

        assert cb.is_available("model-a") is False
        assert cb.is_available("model-b") is True  # Unaffected

    def test_failure_rate(self, cb):
        """Failure rate is computed correctly."""
        for _ in range(7):
            cb.record_success("model-a")
        for _ in range(3):
            cb.record_failure("model-a")

        rate = cb.get_failure_rate("model-a")
        assert abs(rate - 0.3) < 0.01

    def test_get_all_statuses(self, cb):
        """get_all_statuses returns status for all given models."""
        cb.record_success("model-a")
        cb.record_failure("model-b")
        statuses = cb.get_all_statuses(["model-a", "model-b", "model-c"])
        assert len(statuses) == 3
        assert statuses["model-a"].state == CircuitState.CLOSED

    def test_cooloff_until_set_on_open(self, cb):
        """cooloff_until is set when circuit opens."""
        for _ in range(6):
            cb.record_success("model-a")
        for _ in range(4):
            cb.record_failure("model-a")

        status = cb.get_status("model-a")
        assert status.cooloff_until is not None
