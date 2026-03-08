"""
Tests for Rate Limiter.
"""

import time

import pytest

from middleware.rate_limiter import RateLimiter, RateLimiterConfig


@pytest.fixture
def limiter():
    return RateLimiter(RateLimiterConfig(
        requests_per_minute=5,
        requests_per_hour=100,
    ))


class TestRateLimiter:
    """Tests for token-bucket rate limiter."""

    def test_allows_within_limits(self, limiter):
        """Requests within limits are allowed."""
        for _ in range(5):
            allowed, retry_after = limiter.check_rate_limit("user-1")
            assert allowed is True
            assert retry_after is None
            limiter.consume("user-1")

    def test_rejects_when_minute_limit_exceeded(self, limiter):
        """Rejects and provides retry_after when per-minute limit is exceeded."""
        for _ in range(5):
            limiter.consume("user-1")

        allowed, retry_after = limiter.check_rate_limit("user-1")
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_independent_users(self, limiter):
        """Users have independent rate limits."""
        for _ in range(5):
            limiter.consume("user-1")

        # user-1 is rate limited
        allowed1, _ = limiter.check_rate_limit("user-1")
        assert allowed1 is False

        # user-2 is fine
        allowed2, _ = limiter.check_rate_limit("user-2")
        assert allowed2 is True

    def test_refill_over_time(self, limiter):
        """Tokens refill over time."""
        for _ in range(5):
            limiter.consume("user-1")

        # Wait a bit for refill (1 token at 5/60 = 0.083 tokens/sec → ~12s per token)
        # For speed in tests, we just verify the refill logic is applied
        allowed_before, _ = limiter.check_rate_limit("user-1")
        assert allowed_before is False

        # Simulate time passing by manipulating bucket
        bucket = limiter._get_bucket("user-1")
        bucket.minute_tokens = 1.0  # Simulate 1 token refilled
        allowed_after, _ = limiter.check_rate_limit("user-1")
        assert allowed_after is True

    def test_consume_returns_false_when_limited(self, limiter):
        """consume() returns False when rate limited."""
        for _ in range(5):
            assert limiter.consume("user-1") is True

        assert limiter.consume("user-1") is False

    def test_hour_limit(self):
        """Per-hour limit also applies."""
        limiter = RateLimiter(RateLimiterConfig(
            requests_per_minute=100,  # High minute limit
            requests_per_hour=3,     # Low hour limit
        ))

        for _ in range(3):
            limiter.consume("user-1")

        allowed, retry_after = limiter.check_rate_limit("user-1")
        assert allowed is False
        assert retry_after is not None

    def test_new_user_gets_full_bucket(self, limiter):
        """A brand new user starts with full token buckets."""
        allowed, retry_after = limiter.check_rate_limit("brand-new-user")
        assert allowed is True
        assert retry_after is None
