"""
Rate Limiter — token-bucket per-user rate limiting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 20
    requests_per_hour: int = 100


@dataclass
class _UserBucket:
    """Internal token bucket state for a single user."""
    # Minute bucket
    minute_tokens: float = 0.0
    minute_last_refill: float = 0.0
    # Hour bucket
    hour_tokens: float = 0.0
    hour_last_refill: float = 0.0


class RateLimiter:
    """
    Token-bucket rate limiter with per-user limits.

    Two buckets per user: per-minute and per-hour.
    Both must have available tokens for a request to be allowed.
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        self.config = config or RateLimiterConfig()
        self._buckets: dict[str, _UserBucket] = {}

    def _get_bucket(self, user_id: str) -> _UserBucket:
        if user_id not in self._buckets:
            now = time.monotonic()
            self._buckets[user_id] = _UserBucket(
                minute_tokens=float(self.config.requests_per_minute),
                minute_last_refill=now,
                hour_tokens=float(self.config.requests_per_hour),
                hour_last_refill=now,
            )
        return self._buckets[user_id]

    def _refill(self, bucket: _UserBucket) -> None:
        now = time.monotonic()

        # Refill minute bucket
        elapsed_min = now - bucket.minute_last_refill
        refill_min = elapsed_min * (self.config.requests_per_minute / 60.0)
        bucket.minute_tokens = min(
            bucket.minute_tokens + refill_min,
            float(self.config.requests_per_minute),
        )
        bucket.minute_last_refill = now

        # Refill hour bucket
        elapsed_hr = now - bucket.hour_last_refill
        refill_hr = elapsed_hr * (self.config.requests_per_hour / 3600.0)
        bucket.hour_tokens = min(
            bucket.hour_tokens + refill_hr,
            float(self.config.requests_per_hour),
        )
        bucket.hour_last_refill = now

    def check_rate_limit(self, user_id: str) -> Tuple[bool, Optional[float]]:
        """
        Check if a user is within rate limits.

        Returns:
            (allowed, retry_after) — allowed is True if the request can proceed.
            retry_after is the number of seconds to wait if rate-limited, else None.
        """
        bucket = self._get_bucket(user_id)
        self._refill(bucket)

        # Check both buckets
        if bucket.minute_tokens < 1.0:
            # Time until one minute token is available
            retry_after = (1.0 - bucket.minute_tokens) / (
                self.config.requests_per_minute / 60.0
            )
            return False, round(retry_after, 2)

        if bucket.hour_tokens < 1.0:
            retry_after = (1.0 - bucket.hour_tokens) / (
                self.config.requests_per_hour / 3600.0
            )
            return False, round(retry_after, 2)

        return True, None

    def consume(self, user_id: str) -> bool:
        """
        Consume a token from both buckets.
        Should be called after check_rate_limit() returns True.

        Returns True if consumption was successful.
        """
        allowed, _ = self.check_rate_limit(user_id)
        if not allowed:
            return False

        bucket = self._get_bucket(user_id)
        bucket.minute_tokens -= 1.0
        bucket.hour_tokens -= 1.0
        return True
