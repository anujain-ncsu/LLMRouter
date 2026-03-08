"""
Token Accounting — tracks token usage per user and per model,
with cost translation between system tokens and model-native tokens.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from middleware.models import (
    ModelUsageSummary,
    TokenUsageRecord,
    UserUsageSummary,
)


class TokenAccounting:
    """
    Thread-safe token accounting system.

    Tracks model-native tokens per (user, model) pair and translates them
    to system tokens using the model's cost ratio.

    Cost translation formula:
        system_tokens = model_tokens / cost_ratio

    Examples:
        - NovaMind 7B (cost_ratio=2.0): 100 model tokens = 50 system tokens
        - StellarCode 70B (cost_ratio=0.5): 100 model tokens = 200 system tokens
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        # (user_id, model_id) → TokenUsageRecord
        self._usage: dict[tuple[str, str], TokenUsageRecord] = {}
        self._total_requests: int = 0
        self._total_errors: int = 0

    async def record_usage(
        self,
        user_id: str,
        model_id: str,
        model_tokens: int,
        cost_ratio: float,
    ) -> TokenUsageRecord:
        """
        Record token usage for a user on a specific model.

        Args:
            user_id: The user's identifier.
            model_id: The model that was called.
            model_tokens: Number of model-native tokens consumed.
            cost_ratio: The model's cost ratio for system token conversion.

        Returns:
            Updated TokenUsageRecord for this (user, model) pair.
        """
        system_tokens = model_tokens / cost_ratio

        async with self._lock:
            key = (user_id, model_id)
            if key not in self._usage:
                self._usage[key] = TokenUsageRecord(
                    user_id=user_id, model_id=model_id
                )

            record = self._usage[key]
            record.model_tokens += model_tokens
            record.system_tokens += system_tokens
            record.request_count += 1
            self._total_requests += 1

            return TokenUsageRecord(
                user_id=record.user_id,
                model_id=record.model_id,
                model_tokens=record.model_tokens,
                system_tokens=record.system_tokens,
                request_count=record.request_count,
            )

    async def record_error(self) -> None:
        """Record that an error occurred (for metrics)."""
        async with self._lock:
            self._total_errors += 1
            self._total_requests += 1

    async def get_user_usage(self, user_id: str) -> UserUsageSummary:
        """Get aggregated usage summary for a user."""
        async with self._lock:
            summary = UserUsageSummary(user_id=user_id)
            for (uid, model_id), record in self._usage.items():
                if uid == user_id:
                    summary.total_model_tokens += record.model_tokens
                    summary.total_system_tokens += record.system_tokens
                    summary.total_requests += record.request_count
                    summary.per_model[model_id] = TokenUsageRecord(
                        user_id=record.user_id,
                        model_id=record.model_id,
                        model_tokens=record.model_tokens,
                        system_tokens=record.system_tokens,
                        request_count=record.request_count,
                    )
            return summary

    async def get_model_usage(self, model_id: str) -> ModelUsageSummary:
        """Get aggregated usage summary for a model across all users."""
        async with self._lock:
            summary = ModelUsageSummary(model_id=model_id)
            for (uid, mid), record in self._usage.items():
                if mid == model_id:
                    summary.total_model_tokens += record.model_tokens
                    summary.total_system_tokens += record.system_tokens
                    summary.total_requests += record.request_count
            return summary

    async def get_total_requests(self) -> int:
        """Get total number of requests served."""
        async with self._lock:
            return self._total_requests

    async def get_total_errors(self) -> int:
        """Get total number of errors."""
        async with self._lock:
            return self._total_errors

    async def get_error_rate(self) -> float:
        """Get overall error rate."""
        async with self._lock:
            if self._total_requests == 0:
                return 0.0
            return self._total_errors / self._total_requests
