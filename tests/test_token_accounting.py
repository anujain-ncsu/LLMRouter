"""
Tests for Token Accounting.
"""

import asyncio
import pytest

from middleware.token_accounting import TokenAccounting


@pytest.fixture
def accounting():
    return TokenAccounting()


class TestTokenAccounting:
    """Tests for the TokenAccounting system."""

    @pytest.mark.asyncio
    async def test_record_usage_basic(self, accounting):
        """Recording usage creates a correct token record."""
        record = await accounting.record_usage("user-1", "novamind-7b", 100, 2.0)
        assert record.user_id == "user-1"
        assert record.model_id == "novamind-7b"
        assert record.model_tokens == 100
        assert record.system_tokens == 50.0  # 100 / 2.0
        assert record.request_count == 1

    @pytest.mark.asyncio
    async def test_cost_translation_cheap_model(self, accounting):
        """Cheap model: 1 system token = 2 model tokens → 100 model = 50 system."""
        record = await accounting.record_usage("user-1", "novamind-7b", 100, 2.0)
        assert record.system_tokens == 50.0

    @pytest.mark.asyncio
    async def test_cost_translation_standard_model(self, accounting):
        """Standard model: 1:1 → 100 model = 100 system."""
        record = await accounting.record_usage("user-1", "quantumleap-13b", 100, 1.0)
        assert record.system_tokens == 100.0

    @pytest.mark.asyncio
    async def test_cost_translation_expensive_model(self, accounting):
        """Expensive model: 1 system token = 0.5 model tokens → 100 model = 200 system."""
        record = await accounting.record_usage("user-1", "stellarcode-70b", 100, 0.5)
        assert record.system_tokens == 200.0

    @pytest.mark.asyncio
    async def test_cost_translation_very_cheap_model(self, accounting):
        """Very cheap model: 1 system token = 3 model tokens → 300 model = 100 system."""
        record = await accounting.record_usage("user-1", "nebulachat-3b", 300, 3.0)
        assert record.system_tokens == 100.0

    @pytest.mark.asyncio
    async def test_accumulates_across_requests(self, accounting):
        """Multiple requests to the same model accumulate tokens."""
        await accounting.record_usage("user-1", "novamind-7b", 100, 2.0)
        record = await accounting.record_usage("user-1", "novamind-7b", 200, 2.0)
        assert record.model_tokens == 300
        assert record.system_tokens == 150.0  # (100 + 200) / 2.0
        assert record.request_count == 2

    @pytest.mark.asyncio
    async def test_user_usage_summary(self, accounting):
        """User usage summary aggregates across models."""
        await accounting.record_usage("user-1", "novamind-7b", 100, 2.0)
        await accounting.record_usage("user-1", "stellarcode-70b", 50, 0.5)

        summary = await accounting.get_user_usage("user-1")
        assert summary.user_id == "user-1"
        assert summary.total_model_tokens == 150  # 100 + 50
        assert summary.total_system_tokens == 150.0  # 50 + 100
        assert summary.total_requests == 2
        assert len(summary.per_model) == 2

    @pytest.mark.asyncio
    async def test_model_usage_summary(self, accounting):
        """Model usage summary aggregates across users."""
        await accounting.record_usage("user-1", "novamind-7b", 100, 2.0)
        await accounting.record_usage("user-2", "novamind-7b", 200, 2.0)

        summary = await accounting.get_model_usage("novamind-7b")
        assert summary.model_id == "novamind-7b"
        assert summary.total_model_tokens == 300
        assert summary.total_requests == 2

    @pytest.mark.asyncio
    async def test_empty_user_usage(self, accounting):
        """Empty usage returns zero summary."""
        summary = await accounting.get_user_usage("nonexistent")
        assert summary.total_model_tokens == 0
        assert summary.total_system_tokens == 0.0
        assert summary.total_requests == 0

    @pytest.mark.asyncio
    async def test_error_tracking(self, accounting):
        """Error tracking increments error count and total requests."""
        await accounting.record_error()
        await accounting.record_error()
        assert await accounting.get_total_errors() == 2
        assert await accounting.get_total_requests() == 2
        assert await accounting.get_error_rate() == 1.0

    @pytest.mark.asyncio
    async def test_error_rate_with_mixed_requests(self, accounting):
        """Error rate is calculated correctly with mixed successes and errors."""
        await accounting.record_usage("user-1", "novamind-7b", 100, 2.0)
        await accounting.record_usage("user-1", "novamind-7b", 100, 2.0)
        await accounting.record_error()

        rate = await accounting.get_error_rate()
        assert abs(rate - 1.0 / 3.0) < 0.01

    @pytest.mark.asyncio
    async def test_concurrent_access(self, accounting):
        """Concurrent usage recording doesn't lose data."""
        async def record(uid):
            for _ in range(50):
                await accounting.record_usage(uid, "novamind-7b", 10, 2.0)

        await asyncio.gather(record("u1"), record("u2"), record("u3"))

        total = await accounting.get_total_requests()
        assert total == 150

        for uid in ["u1", "u2", "u3"]:
            summary = await accounting.get_user_usage(uid)
            assert summary.total_model_tokens == 500
            assert summary.total_requests == 50
