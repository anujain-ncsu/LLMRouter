"""
Pydantic models and data classes for the LLM Router system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── API Request / Response Schemas ──────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""
    user_id: str = Field(..., min_length=1, max_length=128)
    prompt: str = Field(..., min_length=1)
    model_id: Optional[str] = Field(
        None,
        description="Specific model to use. If None, auto-select is used.",
    )
    max_tokens: int = Field(1024, ge=1, le=8192)


class ModelRecommendation(BaseModel):
    """Result of prompt analysis for auto-model-selection."""
    model_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class ChatResponse(BaseModel):
    """Response returned to the frontend."""
    response_text: str
    model_id: str
    model_name: str
    tokens_used: int
    system_tokens_used: float
    latency_ms: float
    auto_selected: bool
    recommendation: Optional[ModelRecommendation] = None


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str
    detail: str
    retry_after: Optional[float] = None


# ── Model Registry Types ───────────────────────────────────────────────────

class ModelCapability(str, Enum):
    GENERAL = "general"
    FAST = "fast"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CODE = "code"
    COMPLEX = "complex"
    CHAT = "chat"
    CASUAL = "casual"


@dataclass
class ModelInfo:
    """Metadata about a registered LLM model."""
    id: str
    name: str
    description: str
    endpoint_url: str
    cost_ratio: float  # 1 system token = cost_ratio model tokens
    capabilities: list[ModelCapability] = field(default_factory=list)
    max_tokens: int = 4096
    priority: int = 0  # Lower = preferred for fallbacks


# ── Token Accounting Types ─────────────────────────────────────────────────

@dataclass
class TokenUsageRecord:
    """Token usage record for a single model by a single user."""
    user_id: str
    model_id: str
    model_tokens: int = 0
    system_tokens: float = 0.0
    request_count: int = 0


@dataclass
class UserUsageSummary:
    """Aggregated token usage summary for a user."""
    user_id: str
    total_model_tokens: int = 0
    total_system_tokens: float = 0.0
    total_requests: int = 0
    per_model: dict[str, TokenUsageRecord] = field(default_factory=dict)


@dataclass
class ModelUsageSummary:
    """Aggregated token usage summary for a model across all users."""
    model_id: str
    total_model_tokens: int = 0
    total_system_tokens: float = 0.0
    total_requests: int = 0


# ── Circuit Breaker Types ─────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerStatus:
    """Status of a circuit breaker for a single model."""
    model_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    last_failure_time: Optional[float] = None
    cooloff_until: Optional[float] = None


# ── Health / Admin Types ───────────────────────────────────────────────────

class ModelHealthStatus(BaseModel):
    """Health status for a single model (API response)."""
    model_id: str
    model_name: str
    circuit_state: str
    available: bool
    failure_rate: float
    cooloff_remaining_seconds: Optional[float] = None


class SystemHealthResponse(BaseModel):
    """System-wide health response."""
    status: str
    models: list[ModelHealthStatus]
    total_requests_served: int
    uptime_seconds: float


class SystemMetrics(BaseModel):
    """Aggregate metrics for admin dashboard."""
    total_requests: int
    total_model_tokens: int
    total_system_tokens: float
    total_errors: int
    error_rate: float
    per_model_metrics: dict[str, dict]
