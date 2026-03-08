"""
Error injection logic for mock LLM backends.

Controls error rate, error type distribution, and per-model overrides
to simulate real-world LLM service behavior.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorType(Enum):
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503
    TOO_MANY_REQUESTS = 429
    REQUEST_TIMEOUT = 408


# Default distribution weights for each error type
DEFAULT_ERROR_DISTRIBUTION: dict[ErrorType, float] = {
    ErrorType.INTERNAL_SERVER_ERROR: 0.30,
    ErrorType.SERVICE_UNAVAILABLE: 0.30,
    ErrorType.TOO_MANY_REQUESTS: 0.25,
    ErrorType.REQUEST_TIMEOUT: 0.15,
}


@dataclass
class ErrorInjectorConfig:
    """Configuration for error injection."""
    base_error_rate: float = 0.03  # 3% default
    error_distribution: dict[ErrorType, float] = field(
        default_factory=lambda: dict(DEFAULT_ERROR_DISTRIBUTION)
    )
    per_model_error_rate: dict[str, float] = field(default_factory=dict)


class ErrorInjector:
    """
    Injects errors into mock LLM backend responses.

    Randomly decides whether a request should fail, and if so,
    which type of error to return. Supports per-model overrides.
    """

    def __init__(self, config: Optional[ErrorInjectorConfig] = None):
        self.config = config or ErrorInjectorConfig()

    def should_inject_error(self, model_id: str) -> bool:
        """Determine if this request should be an error."""
        error_rate = self.config.per_model_error_rate.get(
            model_id, self.config.base_error_rate
        )
        return random.random() < error_rate

    def get_error_type(self) -> ErrorType:
        """Pick an error type based on the configured distribution."""
        types = list(self.config.error_distribution.keys())
        weights = list(self.config.error_distribution.values())
        return random.choices(types, weights=weights, k=1)[0]

    def get_error_response(self, model_id: str) -> Optional[dict]:
        """
        Returns an error response dict if an error should be injected,
        or None if the request should succeed.

        Returns:
            dict with 'status_code', 'error_type', and 'message' keys, or None.
        """
        if not self.should_inject_error(model_id):
            return None

        error_type = self.get_error_type()
        messages = {
            ErrorType.INTERNAL_SERVER_ERROR: f"Internal server error in model {model_id}",
            ErrorType.SERVICE_UNAVAILABLE: f"Model {model_id} is temporarily unavailable",
            ErrorType.TOO_MANY_REQUESTS: f"Rate limit exceeded for model {model_id}",
            ErrorType.REQUEST_TIMEOUT: f"Request to model {model_id} timed out",
        }

        return {
            "status_code": error_type.value,
            "error_type": error_type.name,
            "message": messages[error_type],
        }
