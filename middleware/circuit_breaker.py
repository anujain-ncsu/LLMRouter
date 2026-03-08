"""
Circuit Breaker — fault-tolerant control plane for LLM model backends.

Implements a per-model circuit breaker with three states:
  CLOSED  → normal operation, tracking failures in a sliding window
  OPEN    → model is unhealthy, requests rejected immediately, cooloff timer running
  HALF_OPEN → cooloff expired, one probe request allowed to test recovery
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

from middleware.models import CircuitBreakerStatus, CircuitState


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: float = 0.30       # Trip when >30% of recent requests fail
    window_size: int = 20                 # Number of recent requests to track
    window_duration_seconds: float = 60.0 # Only consider requests from last 60s
    cooloff_seconds: float = 30.0         # How long to stay OPEN before HALF_OPEN
    half_open_max_probes: int = 1         # Number of probe requests in HALF_OPEN


@dataclass
class _ModelCircuit:
    """Internal state for a single model's circuit breaker."""
    model_id: str
    state: CircuitState = CircuitState.CLOSED
    # Sliding window: deque of (timestamp, success: bool)
    history: deque = field(default_factory=deque)
    cooloff_until: float = 0.0
    half_open_probes: int = 0

    def _prune_history(self, now: float, window_duration: float):
        """Remove entries older than the sliding window."""
        cutoff = now - window_duration
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()


class CircuitBreaker:
    """
    Per-model circuit breaker.

    Thread-safe for single-threaded async use (no locks needed in
    single-event-loop asyncio, but logic is safe regardless).
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._circuits: dict[str, _ModelCircuit] = {}

    def _get_circuit(self, model_id: str) -> _ModelCircuit:
        if model_id not in self._circuits:
            self._circuits[model_id] = _ModelCircuit(model_id=model_id)
        return self._circuits[model_id]

    def is_available(self, model_id: str) -> bool:
        """Check if a model is available for requests."""
        circuit = self._get_circuit(model_id)
        now = time.monotonic()

        if circuit.state == CircuitState.CLOSED:
            return True

        if circuit.state == CircuitState.OPEN:
            # Check if cooloff period has expired
            if now >= circuit.cooloff_until:
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_probes = 0
                return True
            return False

        if circuit.state == CircuitState.HALF_OPEN:
            # Allow limited probe requests
            return circuit.half_open_probes < self.config.half_open_max_probes

        return False

    def record_success(self, model_id: str) -> None:
        """Record a successful request to a model."""
        circuit = self._get_circuit(model_id)
        now = time.monotonic()

        if circuit.state == CircuitState.HALF_OPEN:
            # Probe succeeded → close the circuit
            circuit.state = CircuitState.CLOSED
            circuit.history.clear()
            circuit.half_open_probes = 0
        else:
            circuit.history.append((now, True))
            circuit._prune_history(now, self.config.window_duration_seconds)

    def record_failure(self, model_id: str) -> None:
        """Record a failed request to a model."""
        circuit = self._get_circuit(model_id)
        now = time.monotonic()

        if circuit.state == CircuitState.HALF_OPEN:
            # Probe failed → re-open the circuit
            circuit.state = CircuitState.OPEN
            circuit.cooloff_until = now + self.config.cooloff_seconds
            circuit.half_open_probes = 0
            return

        circuit.history.append((now, False))
        circuit._prune_history(now, self.config.window_duration_seconds)

        # Check if we should trip the breaker
        if len(circuit.history) >= self.config.window_size:
            failures = sum(1 for _, success in circuit.history if not success)
            failure_rate = failures / len(circuit.history)
            if failure_rate > self.config.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.cooloff_until = now + self.config.cooloff_seconds

    def get_status(self, model_id: str) -> CircuitBreakerStatus:
        """Get the current status of a model's circuit breaker."""
        circuit = self._get_circuit(model_id)
        now = time.monotonic()
        circuit._prune_history(now, self.config.window_duration_seconds)

        failures = sum(1 for _, s in circuit.history if not s)
        successes = sum(1 for _, s in circuit.history if s)
        total = len(circuit.history)

        cooloff_until = None
        if circuit.state == CircuitState.OPEN and circuit.cooloff_until > now:
            cooloff_until = circuit.cooloff_until

        return CircuitBreakerStatus(
            model_id=model_id,
            state=circuit.state,
            failure_count=failures,
            success_count=successes,
            total_requests=total,
            last_failure_time=circuit.history[-1][0] if circuit.history and not circuit.history[-1][1] else None,
            cooloff_until=cooloff_until,
        )

    def get_all_statuses(self, model_ids: List[str]) -> dict[str, CircuitBreakerStatus]:
        """Get status for all given model IDs."""
        return {mid: self.get_status(mid) for mid in model_ids}

    def get_failure_rate(self, model_id: str) -> float:
        """Get the current failure rate for a model."""
        circuit = self._get_circuit(model_id)
        now = time.monotonic()
        circuit._prune_history(now, self.config.window_duration_seconds)
        if not circuit.history:
            return 0.0
        failures = sum(1 for _, s in circuit.history if not s)
        return failures / len(circuit.history)
