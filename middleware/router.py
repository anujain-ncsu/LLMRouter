"""
Router — orchestrates the full request lifecycle.

1. Validate input
2. Auto-select or validate the chosen model
3. Check circuit breaker
4. Dispatch to mock backend
5. Record token usage
6. Fallback on failure
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from middleware.audit_logger import AuditLogger
from middleware.circuit_breaker import CircuitBreaker
from middleware.input_guard import InputGuard, InputValidationError
from middleware.model_registry import ModelRegistry
from middleware.models import ChatRequest, ChatResponse, ModelRecommendation
from middleware.prompt_analyzer import analyze_prompt
from middleware.rate_limiter import RateLimiter
from middleware.token_accounting import TokenAccounting


class RoutingError(Exception):
    """Raised when routing fails after all retries."""
    def __init__(self, message: str, status_code: int = 503):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class Router:
    """
    Central routing engine.

    Coordinates input validation, model selection, circuit breaking,
    request dispatch, token accounting, and fallback logic.
    """

    MAX_FALLBACK_RETRIES = 2
    REQUEST_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        model_registry: ModelRegistry,
        circuit_breaker: CircuitBreaker,
        token_accounting: TokenAccounting,
        rate_limiter: RateLimiter,
        input_guard: InputGuard,
        audit_logger: AuditLogger,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.registry = model_registry
        self.circuit_breaker = circuit_breaker
        self.accounting = token_accounting
        self.rate_limiter = rate_limiter
        self.guard = input_guard
        self.audit = audit_logger
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.REQUEST_TIMEOUT_SECONDS)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def route(self, request: ChatRequest) -> ChatResponse:
        """
        Route a chat request end-to-end.

        Raises:
            InputValidationError: If the input is invalid.
            RoutingError: If routing fails after all retries.
        """
        start_time = time.monotonic()

        # 1. Validate input
        validated_uid, sanitized_prompt = self.guard.validate(
            request.user_id, request.prompt
        )

        # 2. Check rate limit
        allowed, retry_after = self.rate_limiter.check_rate_limit(validated_uid)
        if not allowed:
            self.audit.log_request(
                user_id=validated_uid,
                model_id="",
                prompt_length=len(sanitized_prompt),
                status="rate_limited",
                error_detail=f"Retry after {retry_after}s",
                latency_ms=(time.monotonic() - start_time) * 1000,
            )
            raise RoutingError(
                f"Rate limit exceeded. Retry after {retry_after} seconds.",
                status_code=429,
            )
        self.rate_limiter.consume(validated_uid)

        # 3. Determine model
        auto_selected = request.model_id is None
        recommendation: Optional[ModelRecommendation] = None

        if auto_selected:
            recommendation = analyze_prompt(sanitized_prompt)
            target_model_id = recommendation.model_id
        else:
            target_model_id = request.model_id
            if not self.registry.is_valid_model(target_model_id):
                raise RoutingError(
                    f"Unknown model: {target_model_id}",
                    status_code=400,
                )

        # 4. Build ordered list of models to try (primary + fallbacks)
        models_to_try = [target_model_id]
        fallbacks = self.registry.get_fallback_models(target_model_id)
        models_to_try.extend(m.id for m in fallbacks[:self.MAX_FALLBACK_RETRIES])

        # 5. Try each model in order
        last_error: Optional[str] = None
        for model_id in models_to_try:
            model = self.registry.get_model(model_id)
            if model is None:
                continue

            # Check circuit breaker
            if not self.circuit_breaker.is_available(model_id):
                last_error = f"Circuit breaker open for {model_id}"
                continue

            # Dispatch request
            try:
                client = await self._get_client()
                response = await client.post(
                    model.endpoint_url,
                    json={
                        "prompt": sanitized_prompt,
                        "max_tokens": min(request.max_tokens, model.max_tokens),
                    },
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                )

                if response.status_code == 200:
                    data = response.json()
                    model_tokens = data["tokens_used"]
                    system_tokens = model_tokens / model.cost_ratio
                    latency_ms = (time.monotonic() - start_time) * 1000

                    # Record success
                    self.circuit_breaker.record_success(model_id)
                    await self.accounting.record_usage(
                        user_id=validated_uid,
                        model_id=model_id,
                        model_tokens=model_tokens,
                        cost_ratio=model.cost_ratio,
                    )

                    self.audit.log_request(
                        user_id=validated_uid,
                        model_id=model_id,
                        prompt_length=len(sanitized_prompt),
                        model_tokens=model_tokens,
                        system_tokens=system_tokens,
                        latency_ms=latency_ms,
                        status="success",
                        auto_selected=auto_selected,
                    )

                    return ChatResponse(
                        response_text=data["response_text"],
                        model_id=model_id,
                        model_name=model.name,
                        tokens_used=model_tokens,
                        system_tokens_used=round(system_tokens, 2),
                        latency_ms=round(latency_ms, 2),
                        auto_selected=auto_selected,
                        recommendation=recommendation,
                    )
                else:
                    # Non-200 response
                    self.circuit_breaker.record_failure(model_id)
                    last_error = f"{model_id} returned {response.status_code}"

            except httpx.TimeoutException:
                self.circuit_breaker.record_failure(model_id)
                last_error = f"Timeout calling {model_id}"
            except httpx.HTTPError as e:
                self.circuit_breaker.record_failure(model_id)
                last_error = f"HTTP error calling {model_id}: {str(e)}"
            except Exception as e:
                self.circuit_breaker.record_failure(model_id)
                last_error = f"Unexpected error calling {model_id}: {str(e)}"

        # All models failed
        await self.accounting.record_error()
        latency_ms = (time.monotonic() - start_time) * 1000

        self.audit.log_request(
            user_id=validated_uid,
            model_id=target_model_id,
            prompt_length=len(sanitized_prompt),
            status="error",
            error_detail=last_error,
            latency_ms=latency_ms,
            auto_selected=auto_selected,
        )

        raise RoutingError(
            f"All models failed. Last error: {last_error}",
            status_code=503,
        )
