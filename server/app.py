"""
FastAPI application — API gateway for the LLM Router system.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from middleware.audit_logger import AuditLogger
from middleware.circuit_breaker import CircuitBreaker
from middleware.input_guard import InputGuard, InputValidationError
from middleware.model_registry import ModelRegistry
from middleware.models import ChatRequest, ChatResponse, SystemHealthResponse, ModelHealthStatus, CircuitState
from middleware.rate_limiter import RateLimiter
from middleware.router import Router, RoutingError
from middleware.token_accounting import TokenAccounting

# ── Shared instances ───────────────────────────────────────────────────────

model_registry = ModelRegistry()
circuit_breaker = CircuitBreaker()
token_accounting = TokenAccounting()
rate_limiter = RateLimiter()
input_guard = InputGuard()
audit_logger = AuditLogger()

_start_time = time.monotonic()

router_engine = None  # type: Optional[Router]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown of the router engine."""
    global router_engine
    router_engine = Router(
        model_registry=model_registry,
        circuit_breaker=circuit_breaker,
        token_accounting=token_accounting,
        rate_limiter=rate_limiter,
        input_guard=input_guard,
        audit_logger=audit_logger,
    )
    yield
    await router_engine.close()


app = FastAPI(
    title="NexusAI Router",
    description="LLM routing platform with auto-model-selection and fault tolerance",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat prompt to an LLM model."""
    try:
        response = await router_engine.route(request)
        return response
    except InputValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except RoutingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@app.get("/api/models")
async def list_models():
    """List all available models with their current status."""
    models = model_registry.list_models()
    result = []
    for model in models:
        cb_status = circuit_breaker.get_status(model.id)
        result.append({
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "cost_ratio": model.cost_ratio,
            "capabilities": [c.value for c in model.capabilities],
            "max_tokens": model.max_tokens,
            "available": circuit_breaker.is_available(model.id),
            "circuit_state": cb_status.state.value,
        })
    return {"models": result}


@app.get("/api/usage/{user_id}")
async def get_usage(user_id: str):
    """Get token usage summary for a user."""
    summary = await token_accounting.get_user_usage(user_id)
    return {
        "user_id": summary.user_id,
        "total_model_tokens": summary.total_model_tokens,
        "total_system_tokens": round(summary.total_system_tokens, 2),
        "total_requests": summary.total_requests,
        "per_model": {
            mid: {
                "model_tokens": rec.model_tokens,
                "system_tokens": round(rec.system_tokens, 2),
                "request_count": rec.request_count,
            }
            for mid, rec in summary.per_model.items()
        },
    }


@app.get("/api/health", response_model=SystemHealthResponse)
async def health():
    """System health check with circuit breaker statuses."""
    models = model_registry.list_models()
    now = time.monotonic()
    model_statuses = []
    for model in models:
        cb = circuit_breaker.get_status(model.id)
        cooloff_remaining = None
        if cb.cooloff_until and cb.cooloff_until > now:
            cooloff_remaining = round(cb.cooloff_until - now, 1)
        model_statuses.append(
            ModelHealthStatus(
                model_id=model.id,
                model_name=model.name,
                circuit_state=cb.state.value,
                available=circuit_breaker.is_available(model.id),
                failure_rate=round(circuit_breaker.get_failure_rate(model.id), 3),
                cooloff_remaining_seconds=cooloff_remaining,
            )
        )

    total_requests = await token_accounting.get_total_requests()

    return SystemHealthResponse(
        status="healthy" if any(ms.available for ms in model_statuses) else "degraded",
        models=model_statuses,
        total_requests_served=total_requests,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@app.get("/api/admin/metrics")
async def admin_metrics():
    """Aggregate metrics for admin dashboard."""
    total_requests = await token_accounting.get_total_requests()
    total_errors = await token_accounting.get_total_errors()
    error_rate = await token_accounting.get_error_rate()

    per_model = {}
    for model in model_registry.list_models():
        usage = await token_accounting.get_model_usage(model.id)
        per_model[model.id] = {
            "model_name": model.name,
            "total_model_tokens": usage.total_model_tokens,
            "total_system_tokens": round(usage.total_system_tokens, 2),
            "total_requests": usage.total_requests,
            "circuit_state": circuit_breaker.get_status(model.id).state.value,
            "failure_rate": round(circuit_breaker.get_failure_rate(model.id), 3),
        }

    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(error_rate, 4),
        "per_model_metrics": per_model,
    }


# ── Static files (served LAST so API routes take priority) ─────────────────

import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
