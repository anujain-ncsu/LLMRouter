"""
Mock LLM Backend Server.

Simulates 4 fictional LLM model endpoints. Each returns prompt-agnostic
but structurally valid responses for most requests, and injects errors
2-4% of the time to simulate real-world LLM service behavior.
"""

import asyncio
import random
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mock_backends.error_injector import ErrorInjector, ErrorInjectorConfig

# ── Canned response fragments ──────────────────────────────────────────────
RESPONSE_FRAGMENTS = [
    "Based on the analysis of the available information, ",
    "Here's a comprehensive overview: ",
    "I'd be happy to help with that. ",
    "Let me break this down step by step. ",
    "That's an interesting question. ",
    "After careful consideration, ",
    "There are several key points to consider. ",
    "The short answer is that it depends on the context. ",
    "Here's what I found: ",
    "Great question! Let me walk you through this. ",
]

RESPONSE_BODIES = [
    "The key factors include scalability, maintainability, and performance. "
    "Each of these aspects plays a crucial role in determining the overall "
    "effectiveness of the solution. When we consider the trade-offs involved, "
    "it becomes clear that a balanced approach yields the best results.",

    "There are multiple approaches to solving this problem. The first involves "
    "a direct computation method, which is straightforward but may not scale well. "
    "The second approach uses a more sophisticated algorithm that trades some "
    "complexity for better performance characteristics.",

    "This is a well-studied topic in the field. The consensus among experts is "
    "that the optimal strategy depends heavily on the specific constraints of "
    "your use case. I recommend starting with the simplest approach and iterating "
    "based on measured performance metrics.",

    "Let me provide a detailed explanation. The fundamental concept here revolves "
    "around the interaction between different components of the system. Each component "
    "has its own responsibilities and communicates with others through well-defined "
    "interfaces, ensuring loose coupling and high cohesion.",

    "The implementation follows a standard pattern commonly used in production systems. "
    "First, we initialize the necessary data structures. Then, we process the input "
    "through a pipeline of transformations. Finally, we validate the output against "
    "the expected schema before returning the result.",
]

RESPONSE_CLOSINGS = [
    " Let me know if you'd like me to elaborate on any of these points.",
    " I hope this helps! Feel free to ask follow-up questions.",
    " Would you like me to go deeper into any specific aspect?",
    " This should give you a solid foundation to work from.",
    " Please don't hesitate to ask if anything is unclear.",
]

# ── Model-specific flavor text ─────────────────────────────────────────────
MODEL_FLAVORS = {
    "novamind-7b": "As a general-purpose assistant, ",
    "quantumleap-13b": "From an analytical perspective, ",
    "stellarcode-70b": "From a technical implementation standpoint, ",
    "nebulachat-3b": "Hey! So basically, ",
}


def _generate_response(model_id: str) -> tuple[str, int]:
    """Generate a canned response and estimated token count."""
    flavor = MODEL_FLAVORS.get(model_id, "")
    fragment = random.choice(RESPONSE_FRAGMENTS)
    body = random.choice(RESPONSE_BODIES)
    closing = random.choice(RESPONSE_CLOSINGS)

    text = flavor + fragment + body + closing

    # Rough token estimation: ~4 chars per token
    token_count = len(text) // 4 + random.randint(5, 25)
    return text, token_count


# ── FastAPI App ─────────────────────────────────────────────────────────────

VALID_MODELS = {"novamind-7b", "quantumleap-13b", "stellarcode-70b", "nebulachat-3b"}

error_injector = ErrorInjector(ErrorInjectorConfig(base_error_rate=0.03))

app = FastAPI(title="Mock LLM Backends", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024


class GenerateResponse(BaseModel):
    model_id: str
    response_text: str
    tokens_used: int
    latency_ms: float


@app.post("/models/{model_id}/generate", response_model=GenerateResponse)
async def generate(model_id: str, request: GenerateRequest):
    """Generate a response from a mock LLM model."""
    if model_id not in VALID_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    start_time = time.monotonic()

    # Simulate processing latency (50-500ms)
    latency = random.uniform(0.05, 0.5)
    await asyncio.sleep(latency)

    # Check for error injection
    error = error_injector.get_error_response(model_id)
    if error:
        raise HTTPException(
            status_code=error["status_code"],
            detail=error["message"],
        )

    # Generate a successful response
    response_text, tokens_used = _generate_response(model_id)

    # Cap tokens to max_tokens
    if tokens_used > request.max_tokens:
        tokens_used = request.max_tokens

    elapsed_ms = (time.monotonic() - start_time) * 1000

    return GenerateResponse(
        model_id=model_id,
        response_text=response_text,
        tokens_used=tokens_used,
        latency_ms=round(elapsed_ms, 2),
    )


@app.get("/health")
async def health():
    """Health check for mock backend."""
    return {"status": "healthy", "models": list(VALID_MODELS)}
