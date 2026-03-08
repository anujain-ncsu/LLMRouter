"""
Entry point for the LLM Router system.

Starts both the mock LLM backend server and the main API server.
"""

import multiprocessing
import time
import uvicorn


def run_mock_backend():
    """Run the mock LLM backend server on port 8100."""
    uvicorn.run(
        "mock_backends.server:app",
        host="0.0.0.0",
        port=8100,
        log_level="warning",
    )


def run_main_server():
    """Run the main API server on port 8000 (or $PORT)."""
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    print("✓ Starting NexusAI Router on http://localhost:8000")
    print("✓ Mock LLM backends are mounted at /mock")
    print()

    try:
        run_main_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
