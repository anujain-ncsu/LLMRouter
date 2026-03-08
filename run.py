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
    """Run the main API server on port 8000."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    # Start mock backend in a separate process
    backend_process = multiprocessing.Process(target=run_mock_backend, daemon=True)
    backend_process.start()

    # Give the backend a moment to start
    time.sleep(1)
    print("✓ Mock LLM backends started on port 8100")
    print("✓ Starting NexusAI Router on http://localhost:8000")
    print()

    try:
        run_main_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_process.terminate()
        backend_process.join()
