# NexusAI Router

NexusAI Router is an LLM routing platform built with FastAPI that provides auto-model-selection, fault tolerance, and a robust middleware pipeline for requests to Large Language Models.

The system includes a mock backend for testing, a web dashboard for metrics and interaction, and an extensible pipeline architecture designed to handle rate limiting, token accounting, circuit breaking, input validation, and audit logging.

## Project Structure

- `middleware/`: Core pipeline components for the LLM router. Contains modules for rate limiting, token accounting, circuit breaking, input guarding, audit logging, and the model registry.
- `mock_backends/`: Mock LLM API servers to simulate interactions with various LLM providers during development and testing.
- `server/`: The main FastAPI application (`app.py`) that serves the REST API and mounts the static frontend.
- `static/`: Frontend dashboard assets (HTML, CSS, JS) that provide a user interface for interacting with the router and viewing system health.
- `tests/`: Pytest test suite for validating core functionality and middleware components.
- `run.py`: The entry point script to start the application.
- `requirements.txt`: Project dependencies.
- `Procfile` & `runtime.txt`: Configuration files for deploying to Heroku.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLMRouter
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   Ensure you have Python 3.12+ installed as specified in `runtime.txt`.
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally

To start the NexusAI Router locally, run the main entry point:

```bash
python run.py
```

This will start:
- The main API server on `http://localhost:8000` (or the port specified by the `$PORT` environment variable).
- The mock LLM backend, which is automatically mounted at `http://localhost:8000/mock`.

### Usage

- **Web Dashboard**: Open `http://localhost:8000` in your browser to access the interactive web dashboard.
- **REST API Subsystem**: 
  - Chat Endpoint: `POST /api/chat`
  - Models Status: `GET /api/models`
  - System Health: `GET /api/health`
  - Token Usage: `GET /api/usage/{user_id}`
  - Admin Metrics: `GET /api/admin/metrics`

## Deployment to Heroku

The application is pre-configured for deployment to Heroku using the included `Procfile` and `runtime.txt`.

1. **Install the Heroku CLI**: 
   Download and install the [Heroku Command Line Interface](https://devcenter.heroku.com/articles/heroku-cli).

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create a new Heroku app**:
   ```bash
   heroku create your-app-name
   ```

4. **Deploy the application**:
   Push the code to the Heroku remote repository:
   ```bash
   git push heroku main
   ```

5. **Open the deployed app**:
   Once the build finishes and the application is deployed, you can open it in your browser:
   ```bash
   heroku open
   ```

By default, Heroku will use the `Procfile` to start the web process (`python run.py`) and bind it to the dynamic port it assigns.
