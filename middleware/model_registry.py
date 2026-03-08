"""
Model Registry — maintains metadata for all available LLM models.
"""

from __future__ import annotations

from typing import List, Optional

from middleware.models import ModelCapability, ModelInfo

# ── Default Model Definitions ──────────────────────────────────────────────

_DEFAULT_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="novamind-7b",
        name="NovaMind 7B",
        description="Versatile general-purpose model. Good balance of speed and quality.",
        endpoint_url=f"http://127.0.0.1:{__import__('os').environ.get('PORT', 8000)}/mock/models/novamind-7b/generate",
        cost_ratio=2.0,  # Cheap: 1 system token = 2 model tokens
        capabilities=[ModelCapability.GENERAL, ModelCapability.FAST],
        max_tokens=4096,
        priority=1,
    ),
    ModelInfo(
        id="quantumleap-13b",
        name="QuantumLeap 13B",
        description="Excels at reasoning, analysis, and complex question answering.",
        endpoint_url=f"http://127.0.0.1:{__import__('os').environ.get('PORT', 8000)}/mock/models/quantumleap-13b/generate",
        cost_ratio=1.0,  # Standard: 1 system token = 1 model token
        capabilities=[ModelCapability.REASONING, ModelCapability.ANALYSIS],
        max_tokens=8192,
        priority=2,
    ),
    ModelInfo(
        id="stellarcode-70b",
        name="StellarCode 70B",
        description="Premium model for code generation, debugging, and technical tasks.",
        endpoint_url=f"http://127.0.0.1:{__import__('os').environ.get('PORT', 8000)}/mock/models/stellarcode-70b/generate",
        cost_ratio=0.5,  # Expensive: 1 system token = 0.5 model tokens
        capabilities=[ModelCapability.CODE, ModelCapability.COMPLEX],
        max_tokens=8192,
        priority=3,
    ),
    ModelInfo(
        id="nebulachat-3b",
        name="NebulaChat 3B",
        description="Lightweight model for casual conversation and simple queries.",
        endpoint_url=f"http://127.0.0.1:{__import__('os').environ.get('PORT', 8000)}/mock/models/nebulachat-3b/generate",
        cost_ratio=3.0,  # Very cheap: 1 system token = 3 model tokens
        capabilities=[ModelCapability.CHAT, ModelCapability.CASUAL],
        max_tokens=2048,
        priority=0,
    ),
]


class ModelRegistry:
    """
    Registry of all available LLM models.

    Maintains metadata about each model including cost ratios,
    capabilities, and endpoint URLs.
    """

    def __init__(self, models: Optional[List[ModelInfo]] = None):
        self._models: dict[str, ModelInfo] = {}
        for model in (models or _DEFAULT_MODELS):
            self._models[model.id] = model

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a model by its ID."""
        return self._models.get(model_id)

    def list_models(self) -> List[ModelInfo]:
        """List all registered models, sorted by priority."""
        return sorted(self._models.values(), key=lambda m: m.priority)

    def list_model_ids(self) -> List[str]:
        """List all registered model IDs."""
        return [m.id for m in self.list_models()]

    def get_models_by_capability(self, capability: ModelCapability) -> List[ModelInfo]:
        """Get all models that have a given capability."""
        return [
            m for m in self._models.values()
            if capability in m.capabilities
        ]

    def get_fallback_models(self, exclude_model_id: str) -> List[ModelInfo]:
        """
        Get models to try as fallbacks, excluding the given model.
        Sorted by priority (lower priority number = preferred fallback).
        """
        return sorted(
            [m for m in self._models.values() if m.id != exclude_model_id],
            key=lambda m: m.priority,
        )

    def is_valid_model(self, model_id: str) -> bool:
        """Check if a model ID is registered."""
        return model_id in self._models
