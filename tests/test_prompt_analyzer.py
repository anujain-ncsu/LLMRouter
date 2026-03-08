"""
Tests for Prompt Analyzer.
"""

import pytest

from middleware.prompt_analyzer import analyze_prompt


class TestPromptAnalyzer:
    """Tests for heuristic prompt classification."""

    def test_code_prompt_function(self):
        """Code prompt with function request → StellarCode."""
        result = analyze_prompt("Write a Python function to implement quicksort. def quicksort(arr): pass")
        assert result.model_id == "stellarcode-70b"
        assert result.confidence > 0.25

    def test_code_prompt_with_code_block(self):
        """Prompt with code block → StellarCode."""
        result = analyze_prompt("Debug this code:\n```python\ndef foo():\n    return bar\n```")
        assert result.model_id == "stellarcode-70b"

    def test_code_prompt_with_keywords(self):
        """Prompt with code keywords → StellarCode."""
        result = analyze_prompt("Implement a class called UserService with methods for CRUD operations")
        assert result.model_id == "stellarcode-70b"

    def test_reasoning_prompt(self):
        """Analytical prompt → QuantumLeap."""
        result = analyze_prompt(
            "Analyze the pros and cons of microservices architecture compared to "
            "monolithic architecture. Explain why one might be preferred over the other "
            "in different scenarios. What are the implications for team structure?"
        )
        assert result.model_id == "quantumleap-13b"

    def test_reasoning_prompt_with_comparison(self):
        """Comparison prompt → QuantumLeap."""
        result = analyze_prompt(
            "Compare and contrast the advantages and disadvantages of SQL vs NoSQL databases. "
            "Evaluate their performance characteristics and explain the trade-offs involved."
        )
        assert result.model_id == "quantumleap-13b"

    def test_casual_greeting(self):
        """Simple greeting → NebulaChat."""
        result = analyze_prompt("Hello! How are you?")
        assert result.model_id == "nebulachat-3b"

    def test_casual_short(self):
        """Very short casual prompt → NebulaChat."""
        result = analyze_prompt("Hey")
        assert result.model_id == "nebulachat-3b"

    def test_casual_thanks(self):
        """Thank you message → NebulaChat."""
        result = analyze_prompt("Thanks!")
        assert result.model_id == "nebulachat-3b"

    def test_general_prompt(self):
        """General knowledge prompt → NovaMind."""
        result = analyze_prompt("What is the capital of France?")
        assert result.model_id == "novamind-7b"

    def test_ambiguous_prompt_defaults_to_general(self):
        """Ambiguous prompt with moderate length defaults to NovaMind."""
        result = analyze_prompt("Tell me about the weather patterns in the Mediterranean region over the past decade")
        assert result.model_id == "novamind-7b"

    def test_returns_model_recommendation(self):
        """Returns a ModelRecommendation with all expected fields."""
        result = analyze_prompt("Hello there!")
        assert hasattr(result, 'model_id')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning')
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reasoning) > 0

    def test_confidence_threshold(self):
        """Low-signal prompts get default model with moderate confidence."""
        result = analyze_prompt("Hmm")
        # Short but not explicitly casual
        assert result.confidence > 0.0
