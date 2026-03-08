"""
Tests for Input Guard.
"""

import pytest

from middleware.input_guard import InputGuard, InputValidationError


@pytest.fixture
def guard():
    return InputGuard()


class TestInputGuard:
    """Tests for input validation and sanitization."""

    def test_valid_prompt(self, guard):
        """Valid prompt passes through."""
        result = guard.validate_prompt("What is machine learning?")
        assert result == "What is machine learning?"

    def test_empty_prompt_rejected(self, guard):
        """Empty string is rejected."""
        with pytest.raises(InputValidationError, match="empty"):
            guard.validate_prompt("")

    def test_whitespace_only_rejected(self, guard):
        """Whitespace-only string is rejected."""
        with pytest.raises(InputValidationError, match="empty"):
            guard.validate_prompt("   \n\t  ")

    def test_oversized_prompt_rejected(self, guard):
        """Prompt exceeding max length is rejected."""
        huge = "x" * 10_001
        with pytest.raises(InputValidationError, match="maximum length"):
            guard.validate_prompt(huge)

    def test_max_length_passes(self, guard):
        """Prompt at exactly max length passes."""
        prompt = "x" * 10_000
        result = guard.validate_prompt(prompt)
        assert len(result) == 10_000

    def test_strips_control_characters(self, guard):
        """Control characters are stripped (except newlines/tabs)."""
        prompt = "Hello\x00World\x07!"
        result = guard.validate_prompt(prompt)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "HelloWorld!" == result

    def test_preserves_newlines_and_tabs(self, guard):
        """Newlines and tabs are preserved."""
        prompt = "Line 1\nLine 2\tTabbed"
        result = guard.validate_prompt(prompt)
        assert "\n" in result
        assert "\t" in result

    def test_trims_whitespace(self, guard):
        """Leading/trailing whitespace is trimmed."""
        result = guard.validate_prompt("  Hello there  ")
        assert result == "Hello there"

    def test_invalid_characters_rejected(self, guard):
        """Replacement characters (bad encoding) are rejected."""
        with pytest.raises(InputValidationError, match="invalid"):
            guard.validate_prompt("Hello \ufffd World")

    def test_valid_user_id(self, guard):
        """Valid user ID passes."""
        result = guard.validate_user_id("user-1")
        assert result == "user-1"

    def test_empty_user_id_rejected(self, guard):
        """Empty user ID is rejected."""
        with pytest.raises(InputValidationError, match="empty"):
            guard.validate_user_id("")

    def test_special_chars_in_user_id_rejected(self, guard):
        """Special characters in user ID are rejected."""
        with pytest.raises(InputValidationError, match="letters, numbers"):
            guard.validate_user_id("user@1!")

    def test_user_id_allows_dots_hyphens_underscores(self, guard):
        """User ID allows dots, hyphens, and underscores."""
        result = guard.validate_user_id("user.name-1_2")
        assert result == "user.name-1_2"

    def test_validate_both(self, guard):
        """validate() validates both user_id and prompt."""
        uid, prompt = guard.validate("user-1", "  Hello!  ")
        assert uid == "user-1"
        assert prompt == "Hello!"
