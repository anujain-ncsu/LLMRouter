"""
Input Guard — validates and sanitizes user input before processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class InputGuardConfig:
    """Configuration for input validation."""
    max_prompt_length: int = 10_000
    min_prompt_length: int = 1
    max_user_id_length: int = 128


class InputValidationError(Exception):
    """Raised when input fails validation."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class InputGuard:
    """Validates and sanitizes incoming prompts and user data."""

    def __init__(self, config: Optional[InputGuardConfig] = None):
        self.config = config or InputGuardConfig()

    def validate_prompt(self, prompt: str) -> str:
        """
        Validate and sanitize a prompt.

        Returns the sanitized prompt.
        Raises InputValidationError if the prompt is invalid.
        """
        if not prompt or not prompt.strip():
            raise InputValidationError("Prompt cannot be empty.")

        # Strip leading/trailing whitespace
        prompt = prompt.strip()

        # Check length
        if len(prompt) > self.config.max_prompt_length:
            raise InputValidationError(
                f"Prompt exceeds maximum length of {self.config.max_prompt_length} characters "
                f"(got {len(prompt)})."
            )

        # Remove control characters (except newlines, tabs)
        prompt = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt)

        # Validate UTF-8 (Python strings are already valid Unicode,
        # but check for replacement characters indicating bad encoding)
        if '\ufffd' in prompt:
            raise InputValidationError("Prompt contains invalid characters.")

        # Re-check after sanitization
        if not prompt.strip():
            raise InputValidationError(
                "Prompt is empty after sanitization."
            )

        return prompt

    def validate_user_id(self, user_id: str) -> str:
        """
        Validate a user ID.

        Returns the validated user ID.
        Raises InputValidationError if invalid.
        """
        if not user_id or not user_id.strip():
            raise InputValidationError("User ID cannot be empty.")

        user_id = user_id.strip()

        if len(user_id) > self.config.max_user_id_length:
            raise InputValidationError(
                f"User ID exceeds maximum length of {self.config.max_user_id_length}."
            )

        # Only allow alphanumeric, hyphens, underscores, dots
        if not re.match(r'^[a-zA-Z0-9._-]+$', user_id):
            raise InputValidationError(
                "User ID may only contain letters, numbers, dots, hyphens, and underscores."
            )

        return user_id

    def validate(self, user_id: str, prompt: str) -> Tuple[str, str]:
        """
        Validate both user ID and prompt.

        Returns (validated_user_id, sanitized_prompt).
        """
        validated_uid = self.validate_user_id(user_id)
        sanitized_prompt = self.validate_prompt(prompt)
        return validated_uid, sanitized_prompt
