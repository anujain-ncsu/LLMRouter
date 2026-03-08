"""
Audit Logger — structured JSON logging for all requests.
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    user_id: str
    model_id: str
    prompt_length: int
    model_tokens: int
    system_tokens: float
    latency_ms: float
    status: str  # "success" | "error" | "rate_limited" | "circuit_open"
    error_detail: Optional[str] = None
    auto_selected: bool = False


class AuditLogger:
    """
    Structured JSON audit logger.

    Logs every request to stdout as a JSON line for easy parsing
    and pipeline integration.
    """

    def __init__(self, logger_name: str = "llmrouter.audit"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log(self, entry: AuditEntry) -> None:
        """Log an audit entry as a JSON line."""
        self.logger.info(json.dumps(asdict(entry), default=str))

    def log_request(
        self,
        user_id: str,
        model_id: str,
        prompt_length: int,
        model_tokens: int = 0,
        system_tokens: float = 0.0,
        latency_ms: float = 0.0,
        status: str = "success",
        error_detail: Optional[str] = None,
        auto_selected: bool = False,
    ) -> None:
        """Convenience method to log a request."""
        from datetime import datetime, timezone
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            model_id=model_id,
            prompt_length=prompt_length,
            model_tokens=model_tokens,
            system_tokens=system_tokens,
            latency_ms=latency_ms,
            status=status,
            error_detail=error_detail,
            auto_selected=auto_selected,
        )
        self.log(entry)
