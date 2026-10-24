"""Structured logging setup for console and JSONL audit trail."""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


class JSONLinesFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord into a JSON-serialized string."""
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }

        # Include structured extra attributes if passed
        if hasattr(record, "extra_payload") and isinstance(record.extra_payload, dict):
            log_payload.update(record.extra_payload)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, default=str)


def setup_logging(
    log_dir: str | Path = "logs",
    log_level: int = logging.INFO,
    run_timestamp: Optional[str] = None,
) -> Path:
    """Configure root logger to emit to standard output and a timestamped JSONL file.

    Args:
        log_dir: Directory where JSONL logs will be written.
        log_level: Logging severity threshold.
        run_timestamp: Optional explicit timestamp string for log file naming.

    Returns:
        Path: Absolute path to the created JSONL log file.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    if run_timestamp is None:
        run_timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_file = directory / f"run_{run_timestamp}.jsonl"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Console Handler (Human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File Handler (JSONL)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONLinesFormatter())
    root_logger.addHandler(file_handler)

    root_logger.info("Structured logging initialized. JSONL output: %s", log_file.resolve())
    return log_file


def log_event(
    logger: logging.Logger,
    level: int,
    event_type: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a structured event with explicit metadata into JSONL.

    Args:
        logger: Target logger instance.
        level: Logging level (e.g. logging.INFO, logging.WARNING).
        event_type: Category identifier (e.g. 'DATA_VALIDATION', 'MODEL_FIT_ERROR').
        message: Descriptive log text.
        **kwargs: Additional key-value metadata to record.
    """
    payload = {"event_type": event_type, **kwargs}
    logger.log(level, message, extra={"extra_payload": payload})
