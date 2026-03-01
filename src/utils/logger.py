"""
Premium structured logging utility.

Provides a consistent logger factory across all modules with:
1. Rich terminal formatting (colors, levels, tracebacks).
2. Persistent file logging (logs/vuln_detector.log) with markup stripped.
3. Environment-configurable log level via LOG_LEVEL env var.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

# ── Configuration ────────────────────────────────────────────────────────

_DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "vuln_detector.log"
_CONFIGURED = False

# Regex to strip Rich markup tags from file logs
_MARKUP_RE = re.compile(r"\[/?[a-z_ ]+\]")


class _CleanFormatter(logging.Formatter):
    """Formatter that strips Rich markup tags before writing to file."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return _MARKUP_RE.sub("", original)


def _configure_root(level: str = _DEFAULT_LEVEL) -> None:
    """Configure root logger with Rich (terminal) and File handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # ── Terminal handler (Rich) ──────────────────────────────────────
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        ],
    )

    # ── File handler (clean text, no markup) ─────────────────────────
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(
        _CleanFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logger initialised [dim](level=%s, file=%s)[/dim]",
        level,
        _LOG_FILE,
    )


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a named logger with structured formatting.

    Args:
        name: Logger name — typically ``__name__``.
        level: Optional override for log level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    _configure_root(level or _DEFAULT_LEVEL)
    return logging.getLogger(name)
