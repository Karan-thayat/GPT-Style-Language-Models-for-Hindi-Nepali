"""Logging helpers shared by every Phase 1 script.

Every pipeline stage logs to stdout and to a per-run file under the language's
`logs/` directory so that a run can be audited after the fact.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: Path | None = None, name: str = "lma", level: int = logging.INFO) -> logging.Logger:
    """Configure a logger that writes to stdout and, optionally, a timestamped file.

    Args:
        log_dir: Directory to place a `<name>_<timestamp>.log` file in. Created if missing.
        name: Logger name, also used as the log filename prefix.
        level: Logging threshold.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_handler = logging.FileHandler(log_dir / f"{name}_{stamp}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def write_json(path: Path, payload: dict) -> None:
    """Write `payload` as indented UTF-8 JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file written by `write_json`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
