"""Reusable timestamped file logging for long-running pipeline processes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Iterator


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


@contextmanager
def process_log_file(
    logs_dir: str | Path | None,
    process: str,
    *,
    level: int = logging.INFO,
) -> Iterator[Path | None]:
    """Capture root-logger output in a timestamped, process-specific file."""
    if logs_dir is None:
        yield None
        return

    safe_process = re.sub(r"[^A-Za-z0-9_.-]+", "_", process).strip("._") or "process"
    started_at = datetime.now().astimezone()
    output_dir = Path(logs_dir) / safe_process / started_at.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%dT%H%M%S_%f%z")
    path = output_dir / f"{safe_process}_{timestamp}.log"

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root_logger.addHandler(handler)
    if previous_level > level:
        root_logger.setLevel(level)
    try:
        root_logger.info("Started %s process; log_file=%s", safe_process, path)
        yield path
    except Exception:
        root_logger.exception("Failed %s process; log_file=%s", safe_process, path)
        raise
    else:
        root_logger.info("Finished %s process; log_file=%s", safe_process, path)
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        root_logger.setLevel(previous_level)
