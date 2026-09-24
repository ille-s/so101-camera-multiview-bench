"""Logging with millisecond timestamps and an optional file mirror.

Why this exists rather than bare ``logging.basicConfig``: under Isaac Lab, the
Kit engine redirects stdout, so ``print()`` and console handlers can silently
go nowhere. The file sink is therefore the primary channel, exactly as in the
thesis pipeline this package was extracted from.

Entry points call :func:`configure_root_logger` once; library modules only ever
call :func:`get_logger`.

Set ``SO101_BENCH_LOG_FILE`` to mirror everything into a file.
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

__version__ = "1.0.0"

_RESET = "\033[0m"
_BOLD = "\033[1m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[0;37m",
    logging.INFO: "\033[0;34m",
    logging.WARNING: "\033[1;33m",
    logging.ERROR: "\033[0;31m",
    logging.CRITICAL: "\033[0;31m" + _BOLD,
}

_FMT = "[%(asctime)s] [%(levelname)s] %(message)s"


class _MsFormatter(logging.Formatter):
    """``HH:MM:SS.mmm`` timestamps, optionally with ANSI level colours."""

    def __init__(self, fmt: str, use_color: bool = False) -> None:
        super().__init__(fmt)
        self._use_color = use_color

    def formatTime(self, record, datefmt=None):  # noqa: N802 - stdlib API
        ct = datetime.datetime.fromtimestamp(record.created)
        return ct.strftime("%H:%M:%S.") + f"{ct.microsecond // 1000:03d}"

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_color:
            return super().format(record)
        color = _LEVEL_COLORS.get(record.levelno, "")
        original = record.levelname
        record.levelname = f"{color}{record.levelname}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def _attach_handlers(logger: logging.Logger) -> None:
    """Attach a stderr handler and, if configured, a file handler. Idempotent."""
    if logger.handlers:
        return

    logger.setLevel(logging.DEBUG)

    stream = logging.StreamHandler()
    stream.setFormatter(_MsFormatter(_FMT, use_color=sys.stderr.isatty()))
    logger.addHandler(stream)

    # Imported lazily so a test can rebuild Settings before the first log call.
    from so101_mvbench.settings import SETTINGS

    if SETTINGS.log_file is not None:
        SETTINGS.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(SETTINGS.log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(_MsFormatter(_FMT, use_color=False))
        logger.addHandler(file_handler)


def get_logger(name: str, channel: str | None = None) -> logging.Logger:
    """Return a configured logger for *name*.

    Args:
        name: Logger name; pass ``__name__`` from a module.
        channel: Accepted and ignored. Kept so the ~30 call sites carried over
            from the thesis pipeline need no edit; the upstream symbol table it
            selected was workspace-specific and was not carried over.
    """
    del channel
    logger = logging.getLogger(name)
    _attach_handlers(logger)
    logger.propagate = False
    return logger


def configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger so third-party loggers inherit the handlers."""
    root = logging.getLogger()
    _attach_handlers(root)
    root.setLevel(level)


if __name__ == "__main__":  # ponytail: inline self-check, `python -m so101_mvbench.logging`
    log = get_logger("selfcheck")
    assert len(log.handlers) >= 1
    before = len(log.handlers)
    get_logger("selfcheck")
    assert len(log.handlers) == before, "handler attachment must be idempotent"
    rec = logging.LogRecord("selfcheck", logging.INFO, __file__, 1, "hi", None, None)
    out = _MsFormatter(_FMT).format(rec)
    assert out.count(":") >= 2 and "." in out.split("]")[0], out
    log.info("logging self-check passed")
