"""
Structured logging with Rich for beautiful console output.
"""
import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

from src.utils.config import get_settings

@lru_cache(maxsize=1)
def get_logger(name: str = "rag-assistant") -> logging.Logger:
    """Configure and return a Rich-formatted logger."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    console = Console(stderr=True)
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=True,
    )

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger