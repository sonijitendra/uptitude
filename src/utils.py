"""
Shared utilities for the Legal Contract Analysis Pipeline.

Provides:
    - Structured logging configuration
    - Token counting helpers
    - Text chunking utilities
"""

import logging
import sys
from typing import List

import tiktoken

from config.settings import get_settings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str | None = None) -> logging.Logger:
    """Configure and return a logger with structured formatting.

    Sets up a console handler with timestamp, level, module name,
    and message. Idempotent — safe to call multiple times.

    Args:
        name: Logger name. Defaults to root logger if None.

    Returns:
        Configured logging.Logger instance.
    """
    settings = get_settings()
    logger = logging.getLogger(name or "contract_pipeline")

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# Token Counting
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count the number of tokens in a text string.

    Uses tiktoken for accurate OpenAI model token counting.

    Args:
        text: Input text to tokenize.
        model: Model name for tokenizer selection.

    Returns:
        Number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """Split text into overlapping chunks by character count.

    Uses paragraph boundaries when possible to avoid splitting
    mid-sentence. Falls back to character-level splitting if
    paragraphs are too large.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
            Defaults to settings value.
        chunk_overlap: Overlap between adjacent chunks.
            Defaults to settings value.

    Returns:
        List of text chunks with overlap.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph boundary
        if end < len(text):
            # Look for paragraph break within last 20% of chunk
            search_start = max(start, end - chunk_size // 5)
            para_break = text.rfind("\n\n", search_start, end)

            if para_break != -1:
                end = para_break + 2  # Include the double newline
            else:
                # Fall back to sentence boundary
                sentence_break = text.rfind(". ", search_start, end)
                if sentence_break != -1:
                    end = sentence_break + 2

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = end - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text to a maximum length with ellipsis.

    Args:
        text: Input text.
        max_chars: Maximum character count.

    Returns:
        Truncated text with '...' appended if shortened.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
