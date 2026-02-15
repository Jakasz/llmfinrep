"""Text cleaning, truncation, and token estimation utilities."""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove excessive whitespace and control characters, keep Ukrainian chars."""
    # Замінити послідовності пробілів/табуляцій на один пробіл
    text = re.sub(r"[^\S\n]+", " ", text)
    # Замінити 3+ послідовних порожніх рядків на 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Rough token estimate for Ukrainian/Cyrillic text.

    Ukrainian text typically yields ~1 token per 3.5 characters
    due to multi-byte encoding and subword tokenization.
    """
    if not text:
        return 0
    return int(len(text) / 3.5)


def truncate_text(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to approximately max_tokens.

    Returns:
        Tuple of (possibly truncated text, was_truncated flag).
    """
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text, False

    # Приблизна кількість символів для бажаного ліміту токенів
    target_chars = int(max_tokens * 3.5)
    truncated = text[:target_chars]

    # Обрізати по останньому повному рядку
    last_newline = truncated.rfind("\n")
    if last_newline > target_chars * 0.8:
        truncated = truncated[:last_newline]

    warning = (
        "\n\n[УВАГА: Текст було скорочено через перевищення ліміту токенів. "
        f"Оригінальний розмір: ~{current_tokens} токенів, "
        f"ліміт: {max_tokens} токенів.]"
    )
    logger.warning(
        "Text truncated from ~%d to ~%d estimated tokens",
        current_tokens,
        estimate_tokens(truncated),
    )
    return truncated + warning, True


def combine_extracted_texts(file_texts: dict[str, str]) -> str:
    """Combine extracted texts from multiple files with headers.

    Args:
        file_texts: Mapping of filename to extracted text.
    """
    parts = []
    for filename, text in file_texts.items():
        parts.append(f"=== FILE: {filename} ===\n{text}")
    return "\n\n".join(parts)
