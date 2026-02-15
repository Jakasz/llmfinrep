"""File type dispatcher: detects file type by extension and calls the correct extractor."""

import logging

from app.services.image_extractor import extract_text_from_image, get_ocr_func
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.excel_extractor import extract_text_from_excel
from app.services.docx_extractor import extract_text_from_docx
from app.utils.text_utils import clean_text

logger = logging.getLogger(__name__)

# Дозволені розширення файлів
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".docx"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_allowed_extension(filename: str) -> bool:
    """Check if file extension is supported."""
    ext = _get_extension(filename)
    return ext in ALLOWED_EXTENSIONS


def _get_extension(filename: str) -> str:
    """Extract lowercase extension from filename."""
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        return ""
    return filename[dot_idx:].lower()


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from a file based on its extension.

    Args:
        filename: Original filename (used to detect type).
        file_bytes: Raw file content.

    Returns:
        Extracted and cleaned text.

    Raises:
        ValueError: If file extension is not supported.
    """
    ext = _get_extension(filename)
    logger.info("Extracting text from '%s' (extension: %s, size: %d bytes)", filename, ext, len(file_bytes))

    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes, get_ocr_func())
    elif ext in IMAGE_EXTENSIONS:
        raw_text = extract_text_from_image(file_bytes)
    elif ext == ".xlsx":
        raw_text = extract_text_from_excel(file_bytes)
    elif ext == ".docx":
        raw_text = extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    cleaned = clean_text(raw_text)
    logger.info("Extraction complete for '%s': %d characters", filename, len(cleaned))
    return cleaned
