"""Image OCR text extraction using PaddleOCR (PP-OCRv5).

Supports Ukrainian/Cyrillic text out of the box.
The OCR engine is initialized once and reused across requests.
"""

import io
import logging
from typing import Callable

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from app.config import AppConfig

logger = logging.getLogger(__name__)

_ocr_engine: PaddleOCR | None = None


def init_ocr_engine(config: AppConfig) -> None:
    """Initialize PaddleOCR engine (call once at startup)."""
    global _ocr_engine
    # PaddleOCR lang parameter: перша мова зі списку конфігурації
    lang = config.ocr_languages[0] if config.ocr_languages else "uk"
    device = "gpu" if config.ocr_use_gpu else "cpu"
    logger.info("Initializing PaddleOCR engine (lang=%s, device=%s)", lang, device)
    _ocr_engine = PaddleOCR(
        use_textline_orientation=True,
        lang=lang,
        device=device,
    )
    logger.info("PaddleOCR engine initialized successfully")


def get_ocr_engine() -> PaddleOCR:
    """Return the initialized OCR engine."""
    if _ocr_engine is None:
        raise RuntimeError("OCR engine not initialized. Call init_ocr_engine() first.")
    return _ocr_engine


def ocr_from_bytes(image_bytes: bytes) -> str:
    """Extract text from image bytes using PaddleOCR.

    Args:
        image_bytes: Raw image data (PNG, JPG, etc.).

    Returns:
        Extracted text with lines joined by newlines.
    """
    engine = get_ocr_engine()
    img = np.array(Image.open(io.BytesIO(image_bytes)))

    # PaddleOCR 3.4+: predict() returns list of result objects
    # Each result has "rec_texts" (list of str) and "rec_scores" (list of float)
    results = engine.predict(img)

    lines = []
    if results:
        for result in results:
            rec_texts = result.get("rec_texts", [])
            lines.extend(rec_texts)

    return "\n".join(lines)


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from an image file.

    Args:
        file_bytes: Raw image file content.

    Returns:
        Extracted text.
    """
    logger.info("Starting OCR on image (%d bytes)", len(file_bytes))
    text = ocr_from_bytes(file_bytes)
    logger.info("OCR completed, extracted %d characters", len(text))
    return text


def get_ocr_func() -> Callable[[bytes], str]:
    """Return the OCR function for use by other extractors (e.g. PDF fallback)."""
    return ocr_from_bytes
