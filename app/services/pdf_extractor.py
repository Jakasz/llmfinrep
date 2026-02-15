"""PDF text extraction using PyMuPDF with OCR fallback.

For each page:
- Try native text extraction via PyMuPDF first.
- If extracted text is too short (< 50 chars), treat page as scanned
  and run OCR on a rendered page image.
"""

import logging
from typing import Callable

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Мінімальна довжина тексту сторінки для вважання її "нативною"
MIN_TEXT_LENGTH = 50


def extract_text_from_pdf(
    file_bytes: bytes,
    ocr_func: Callable[[bytes], str],
) -> str:
    """Extract text from a PDF file.

    Args:
        file_bytes: Raw PDF file content.
        ocr_func: Function that accepts image bytes and returns OCR text.

    Returns:
        Combined text from all pages.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    logger.info("PDF opened: %d pages", total_pages)

    full_text = []
    ocr_pages = 0

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if len(text.strip()) < MIN_TEXT_LENGTH:
            # Сторінка ймовірно відсканована — рендеримо як зображення та розпізнаємо
            logger.debug(
                "Page %d: native text too short (%d chars), falling back to OCR",
                page_num + 1,
                len(text.strip()),
            )
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            text = ocr_func(img_bytes)
            ocr_pages += 1

        full_text.append(f"--- Page {page_num + 1} ---\n{text}")

    doc.close()

    if ocr_pages > 0:
        logger.info("OCR was used for %d out of %d pages", ocr_pages, total_pages)

    return "\n\n".join(full_text)
