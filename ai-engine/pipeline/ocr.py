"""
Pipeline stage for Optical Character Recognition (OCR).
"""

import logging

from models.ocr_client import OCRClient

logger = logging.getLogger(__name__)

# Instantiate the OCR client once at the module level to avoid setup overhead per batch
_ocr_client = OCRClient()


def process_batch(images: list[bytes]) -> list[str]:
    """
    Extracts raw text from a chronological batch of preprocessed images.

    Args:
        images: A list of preprocessed image bytes (e.g., standard JPEGs).

    Returns:
        A list of extracted text strings, preserving the exact input order. 
        If text extraction fails or no text is found for a specific frame, 
        an empty string is returned for that index.
    """
    if not images:
        return []

    extracted_texts: list[str] = []

    for idx, img_bytes in enumerate(images):
        logger.debug(f"Running OCR on image {idx + 1}/{len(images)}...")
        try:
            text = _ocr_client.extract_text(img_bytes)
            extracted_texts.append(text)
        except Exception as e:
            logger.error(f"Unhandled error during OCR on frame {idx}: {str(e)}")
            # Always append an empty string on failure to maintain list length parity
            extracted_texts.append("")

    return extracted_texts