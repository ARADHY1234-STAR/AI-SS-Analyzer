"""
Optical Character Recognition (OCR) client wrapper.
"""

import io
import logging
from PIL import Image, UnidentifiedImageError
import pytesseract

from config.settings import settings

# Configure the global tesseract command path from settings
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

logger = logging.getLogger(__name__)


class OCRClient:
    """
    Client for extracting text from images using Tesseract OCR.
    """

    def __init__(self) -> None:
        """
        Initializes the OCR client. 
        Currently supports Tesseract as defined in settings.ocr_engine.
        """
        self.engine = settings.ocr_engine

    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extracts raw text from an image byte array.

        Args:
            image_bytes: Raw image data (e.g., JPEG, PNG).

        Returns:
            Extracted text as a string. Returns an empty string if OCR fails 
            or if the image contains no readable text.
        """
        if not image_bytes:
            return ""

        if self.engine.lower() != "tesseract":
            logger.warning(f"Unsupported OCR engine '{self.engine}'. Falling back to Tesseract.")

        try:
            # Load the image from bytes using Pillow
            image = Image.open(io.BytesIO(image_bytes))
            
            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(image)
            
            return extracted_text.strip()
            
        except UnidentifiedImageError:
            logger.error("Failed to extract text: Invalid or corrupted image bytes.")
            return ""
        except pytesseract.TesseractNotFoundError:
            logger.error(
                f"Tesseract executable not found at '{settings.tesseract_cmd}'. "
                "Ensure Tesseract is installed and TESSERACT_CMD is configured correctly."
            )
            return ""
        except Exception as e:
            logger.error(f"Unexpected error during OCR processing: {str(e)}")
            return ""