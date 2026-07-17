"""
Image preprocessing and deduplication module.
"""

import io
import logging
from typing import Optional

import imagehash
from PIL import Image, UnidentifiedImageError

from config.settings import settings

logger = logging.getLogger(__name__)


def process_image(image_bytes: bytes) -> Optional[bytes]:
    """
    Validates, resizes, and standardizes a single image.
    Converts to RGB JPEG and resizes down to the maximum dimension specified in settings.

    Args:
        image_bytes: The raw image payload.

    Returns:
        The processed image bytes, or None if the image is invalid or unreadable.
    """
    if not image_bytes:
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Drop alpha channels and standardize to RGB for consistent downstream processing
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

            # thumbnail() securely preserves aspect ratio while restricting max dimensions
            max_dim = settings.resize_max_dimension
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            # Save as standard JPEG to balance visual fidelity and payload size for VLM APIs
            img.save(output, format="JPEG", quality=85)
            
            # Fast check against the configured max size (in bytes)
            processed_bytes = output.getvalue()
            max_bytes = settings.max_image_size_mb * 1024 * 1024
            
            if len(processed_bytes) > max_bytes:
                logger.warning(f"Image exceeds {settings.max_image_size_mb}MB even after resizing.")
                # We could compress further, but returning None safely drops it from the batch
                return None
                
            return processed_bytes

    except UnidentifiedImageError:
        logger.error("Preprocessing failed: Invalid or corrupted image bytes.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during image preprocessing: {str(e)}")
        return None


def deduplicate_batch(image_batch: list[bytes]) -> list[bytes]:
    """
    Filters a chronological sequence of images to remove redundant frames.
    Uses perceptual hashing (pHash) to detect structurally identical or highly similar images.

    Args:
        image_batch: A list of processed image bytes.

    Returns:
        A deduplicated list of image bytes preserving the original chronological order.
    """
    if not image_batch:
        return []

    unique_images: list[bytes] = []
    seen_hashes: list[imagehash.ImageHash] = []
    threshold = settings.duplicate_phash_threshold

    for img_bytes in image_batch:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # pHash evaluates visual structure, ignoring minor compression artifacts
                current_hash = imagehash.phash(img)
                
                is_duplicate = False
                for seen_hash in seen_hashes:
                    # Subtracting imagehash objects computes the Hamming distance
                    if current_hash - seen_hash <= threshold:
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    seen_hashes.append(current_hash)
                    unique_images.append(img_bytes)

        except Exception as e:
            logger.warning(f"Failed to hash image, retaining it by default: {str(e)}")
            unique_images.append(img_bytes)

    return unique_images


def preprocess_batch(raw_images: list[bytes]) -> list[bytes]:
    """
    The main entry point for the preprocessing pipeline stage.
    Standardizes dimensions/formats and strips out identical frames.

    Args:
        raw_images: The list of raw bytes directly from the API request.

    Returns:
        A list of clean, properly sized, deduplicated JPEG bytes.
    """
    processed = []
    for raw in raw_images:
        proc = process_image(raw)
        if proc is not None:
            processed.append(proc)
            
    return deduplicate_batch(processed)