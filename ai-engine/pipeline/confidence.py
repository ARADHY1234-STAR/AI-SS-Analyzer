"""
Pipeline stage for calculating the overall confidence score of the session analysis.
"""

import logging

from models.vlm_base import VLMResult

logger = logging.getLogger(__name__)


def calculate_overall_confidence(
    vlm_result: VLMResult,
    ocr_texts: list[str],
    image_count: int
) -> float:
    """
    Computes a holistic confidence score for the batch analysis.
    Blends the subjective confidence provided by the VLM with objective data quality metrics.

    Args:
        vlm_result: The structured output from the VLM provider.
        ocr_texts: The list of extracted OCR strings.
        image_count: The total number of images processed in the batch.

    Returns:
        A float between 0.0 and 1.0 representing the system's overall confidence 
        in its productivity assessment.
    """
    if image_count == 0:
        logger.warning("Confidence calculation aborted: 0 images processed.")
        return 0.0

    # The VLM provider dictates the baseline confidence based on its visual reasoning
    base_confidence = vlm_result.confidence

    # Data Quality Heuristic:
    # If the images are so blurry or abstract that OCR completely failed across the board,
    # the system's holistic visibility is slightly degraded. We apply a minor penalty.
    empty_ocr_count = sum(1 for text in ocr_texts if not text.strip())
    ocr_failure_rate = empty_ocr_count / image_count
    
    # Maximum penalty of 10% (0.10) if OCR is entirely blind
    data_quality_penalty = ocr_failure_rate * 0.10

    # Calculate final blended score, clamping securely between 0.0 and 1.0
    final_confidence = max(0.0, min(1.0, base_confidence - data_quality_penalty))

    logger.debug(
        f"Calculated overall confidence: {final_confidence:.4f} "
        f"(Base VLM: {base_confidence}, OCR Penalty: -{data_quality_penalty:.4f})"
    )

    return round(final_confidence, 4)