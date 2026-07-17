"""
Core orchestration module that coordinates all pipeline stages.
"""

import logging

from pipeline import (
    preprocess,
    ocr,
    embeddings,
    comparison,
    vlm_reasoning,
    confidence,
    aggregate
)
from pipeline.progress_detectors import (
    code,
    document,
    spreadsheet,
    presentation,
    design,
    ticket
)

logger = logging.getLogger(__name__)


def analyze_session(
    raw_images: list[bytes], 
    model_override: str | None = None,
    persona_id: str | None = None,
    include_keystrokes: bool = False
) -> aggregate.AggregatedResult:
    """
    Executes the complete end-to-end analysis pipeline on a batch of screenshots.

    This function is completely synchronous and maps to the /analyze endpoint.
    It passes the data through every defined pipeline stage in sequential order.

    Args:
        raw_images: A chronological list of raw image byte payloads from the client.
        model_override: Optional model slug to override the default VLM model.

    Returns:
        An AggregatedResult containing the final blended productivity score, 
        confidence, AI judgment, and heuristic breakdowns.
        
    Raises:
        ValueError: If the input list is entirely empty or all images are invalid.
    """
    logger.info(f"Starting session analysis for batch of {len(raw_images)} raw images.")

    if not raw_images:
        raise ValueError("Cannot analyze an empty batch of images.")

    # 1. Preprocessing (Resize, standard format, deduplication)
    processed_images = preprocess.preprocess_batch(raw_images)
    
    if not processed_images:
        raise ValueError("All provided images were invalid, unreadable, or completely duplicated.")
        
    logger.info(f"Preprocessing complete. Retained {len(processed_images)} unique frames.")

    # 2. Extract Text (OCR)
    ocr_texts = ocr.process_batch(processed_images)

    # 3. Generate Semantic Embeddings (CLIP & MiniLM)
    image_embeddings, text_embeddings = embeddings.process_batch(processed_images, ocr_texts)

    # 4. Compute Frame-to-Frame Mathematical Comparisons
    comparison_signals = comparison.process_batch(
        images=processed_images,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings
    )

    # 5. Run Domain-Specific Detectors
    # We pass the extracted OCR text to our lightweight heuristic detectors
    # to evaluate domain-specific activity without hitting an external API.
    domain_signals = {
        "code": code.detect(ocr_texts),
        "document": document.detect(ocr_texts),
        "spreadsheet": spreadsheet.detect(ocr_texts),
        "presentation": presentation.detect(ocr_texts),
        "design": design.detect(ocr_texts),
        "ticket": ticket.detect(ocr_texts),
    }

    # 6. Vision-Language Model (VLM) Reasoning
    # Evaluates the contextual flow of the visual frames and objective metadata
    vlm_result = vlm_reasoning.process_batch(
        images=processed_images,
        ocr_texts=ocr_texts,
        comparison_signals=comparison_signals,
        model_override=model_override,
        persona_id=persona_id,
        include_keystrokes=include_keystrokes
    )

    # 7. Calculate Confidence
    # Adjust the VLM's subjective certainty against objective data quality markers
    overall_conf = confidence.calculate_overall_confidence(
        vlm_result=vlm_result,
        ocr_texts=ocr_texts,
        image_count=len(processed_images)
    )

    # 8. Final Aggregation
    # Blend all signals using the weights defined in settings.py
    final_result = aggregate.process_batch(
        vlm_result=vlm_result,
        comparison_signals=comparison_signals,
        domain_signals=domain_signals,
        overall_confidence=overall_conf
    )

    logger.info("Session analysis completed successfully.")
    return final_result