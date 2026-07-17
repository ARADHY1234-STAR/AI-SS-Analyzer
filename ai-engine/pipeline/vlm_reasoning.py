"""
Pipeline stage for Vision-Language Model (VLM) reasoning.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from config.settings import settings
from models.vlm_base import VLMResult
from models.vlm_factory import get_vlm_service
from pipeline.comparison import ComparisonSignal

logger = logging.getLogger(__name__)

# Path to the telemetry file produced by the desktop agent
_METRICS_FILE = Path(__file__).resolve().parent.parent / "session_metrics.json"


def _load_session_metrics() -> list[dict]:
    """
    Reads session_metrics.json from the ai-engine root directory.

    Returns the entries sorted chronologically by startTime so they can be
    mapped 1-to-1 against the incoming screenshot batch by index.
    Falls back to an empty list if the file is missing or malformed.
    """
    if not _METRICS_FILE.is_file():
        logger.warning(f"Session metrics file not found at {_METRICS_FILE}. Skipping telemetry injection.")
        return []

    try:
        with open(_METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.warning("session_metrics.json is not a JSON array. Skipping.")
            return []

        # Sort chronologically to guarantee correct index alignment with frames
        data.sort(key=lambda entry: entry.get("startTime", ""))
        logger.info(f"Loaded {len(data)} telemetry entries from session_metrics.json")
        return data

    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read session_metrics.json: {e}")
        return []


def process_batch(
    images: list[bytes],
    ocr_texts: list[str],
    comparison_signals: list[ComparisonSignal],
    model_override: str | None = None,
    persona_id: str | None = None,
    include_keystrokes: bool = False,
) -> VLMResult:
    """
    Passes the chronological batch of images and associated metadata to the configured VLM.
    The VLM determines the current task, progress, and assigns a subjective productivity score.

    Args:
        images: Chronological list of preprocessed image bytes.
        ocr_texts: Chronological list of extracted OCR text strings.
        comparison_signals: Chronological list of structural/semantic comparison metrics.
        model_override: Optional model slug to override the default VLM model for this call.

    Returns:
        A VLMResult object containing the structured analysis from the AI.
    """
    if not images:
        logger.warning("No images provided to VLM reasoning stage.")
        return VLMResult(
            task_description="No input data",
            progressed=False,
            score=0.0,
            confidence=1.0,
            evidence=["Empty batch."],
            activity_summary=[],
            raw_response={"model_used": "None"}
        )

    # Check the global toggle before burning API credits
    if not settings.vlm_enabled:
        logger.info("VLM reasoning is disabled via settings. Skipping.")
        return VLMResult(
            task_description="VLM Disabled",
            progressed=False,
            score=0.0,
            confidence=1.0,
            evidence=["VLM_ENABLED is set to False in configuration."],
            activity_summary=[],
            raw_response={"model_used": "Disabled"}
        )

    # Build the context payload to help ground the VLM's reasoning
    context_data: dict[str, Any] = {
        "batch_size": len(images),
        "frame_data": []
    }

    # ── Load telemetry from session_metrics.json ──
    metrics_entries = _load_session_metrics()

    # Map telemetry entries to frames by chronological index.
    # If fewer metrics than frames exist, extra frames get no telemetry.
    # If fewer frames than metrics, excess metrics are silently ignored.
    timeline_metrics: list[dict] = []
    for i in range(len(images)):
        if i < len(metrics_entries):
            entry = metrics_entries[i]
            timeline_metrics.append({
                "frame_index": i,
                "keyboard_strokes": entry.get("keyboardStrokes", 0) if include_keystrokes else 0,
                "mouse_strokes": entry.get("mouseStrokes", 0) if include_keystrokes else 0,
                "start_time": entry.get("startTime", ""),
                "end_time": entry.get("endTime", ""),
                "user_name": entry.get("userName", "Unknown"),
            })
        else:
            # No telemetry available for this frame — mark explicitly
            timeline_metrics.append({
                "frame_index": i,
                "keyboard_strokes": None,
                "mouse_strokes": None,
                "start_time": None,
                "end_time": None,
                "user_name": None,
            })

    context_data["timeline_metrics"] = timeline_metrics

    # Align the metadata per frame
    # We loop over the images length safely to prevent indexing errors if earlier stages dropped data
    for i in range(len(images)):
        frame_info: dict[str, Any] = {
            "frame_index": i
        }
        
        # Inject truncated OCR text (truncated to save token costs while retaining context)
        if i < len(ocr_texts) and ocr_texts[i]:
            frame_info["extracted_text_snippet"] = ocr_texts[i][:500] 
            
        # Inject heuristic signals (idle/trivial change) so the VLM knows if nothing actually happened
        if i < len(comparison_signals):
            signal = comparison_signals[i]
            frame_info["metrics"] = {
                "is_trivial_change": signal.is_trivial_change,
                "is_idle": signal.is_idle
            }
            
        context_data["frame_data"].append(frame_info)

    try:
        # CRITICAL ARCHITECTURAL COMPLIANCE: 
        # Retrieve the configured VLM service dynamically. 
        # We never hardcode Anthropic, OpenAI, or Gemini here.
        vlm_service = get_vlm_service()
        
        logger.debug(f"Sending batch of {len(images)} images to VLM provider: {settings.vlm_provider}")
        result = vlm_service.analyze_batch(
            images=images,
            context=context_data,
            model_override=model_override,
            persona_id=persona_id,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"VLM reasoning stage failed completely: {str(e)}")
        return VLMResult(
            task_description="Analysis Failed",
            progressed=False,
            score=0.0,
            confidence=0.0,
            evidence=[f"Fatal error in VLM pipeline: {str(e)}"],
            activity_summary=[],
            raw_response={"error": str(e), "model_used": model_override or settings.vlm_model}
        )