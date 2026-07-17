"""
Pipeline stage for aggregating all heuristic and AI signals into a final productivity score.
"""

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import settings
from models.vlm_base import VLMResult
from pipeline.comparison import ComparisonSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregatedResult:
    """
    The final structured output of the entire pipeline analysis.
    """
    final_score: float
    confidence: float
    vlm_summary: VLMResult
    metrics_breakdown: dict[str, float]
    domain_activity: dict[str, Any]
    model_used: str | None = None


def process_batch(
    vlm_result: VLMResult,
    comparison_signals: list[ComparisonSignal],
    domain_signals: dict[str, dict[str, float | bool]],
    overall_confidence: float
) -> AggregatedResult:
    """
    Blends the domain-specific metrics, mathematical image comparison deltas, 
    and the subjective VLM judgment into a single, weighted productivity score.

    Args:
        vlm_result: The structured analysis from the AI provider.
        comparison_signals: The mathematical frame-to-frame distance metrics.
        domain_signals: The combined outputs from all domain-specific detectors.
        overall_confidence: The holistic confidence score calculated previously.

    Returns:
        An AggregatedResult containing the final score and the detailed breakdown.
    """
    # 1. Calculate the Domain Score & Identify Active Domain
    # We take the maximum progress score across all active domains and store which domain won.
    domain_score = 0.0
    active_domain_name = ""
    
    if domain_signals:
        # Find the domain name with the highest progress score
        active_domain_name = max(
            domain_signals.keys(), 
            key=lambda k: float(domain_signals[k].get("progress_score", 0.0)), 
            default=""
        )
        domain_score = float(domain_signals.get(active_domain_name, {}).get("progress_score", 0.0))

    # 2. Calculate average Structural and Semantic Text Deltas
    struct_diffs: list[float] = []
    text_diffs: list[float] = []
    
    transitions = comparison_signals[1:]
    
    if not transitions:
        avg_struct_diff = 0.0
        avg_text_diff = 0.0
    else:
        for sig in transitions:
            if sig.is_idle:
                struct_diffs.append(0.0)
                text_diffs.append(0.0)
            else:
                struct_diffs.append(max(0.0, 1.0 - sig.ssim_score))
                text_diffs.append(max(0.0, 1.0 - sig.text_similarity))
                
        avg_struct_diff = sum(struct_diffs) / len(struct_diffs)
        avg_text_diff = sum(text_diffs) / len(text_diffs)

    # =========================================================================
    # 2b. HEURISTIC NORMALIZATION & NOISE FLOOR
    # Scale realistic "large changes" to a perfect 1.0 signal, and cancel micro-noise.
    # =========================================================================
    if avg_struct_diff < 0.01:
        avg_struct_diff = 0.0
    else:
        avg_struct_diff = min(1.0, avg_struct_diff / 0.30)
        
    if avg_text_diff < 0.01:
        avg_text_diff = 0.0
    else:
        avg_text_diff = min(1.0, avg_text_diff / 0.40)

    # =========================================================================
    # 3. DYNAMIC WEIGHT CONFIGURATION
    # If the user is working in visually static or low-text-turnover domains 
    # like Spreadsheets or Presentations, shift weight away from Semantic Text Diff 
    # and give greater authority to the VLM's contextual judgment.
    # =========================================================================
    weight_struct = settings.weight_structural_diff
    weight_text = settings.weight_semantic_text_diff
    weight_domain = settings.weight_domain_signal
    weight_vlm = settings.weight_vlm_judgment

    if active_domain_name.lower() in ["spreadsheet", "excel"]:
        logger.debug("Applying Spreadsheet weight overrides (low text diff, high VLM authority).")
        weight_text = 0.10  # Reduced from default (e.g., 0.35 -> 0.10)
        weight_vlm = 0.40   # Boosted from default (e.g., 0.15 -> 0.40)
    elif active_domain_name.lower() in ["presentation", "powerpoint", "design"]:
        logger.debug("Applying Presentation/Design weight overrides.")
        weight_text = 0.15
        weight_struct = 0.35

    weighted_structural = avg_struct_diff * weight_struct
    weighted_text = avg_text_diff * weight_text
    weighted_domain = domain_score * weight_domain
    weighted_vlm = (vlm_result.score / 100.0) * weight_vlm

    raw_final_score = weighted_structural + weighted_text + weighted_domain + weighted_vlm
    
    # =========================================================================
    # 4. GATEKEEPER / SHORT-CIRCUIT OVERRIDE
    # If the VLM determines the user was idle, we refuse to let passive domain 
    # signals inflate the score. Apply a harsh penalty multiplier.
    # =========================================================================
    idle_override_applied = False

    vlm_idle_detected = vlm_result.raw_response.get("idle_detected", False)
    vlm_no_progress = not vlm_result.progressed
    vlm_score_very_low = vlm_result.score <= 20.0

    if vlm_idle_detected or (vlm_no_progress and vlm_score_very_low):
        idle_override_applied = True
        # Cap the max idle score to strictly less than 15% (e.g. 10%)
        raw_final_score = min(raw_final_score * 0.25, 0.10)
        logger.warning(
            f"IDLE GATEKEEPER TRIGGERED — VLM idle_detected={vlm_idle_detected}, "
            f"progressed={vlm_result.progressed}, vlm_score={vlm_result.score:.2f}. "
            f"Score overridden to {raw_final_score:.4f}"
        )

    # =========================================================================
    # 5. CEILING BOOST & TRUE ZERO FLOOR
    # Ensure highly productive sessions can hit 1.0, and completely idle sessions hit 0.0.
    # =========================================================================
    if raw_final_score < 0.05:
        raw_final_score = 0.0
    # Ceiling boost temporarily disabled to show exact scores
    # elif raw_final_score > 0.85:
    #     raw_final_score = 1.0

    # Clamp the final result strictly between 0.0 and 1.0
    final_score = max(0.0, min(1.0, raw_final_score))

    logger.info(f"Batch aggregation complete. Final Score: {final_score:.4f} (Conf: {overall_confidence:.2f})")

    return AggregatedResult(
        final_score=round(final_score, 4),
        confidence=overall_confidence,
        vlm_summary=vlm_result,
        metrics_breakdown={
            "structural_diff_contribution": round(weighted_structural, 4),
            "semantic_text_diff_contribution": round(weighted_text, 4),
            "domain_signal_contribution": round(weighted_domain, 4),
            "vlm_judgment_contribution": round(weighted_vlm, 4),
            "raw_vlm_score": vlm_result.score,
            "raw_domain_score": round(domain_score, 4),
            "idle_override_applied": 1.0 if idle_override_applied else 0.0,
            "dynamic_text_weight_used": weight_text,
            "dynamic_vlm_weight_used": weight_vlm
        },
        domain_activity=domain_signals,
        model_used=vlm_result.raw_response.get("model_used")
    )