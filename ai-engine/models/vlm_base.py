"""
Abstract base definitions for Vision-Language Model services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VLMResult:
    """
    Structured output from a VLM provider after analyzing a batch of screenshots.
    """
    task_description: str
    progressed: bool
    score: float
    confidence: float
    evidence: list[str]
    activity_summary: list[str]
    raw_response: dict[str, Any]


class VLMService(ABC):
    """
    Abstract base class defining the contract for all VLM providers.
    """

    @abstractmethod
    def analyze_batch(
        self,
        images: list[bytes],
        context: dict[str, Any],
        model_override: str | None = None,
        persona_id: str | None = None
    ) -> VLMResult:
        """
        Analyze a sequence of chronological screenshots to evaluate work progress.

        Args:
            images: A list of raw image bytes (e.g., JPEG or PNG) in chronological order.
            context: Additional metadata (e.g., OCR text, timestamps) to guide the VLM.
            model_override: Optional OpenRouter model slug to override the default model.
            persona_id: Optional string identifying the active persona role for benchmarking.

        Returns:
            VLMResult containing the parsed judgment, score, and evidence.
        """
        raise NotImplementedError("Subclasses must implement analyze_batch")