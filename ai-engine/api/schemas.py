"""
Pydantic schemas for FastAPI request and response validation.
"""

from typing import Any
from pydantic import BaseModel, Field


class VLMResponseSchema(BaseModel):
    """
    Public schema for the subjective Vision-Language Model analysis.
    Mirrors the internal VLMResult dataclass, but drops the 'raw_response' 
    field to avoid leaking provider-specific API structures to the client.
    """
    task_description: str = Field(
        ..., 
        description="A short description of the user's detected task."
    )
    progressed: bool = Field(
        ..., 
        description="Whether meaningful progress was made during the session."
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="The VLM's subjective productivity score (0.0 to 100.0)."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="The VLM's confidence in its own judgment."
    )
    evidence: list[str] = Field(
        ..., 
        description="List of specific observations supporting the VLM's judgment."
    )
    activity_summary: list[str] = Field(
        default=[],
        description="2-3 concise bullet points summarizing what the AI infers the user was doing between screenshots."
    )


class DomainSignalSchema(BaseModel):
    """
    Standardized schema for individual domain detector outputs.
    """
    is_active: bool = Field(..., description="True if this specific type of activity was detected.")
    progress_score: float = Field(..., ge=0.0, le=1.0, description="Heuristic progress score for this domain.")


class AnalyzeSessionResponse(BaseModel):
    """
    The primary response model for the POST /api/v1/sessions/analyze endpoint.
    Maps directly to the internal pipeline.aggregate.AggregatedResult.
    """
    final_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="The final, weighted productivity score for the session."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="The engine's holistic confidence in this analysis, penalized by poor data quality."
    )
    vlm_summary: VLMResponseSchema = Field(
        ..., 
        description="The reasoning and evidence provided by the AI provider."
    )
    metrics_breakdown: dict[str, float] = Field(
        ..., 
        description="Detailed numerical breakdown showing how much each pipeline stage contributed to the final score."
    )
    domain_activity: dict[str, DomainSignalSchema | dict[str, Any]] = Field(
        ..., 
        description="Results from the domain-specific heuristic detectors (code, document, spreadsheet, etc.)."
    )
    model_used: str | None = Field(
        default=None,
        description="The VLM model slug used during this analysis."
    )