"""
Configuration and environment variables module using Pydantic Settings.
"""

import os
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    # --- General ---
    app_name: str = "AI Engine"
    environment: str = Field(default="local", alias="APP_ENV")
    debug: bool = True

    # --- Storage ---
    upload_dir: str = "./data/uploads"
    max_screenshots_per_batch: int = 10
    max_image_size_mb: int = 15

    # --- Preprocessing ---
    resize_max_dimension: int = 1568
    duplicate_phash_threshold: int = 4

    # --- OCR ---
    ocr_engine: str = "tesseract"
    tesseract_cmd: str = "tesseract"

    # --- Embeddings ---
    clip_model_name: str = "clip-ViT-B-32"
    text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- VLM Reasoning Stage ---
    vlm_enabled: bool = True
    vlm_provider: str = "groq"
    vlm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    vlm_max_tokens: int = 2000

    # API Keys

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    qwen_api_key: str | None = Field(default=None, alias="QWEN_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    # Provider Specific Overrides
    qwen_api_base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # --- Comparison Heuristic Thresholds ---
    ssim_trivial_change_threshold: float = 0.97
    embedding_similarity_idle_threshold: float = 0.995

    # --- Aggregation Scoring Weights ---
    weight_structural_diff: float = 0.25
    weight_semantic_text_diff: float = 0.35
    weight_domain_signal: float = 0.25
    weight_vlm_judgment: float = 0.15

    # Pydantic V2 config dict to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("vlm_provider", mode="before")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Ensures the selected provider is explicitly supported by the factory."""
        allowed = ["anthropic", "openai", "gemini", "qwen", "groq", "openrouter"]
        if not v or v.lower() not in allowed:
            raise ValueError(f"Provider must be one of {allowed}")
        return v.lower()

    @model_validator(mode="after")
    def validate_weights(self) -> "Settings":
        """Ensures the mathematical aggregation weights sum properly to 1.0."""
        total_weight = (
            self.weight_structural_diff +
            self.weight_semantic_text_diff +
            self.weight_domain_signal +
            self.weight_vlm_judgment
        )
        
        # Handle minor floating point inaccuracies
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                f"Aggregation weights must sum to exactly 1.0. Current sum is {total_weight}"
            )
            
        return self


# Instantiate the settings securely at module load time
settings = Settings()

# Ensure the designated upload/processing directory exists immediately
os.makedirs(settings.upload_dir, exist_ok=True)