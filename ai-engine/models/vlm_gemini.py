"""
Google Gemini implementation of the VLM Service.
"""

import json
from typing import Any

import google.generativeai as genai
from google.generativeai.types import generation_types

from config.settings import settings
from models.vlm_base import VLMService, VLMResult


class GeminiVLMService(VLMService):
    """
    Concrete implementation of VLMService using Google's Gemini models.
    """

    def __init__(self) -> None:
        """
        Initialize the Google Generative AI client.
        Requires GEMINI_API_KEY to be set in the environment/settings.
        """
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = settings.vlm_model
        self.max_tokens = settings.vlm_max_tokens
        
        # Instantiate the model reference
        self.model = genai.GenerativeModel(self.model_name)

    def analyze_batch(self, images: list[bytes], context: dict[str, Any]) -> VLMResult:
        """
        Analyzes a batch of chronological screenshots using Gemini's multimodal capabilities.
        """
        if not images:
            return VLMResult(
                task_description="No images provided",
                progressed=False,
                score=0.0,
                confidence=1.0,
                evidence=["Empty image batch received."],
                raw_response={}
            )

        # The prompt dictating the reasoning task
        prompt_text = (
            "You are an AI analyzing a sequence of user workspace screenshots to determine productivity and task progress. "
            f"Context metadata: {json.dumps(context)}\n\n"
            "MEETING RULE: If visual evidence indicates an active meeting or video call (e.g., Zoom, Google Meet, MS Teams), ignore low keyboard/mouse activity and do not penalize it. Treat the meeting as productive time unless visually proven otherwise.\n\n"
            "Analyze the provided chronological screenshots and output your reasoning STRICTLY as a JSON object with the following schema:\n"
            "{\n"
            '  "task_description": "A short string describing the user\'s current task",\n'
            '  "progressed": true/false,\n'
            '  "score": a float between 0.0 and 1.0 representing work done,\n'
            '  "confidence": a float between 0.0 and 1.0 representing your certainty,\n'
            '  "evidence": [\n'
            '    "What was detected: <Explain exactly what apps and tasks the user is interacting with>",\n'
            '    "Why it was classified that way: <Explain your reasoning for the productivity score>",\n'
            '    "Visual evidence: <List specific frame numbers, UI elements, or text changes that support this>",\n'
            '    "Recommendation: <Explain how or why productivity could increase based on these screenshots>"\n'
            '  ]\n'
            "}\n"
        )

        # Gemini accepts a mixed list of strings and dictionaries for multimodal payloads
        contents: list[Any] = [prompt_text]

        for img_bytes in images:
            contents.append({
                "mime_type": "image/jpeg",
                "data": img_bytes
            })

        try:
            # We use generation_config to natively enforce JSON output
            response = self.model.generate_content(
                contents,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    max_output_tokens=self.max_tokens
                )
            )
            
            raw_text = response.text
            if not raw_text:
                raise ValueError("Received empty content from Gemini")

            parsed_json = json.loads(raw_text.strip())
            
            return VLMResult(
                task_description=parsed_json.get("task_description", "Unknown task"),
                progressed=bool(parsed_json.get("progressed", False)),
                score=float(parsed_json.get("score", 0.0)),
                confidence=float(parsed_json.get("confidence", 0.0)),
                evidence=parsed_json.get("evidence", []),
                raw_response={"raw_text": raw_text}
            )
            
        except generation_types.StopCandidateException as e:
            return self._build_error_result(f"Gemini generation stopped unexpectedly: {str(e)}")
        except json.JSONDecodeError as e:
            return self._build_error_result(f"Failed to parse Gemini's response as JSON: {str(e)}")
        except Exception as e:
            return self._build_error_result(f"Unexpected error during VLM analysis: {str(e)}")

    def _build_error_result(self, error_msg: str) -> VLMResult:
        """Helper to construct a standard failure result."""
        return VLMResult(
            task_description="VLM Analysis Failed",
            progressed=False,
            score=0.0,
            confidence=0.0,
            evidence=[error_msg],
            raw_response={"error": error_msg}
        )