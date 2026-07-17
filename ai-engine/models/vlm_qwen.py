"""
Qwen (Alibaba DashScope) implementation of the VLM Service.
"""

import base64
import json
from typing import Any

import openai

from config.settings import settings
from models.vlm_base import VLMService, VLMResult


class QwenVLMService(VLMService):
    """
    Concrete implementation of VLMService using Alibaba's Qwen Vision models 
    via the DashScope OpenAI-compatible endpoint.
    """

    def __init__(self) -> None:
        """
        Initialize the OpenAI client pointing to the Qwen base URL.
        Requires QWEN_API_KEY and QWEN_API_BASE_URL to be set.
        """
        self.client = openai.OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_api_base_url
        )
        self.model = settings.vlm_model
        self.max_tokens = settings.vlm_max_tokens

    def analyze_batch(self, images: list[bytes], context: dict[str, Any]) -> VLMResult:
        """
        Analyzes a batch of chronological screenshots using Qwen's VL models.
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

        # Prompt explicitly enforcing a JSON schema response
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
            "Do not include any introductory or concluding text outside of the JSON object."
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        # Qwen's compatible mode accepts the standard base64 image_url format
        for img_bytes in images:
            b64_data = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_data}"
                }
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                # Note: Not all Qwen models strictly support the `response_format={"type": "json_object"}` 
                # parameter via the compatible API, so we rely on prompt engineering and markdown stripping.
            )
            
            raw_text = response.choices[0].message.content
            if not raw_text:
                raise ValueError("Received empty content from Qwen")

            # Strip potential markdown formatting (Qwen often wraps JSON in ```json)
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            parsed_json = json.loads(cleaned_text.strip())
            
            return VLMResult(
                task_description=parsed_json.get("task_description", "Unknown task"),
                progressed=bool(parsed_json.get("progressed", False)),
                score=float(parsed_json.get("score", 0.0)),
                confidence=float(parsed_json.get("confidence", 0.0)),
                evidence=parsed_json.get("evidence", []),
                raw_response={"raw_text": raw_text, "usage": dict(response.usage) if response.usage else {}}
            )
            
        except openai.OpenAIError as e:
            return self._build_error_result(f"Qwen API Error: {str(e)}")
        except json.JSONDecodeError as e:
            return self._build_error_result(f"Failed to parse Qwen's response as JSON: {str(e)}")
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