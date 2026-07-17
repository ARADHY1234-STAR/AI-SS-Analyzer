"""
Anthropic implementation of the VLM Service.
"""

import base64
import json
from typing import Any

import anthropic

from config.settings import settings
from models.vlm_base import VLMService, VLMResult


class AnthropicVLMService(VLMService):
    """
    Concrete implementation of VLMService using Anthropic's Claude models.
    """

    def __init__(self) -> None:
        """
        Initialize the Anthropic client. 
        Requires ANTHROPIC_API_KEY to be set in the environment/settings.
        """
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.vlm_model
        self.max_tokens = settings.vlm_max_tokens

    def analyze_batch(self, images: list[bytes], context: dict[str, Any]) -> VLMResult:
        """
        Analyzes a batch of chronological screenshots using Claude's vision capabilities.
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

        content_blocks: list[dict[str, Any]] = []
        
        # Claude accepts a list of content blocks for multi-modal input.
        # We append each image in chronological order.
        for img_bytes in images:
            b64_data = base64.b64encode(img_bytes).decode("utf-8")
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg", # Defaulting to jpeg for API transport
                    "data": b64_data,
                }
            })

        # Append the strict JSON-enforcing text prompt
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
        
        content_blocks.append({
            "type": "text",
            "text": prompt_text
        })

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": content_blocks
                    }
                ]
            )
            
            # The response content is a list of blocks; grab the text block
            raw_text = response.content[0].text
            
            # Strip potential markdown formatting (Claude sometimes wraps JSON in ```json)
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
                raw_response={"raw_text": raw_text}
            )
            
        except anthropic.APIError as e:
            return self._build_error_result(f"Anthropic API Error: {str(e)}")
        except json.JSONDecodeError as e:
            return self._build_error_result(f"Failed to parse Claude's response as JSON: {str(e)}")
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