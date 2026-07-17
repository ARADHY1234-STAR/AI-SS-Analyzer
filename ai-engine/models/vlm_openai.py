"""
OpenAI-compatible Vision-Language Model implementation.
Natively supports OpenAI (GPT-4o), Groq (LLaMA 3.2 Vision), and OpenRouter
with integrated 20-minute idle detection, chronological timeline interval
summaries, and Pillow-based image optimization for high-frame-count batches.
"""

import base64
import io
import json
import logging
import os
from typing import Any

from openai import OpenAI
from PIL import Image

from config.settings import settings
from models.vlm_base import VLMService, VLMResult

# 1. Force strict offline operation to bypass the Hugging Face 504 CDN timeouts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def initialize_embedding_model():
    """Loads the CLIP model entirely from local cache to secure fast, offline bootups."""
    try:
        # Check local path first, default to local cache fallback
        local_path = "./my_models/clip-ViT-B-32"
        model_path = local_path if os.path.exists(local_path) else "sentence-transformers/clip-ViT-B-32"
        
        return SentenceTransformer(model_path, local_files_only=True)
    except Exception as e:
        print(f"\n[WARNING] Offline model load failed: {e}")
        print("Please run: hf download sentence-transformers/clip-ViT-B-32 --local-dir ./my_models/clip-ViT-B-32\n")
        # Direct fallback attempt if path check was strict
        try:
            return SentenceTransformer("sentence-transformers/clip-ViT-B-32", local_files_only=True)
        except Exception as fallback_err:
            print(f"\n[ERROR] Direct fallback offline load also failed: {fallback_err}")
            return None


class ProductivityAnalyzer:
    def __init__(self):
        self.encoder = initialize_embedding_model()

    def analyze_batch(self, telemetry_data, parsed_vlm_json, ss_are_same: bool = False):
        """
        Cross-examines VLM outputs against actual physical telemetry logs.
        Enforces custom dual-path rules for meetings to protect genuine listeners
        while crushing identical-frame simulators and mouse jigglers.
        """
        strokes = telemetry_data.get("keyboard_strokes", 0)
        clicks = telemetry_data.get("mouse_clicks", 0)
        
        # Normalize to float to handle potential string format from VLM
        def to_float(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                cleaned = val.strip().lower()
                if cleaned in ("0", "0.0", "none", "zero", "0%", ""):
                    return 0.0
                try:
                    return float(cleaned)
                except ValueError:
                    pass
            return 1.0

        struct_diff = to_float(parsed_vlm_json.get("structural_diff", 0.0))
        sem_diff = to_float(parsed_vlm_json.get("semantic_diff", 0.0))
        
        if ss_are_same:
            struct_diff = 0.0
            sem_diff = 0.0
            
        judgment = parsed_vlm_json.get("vlm_judgment", "").lower()
        
        is_meeting = "meeting" in judgment or "conference" in judgment
        
        # Rule 1: Bit-by-bit identical clone (100% frozen, 0 inputs)
        if struct_diff == 0.0 and sem_diff == 0.0 and strokes == 0 and clicks == 0:
            parsed_vlm_json["score"] = 0.0
            parsed_vlm_json["domain_signal"] = 0.0
            parsed_vlm_json["vlm_judgment"] = "STAGNANT / NO PROGRESS"
            parsed_vlm_json["idle_detected"] = True
            parsed_vlm_json["progress_status"] = "NO PROGRESS DETECTED"
            parsed_vlm_json["evidence"] = (
                "Cheating detected: 100% frozen, bit-by-bit identical session "
                "with zero user interaction. Screen is unattended or fixed on a static mockup."
            )
            return parsed_vlm_json
        # Rule 2: The "Mouse Jiggler / Cursor Wiggle" Trap (Cheating via artificial mouse movement or frozen webcam feeds)
        # Why: When a cursor moves over a frozen screenshot, sem_diff is 0.0, but struct_diff 
        # rises slightly (0.01 - 0.02) due to the cursor arrow changing pixel coordinates.
        # Also, if we are in a meeting context but there is zero semantic text change and struct_diff is minimal,
        # it indicates no webcam feed frame changes (no head movement, no participant changes).
        is_static_meeting = is_meeting and sem_diff == 0.0 and struct_diff <= 0.02
        if (sem_diff == 0.0 and struct_diff <= 0.02 and strokes == 0 and clicks >= 0) or is_static_meeting:
            parsed_vlm_json["score"] = 0.0
            parsed_vlm_json["domain_signal"] = 0.0
            parsed_vlm_json["vlm_judgment"] = "CHEATING / MOUSE JIGGLER DETECTED"
            parsed_vlm_json["idle_detected"] = True
            parsed_vlm_json["progress_status"] = "NO PROGRESS DETECTED"
            parsed_vlm_json["evidence"] = (
                "Cheating detected: Screenshots are completely static all the way through; "
                "only the mouse cursor position is changing over a frozen background, or there is "
                "no head movement, face expression changes, or participant webcam feed updates in the meeting call. "
                "This indicates artificial simulation using a mouse jiggler or hovering over a static Photo Explorer image."
            )
            return parsed_vlm_json

        # Rule 3: Valid Active Listening Mode (Dynamic meeting with actual video/slide changes)
        # A real meeting will have visual shifts (> 0.02 struct_diff or > 0 sem_diff from slides/chat)
        if is_meeting and (struct_diff > 0.02 or sem_diff > 0.0) and (strokes < 50 and clicks < 20):
            parsed_vlm_json["score"] = max(parsed_vlm_json.get("score", 85.0), 85.0)
            parsed_vlm_json["progress_status"] = "PROGRESS CONFIRMED"
            parsed_vlm_json["idle_detected"] = False
            
            # Safe append to avoid string/list addition errors
            evidence = parsed_vlm_json.get("evidence", [])
            meeting_note = "(Validated active video meeting; legitimate visual progression confirmed without requiring keyboard density.)"
            if isinstance(evidence, list):
                evidence.append(meeting_note)
            else:
                parsed_vlm_json["evidence"] = str(evidence) + " " + meeting_note
            return parsed_vlm_json
            
        # Standard workflow evaluations proceed normally for general tasks...
        parsed_vlm_json["progress_status"] = "PROGRESS CONFIRMED" if parsed_vlm_json.get("score", 0) > 40 else "NO PROGRESS DETECTED"
        return parsed_vlm_json



class VLMOpenAI(VLMService):
    """
    Implementation for OpenAI and OpenAI-compatible vision models.
    Analyzes batches of 3 to 10 sequential screenshots to evaluate productivity,
    detect inactivity/idle notifications, and generate timeline breakdowns.

    Supports automatic OpenRouter detection: if an OpenRouter API key is found
    in the environment, the client is configured with the correct base URL and
    required HTTP headers without any hardcoded values.
    """

    # Maximum number of frames the service will accept in a single batch
    MAX_FRAMES = 10

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_override: str | None = None,
    ) -> None:
        """
        Initializes the OpenAI-compatible client with cascading key resolution.

        Key precedence order:
          1. Explicit `api_key` parameter (e.g., injected by the factory for Groq)
          2. `settings.openrouter_api_key`  (from OPENROUTER_API_KEY in .env)
          3. `os.getenv("OPENROUTER_API_KEY")`  (direct env fallback)
          4. `settings.openai_api_key`  (from OPENAI_API_KEY in .env)

        When an OpenRouter key is resolved, `base_url` defaults to the
        OpenRouter endpoint and the required HTTP headers are injected.

        Args:
            api_key: Optional explicit override for the API key.
            base_url: Optional explicit override for the base URL.
            model_override: Optional model name that overrides settings.vlm_model.
                            Useful for benchmarking multiple models on a single client.
        """
        # --- Step 1: Resolve the API key using cascading precedence ---
        final_api_key = api_key
        is_openrouter = False

        if not final_api_key:
            # Check OpenRouter key sources first
            openrouter_key = (
                getattr(settings, "openrouter_api_key", None)
                or os.getenv("OPENROUTER_API_KEY")
            )
            if openrouter_key:
                final_api_key = openrouter_key
                is_openrouter = True
            else:
                # Fall back to standard OpenAI key
                final_api_key = settings.openai_api_key

        if not final_api_key:
            raise ValueError(
                "No API key found. Provide one via parameter, "
                "OPENROUTER_API_KEY, or OPENAI_API_KEY in your .env file."
            )

        # --- Step 2: Resolve base URL ---
        if base_url:
            final_base_url = base_url
        elif is_openrouter:
            final_base_url = "https://openrouter.ai/api/v1"
        else:
            final_base_url = None  # Use OpenAI's default

        # --- Step 3: Build default headers (required by OpenRouter) ---
        default_headers = {
            "HTTP-Referer": "https://localhost:5500",
            "X-Title": "AI Productivity Engine",
        }

        # --- Step 4: Instantiate the OpenAI client ---
        client_kwargs: dict[str, Any] = {
            "api_key": final_api_key,
            "default_headers": default_headers,
        }
        if final_base_url:
            client_kwargs["base_url"] = final_base_url

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_override or settings.vlm_model
        self.max_tokens = settings.vlm_max_tokens
        self.analyzer = ProductivityAnalyzer()

        logger.info(
            f"VLMOpenAI initialized — model={self.model_name}, "
            f"base_url={final_base_url or 'default (OpenAI)'}, "
            f"openrouter={is_openrouter}"
        )

    # -----------------------------------------------------------------
    #  Image Optimization
    # -----------------------------------------------------------------

    def _optimize_image(self, img_bytes: bytes, max_width: int = 1280) -> str:
        """
        Resize and compress a screenshot to reduce token cost and avoid
        HTTP 429 rate-limit errors when sending up to 10 frames.

        Processing steps:
          1. Open the raw bytes as a PIL Image.
          2. If the image is wider than `max_width`, downscale proportionally.
          3. Convert RGBA / P colour modes to RGB (JPEG requirement).
          4. Save to an in-memory buffer as optimized JPEG (quality=80).
          5. Return the Base64-encoded string ready for a data URI.

        Args:
            img_bytes: Raw screenshot bytes (JPEG, PNG, etc.).
            max_width: Maximum pixel width; images wider than this are resized.

        Returns:
            Base64-encoded JPEG string.
        """
        img = Image.open(io.BytesIO(img_bytes))

        # Proportional downscale if wider than max_width
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        # JPEG does not support transparency — convert if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Compress to an in-memory JPEG buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80, optimize=True)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    # -----------------------------------------------------------------
    #  System Prompt
    # -----------------------------------------------------------------

    def _get_system_prompt(self, session_duration_mins: int, use_persona: bool = False, persona_id: str = "default_general") -> str:
        """
        Returns the strict system instructions defining the AI's persona, JSON schema,
        idle detection rules, and flexible time-based interval summaries.

        Updated for dynamic batch lengths and advanced cross-modal telemetry.
        """
        prompt = (
            "You are an expert productivity analyst. You will be provided with a chronological "
            f"batch of screenshots (representing ~{session_duration_mins} minutes of "
            "real-time session data) from a user's computer, along with metadata including OCR "
            "text, mathematical frame-to-frame difference heuristics, and INPUT TELEMETRY "
            "metrics.\n\n"
            
            "CRITICAL: INPUT TELEMETRY (telemetry_timeline):\n"
            "The context metadata contains a 'telemetry_timeline' array. Each entry is aligned "
            "to a frame by 'frame_index' and includes:\n"
            "  - 'keyboard_strokes': total key presses during the 15-minute frame interval\n"
            "  - 'mouse_clicks': total mouse clicks during the 15-minute frame interval\n"
            "  - 'active_window_title': the title of the active window\n"
            "You MUST read and cross-reference this telemetry with the visual evidence.\n\n"
            
            "QUANTITATIVE PERSONA TELEMETRY DENSITY MATRIX (baselines per 15-min interval):\n"
            "- coder: 400-800+ keyboard strokes / 15-60 mouse clicks. (Keyboard heavy).\n"
            "- frontend: 200-500 keyboard strokes / 100-300+ mouse clicks. (Balanced/Mouse heavy).\n"
            "- architect: 50-250 keyboard strokes / 40-150 mouse clicks. (Low density cognitive reading/review. Do not penalize low telemetry if complex diagrams/docs are visible).\n"
            "- qa: 100-300 keyboard strokes / 150-400+ mouse clicks. (Click heavy UI workflows).\n"
            "- analyst: 300-600 keyboard strokes / 80-200 mouse clicks. (Balanced).\n"
            "- default_general: 150+ keyboard strokes / 50+ mouse clicks. (Bypass role constraints, judge purely on general active creation vs idleness).\n\n"
            f"The user's active persona is: {persona_id}\n\n"
            
            "MANDATORY PERSONA MATRIX CALIBRATION (Apply these EXACT mathematical modifiers):\n"
            "1. For `coder`: If `keyboard_strokes` are < 200, apply a mandatory -30 point penalty to the final score, regardless of visual progress. If `keyboard_strokes` are > 500 and text diffs are visually present, award a +15 point modifier (allowing scores up to 100).\n"
            "2. For `frontend` or `qa`: If `mouse_clicks` are < 50, apply a mandatory -25 point penalty. If `mouse_clicks` match high structural UI changes, maximize the `structural_diff` signal.\n"
            "3. For `architect`: Completely invert the penalty logic. If telemetry is extremely low (e.g., < 50 strokes/clicks) but the screen shows architectural diagrams, code reviews, or documentation, apply a +20 point cognitive focus reward. If telemetry is high, scrutinize for distraction.\n"
            "4. For `default_general`: Evaluate telemetry linearly without strict domain filters. Targeted personas MUST apply the specific domain filters above. Changing the persona using the same dataset MUST result in a significantly different numerical matrix.\n\n"
            
            "STRICT CROSS-MODAL RULES:\n"
            "1. Cursor-Only Movement on Static Screen: If the screenshots are completely identical (the underlying content, windows, text, code, or video feeds do not change at all) and the ONLY difference across the images is that the mouse cursor is located in different positions on the screen, you MUST classify this session as IDLE / NO PROGRESS. You MUST set \"idle_detected\": true, set \"score\": 0.0, and set the final qualitative judgment to state that a cursor-only wiggle or static background was detected.\n"
            "2. Typing Validation: High keyboard activity must be verified by visual text evolution (OCR/code diff) between frames. If verified, lift the previous 90% score cap and allow scores up to 100 (range 90-100).\n"
            "3. Macro/Spam Guardrail: High telemetry paired with static visual frames or system/desktop/notification interaction must force \"idle_detected\": true and drop the score below 20.\n"
            "4. Cognitive Flow: Low telemetry metrics during `architect` or `default_general` sessions showing architectural layouts or documentation must not be penalized (target score 70-85).\n"
            "5. Meeting Dual-Path Calibration: If visual evidence indicates an active meeting or video call (e.g., Zoom, Google Meet, MS Teams), you MUST classify the session into exactly one of the following two paths:\n"
            "   PATH A — Dynamic Meeting (Valid Listening): The meeting frames show small but real structural or layout variations between Frame N and Frame N+1 (e.g., webcam feeds shifting, presentation slides advancing, chat panel messages appearing, participant tiles rearranging). In this scenario, even if keyboard/mouse telemetry is low or zero, the user is legitimately attending the meeting. Do NOT penalize low telemetry. Grant a lenient high score in the range 75-90, relax keyboard/mouse benchmarks entirely, and note in the evidence that dynamic meeting visuals validated active participation.\n"
            "   PATH B — Frozen Clone (System Cheating): The meeting frames are completely identical bit-by-bit across intervals (`structural_diff` == 0.0 AND `semantic_diff` == 0.0). A real video conferencing feed can NEVER produce perfectly identical frames. If consecutive screenshots match bit-by-bit while hardware telemetry is zero, you MUST explicitly mention in the response text (e.g., under evidence / vlm_judgment) that the session appears to be simulated or left unmonitored (e.g., \"User may be idling with an image viewer open or a frozen video feed\" or \"Photo Explorer or a static window loop\"), evaluate the status explicitly as a failure, forcefully drop the score to 0, and set `\"idle_detected\": true`.\n"
            "6. Static Session Cheat Detection: If screenshots match bit-by-bit (`structural_diff` == 0.0 and `semantic_diff` == 0.0) while hardware telemetry is zero, the session is stagnant/fraudulent. You MUST explicitly state that the user may be idling with a photo viewer/mockup or frozen app open, drop the score to 0, set `\"idle_detected\": true`, and evaluate the status explicitly as a failure.\n"
            "7. Mouse Jiggler / Cursor Wiggle & Frozen Webcam Guardrail: If the underlying workspace, video grid, or presentation content is completely static across frames, but the mouse cursor changes position (or clicks occur without any responsive UI changes), this is artificial activity simulation. Furthermore, if you detect a video meeting interface but notice absolutely no head movement of the user, no change in the webcam frames of meeting participants (frozen human video feeds), or no change in the presenter/content over multiple screenshots—even if the mouse cursor or overlay moves slightly—this MUST be classified as cheating (using a frozen meeting mockup, background image, or loop). You MUST explicitly state in the response evaluation: \"Screenshots are completely static across frames; only the mouse cursor is moving, indicating artificial activity simulation (e.g., using a mouse jiggler or hovering over a static Photo Explorer image). No head movement or webcam updates were detected in the meeting, confirming a static simulation.\" Drop the score to 0, set `\"idle_detected\": true`, and classify this as a cheating failure.\n\n"
            
            "CRITICAL SCORING DIRECTIVE (ANTI-HEDGING):\n"
            "Score on a scale of 0 to 100.\n"
            "1. LOW EXTREME (0-20): Visual evidence shows no meaningful work creation (desktop, notifications, idle).\n"
            "2. HIGH EXTREME (80-100): Clear workflow progression and domain congruence.\n"
            "3. THE RESTRICTED MIDDLE (35-65): Only for legitimate *passive* work. Do not use as a safe compromise.\n\n"
            
            "You MUST return a valid JSON object exactly matching these keys:\n"
            "{\n"
            '  "score": float between 0 and 100,\n'
            '  "confidence": float between 0.0 and 1.0,\n'
            '  "evidence": [\n'
            '    "What was detected: <Explain exactly what apps, tasks, or idle indicators/notifications were seen>",\n'
            '    "Why it was classified that way: <Explain your reasoning for the overall score and idle status>",\n'
            '    "Visual evidence: <List specific frame numbers, UI elements, notifications, or text changes>",\n'
            '    "Telemetry evidence: <Cite specific frame keyboard/mouse counts that influenced scoring>",\n'
            '    "Recommendation: <Explain how or why productivity could increase or flag an idle intervention>"\n'
            '  ],\n'
            '  "structural_diff": "string (summary of UI layout changes)",\n'
            '  "semantic_diff": "string (summary of text/code changes)",\n'
            '  "domain_signal": "string (how well the activity matches the persona)",\n'
            '  "vlm_judgment": "string (final qualitative verdict on the interval)",\n'
            '  "idle_detected": boolean (true if idle/spam, false otherwise),\n'
            '  "is_just_notification_clearing": boolean (true if solely desktop/notification interaction),\n'
            '  "persona_alignment_notes": "A brief sentence explaining exactly how the telemetry matched or violated the specific selected persona\'s constraints."\n'
            "}\n"
            "IMPORTANT: The 'evidence' array MUST contain exactly those five distinct strings, starting with those exact prefixes.\n"
        )
        return prompt

    # -----------------------------------------------------------------
    #  Core Analysis
    # -----------------------------------------------------------------

    def analyze_batch(
        self,
        images: list[bytes],
        context: dict[str, Any],
        model_override: str | None = None,
        persona_id: str | None = None,
    ) -> VLMResult:
        """
        Submits the sequential screenshots and metadata to the OpenAI-compatible API.

        Accepts up to 10 frames.  Each frame is run through `_optimize_image`
        (resize + JPEG compression) before being Base64-encoded into the payload,
        preventing token-limit crashes and HTTP 429 rate-limit errors.

        Args:
            images: List of raw byte strings for each screenshot (supports 3 to 10 frames).
            context: Dictionary containing session metadata, OCR results, and heuristics.
            model_override: Optional per-call model name override for benchmarking.

        Returns:
            VLMResult containing parsed productivity scores, timeline breakdowns, and idle flags.
        """
        if not images:
            raise ValueError("No images provided for VLM analysis.")

        # Enforce the 10-frame ceiling
        images = images[:self.MAX_FRAMES]

        # Resolve which model to use for this specific call
        active_model = model_override or self.model_name

        # The frame interval is 15 minutes per screenshot.
        session_duration_mins = len(images) * 15

        logger.info(
            f"Sending {len(images)} frames (~{session_duration_mins} mins of session data) "
            f"to model: {active_model}"
        )

        use_persona = persona_id != "default_general"
        actual_persona = persona_id or "default_general"

        # 1. Construct the message payload with injected metadata
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Here is the chronological session data (total frames: {len(images)}). "
                    f"Context metadata: {json.dumps(context)}"
                ),
            }
        ]

        # 2. Optimize and append all images as Base64 data URIs
        for idx, img_bytes in enumerate(images):
            try:
                b64_str = self._optimize_image(img_bytes)
            except Exception as opt_err:
                logger.warning(
                    f"Image optimization failed for frame {idx}, "
                    f"falling back to raw Base64: {opt_err}"
                )
                b64_str = base64.b64encode(img_bytes).decode("utf-8")

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_str}",
                    "detail": "auto",
                },
            })

        messages = [
            {"role": "system", "content": self._get_system_prompt(session_duration_mins=session_duration_mins, use_persona=use_persona, persona_id=actual_persona)},
            {"role": "user", "content": user_content},
        ]

        try:
            # 3. Execute the API call
            response = self.client.chat.completions.create(
                model=active_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Keep low for analytical consistency and strict JSON adherence
                response_format={"type": "json_object"},
            )

            raw_output = response.choices[0].message.content

            # 3a. Validate the raw output before attempting JSON parse
            if not raw_output or not raw_output.strip().startswith("{"):
                logger.warning(
                    f"VLM returned empty or non-JSON response (model: {active_model}). "
                    f"Raw output preview: {repr((raw_output or '')[:200])}"
                )
                raise json.JSONDecodeError(
                    msg="Response is empty or does not start with '{'",
                    doc=raw_output or "",
                    pos=0,
                )

            parsed_data = json.loads(raw_output)

            # Python Backend Post-Processing
            telemetry_data = context.get("telemetry_data", {})
            if not telemetry_data:
                # Build it from timeline if available
                timeline = context.get("timeline_metrics", [])
                telemetry_data = {
                    "keyboard_strokes": sum(t.get("keyboard_strokes") or 0 for t in timeline),
                    "mouse_clicks": sum(t.get("mouse_strokes") or 0 for t in timeline)
                }

            # Check if screenshots are mathematically static (no change in any frame interval)
            frame_data = context.get("frame_data", [])
            ss_are_same = len(frame_data) > 1 and all(
                frame.get("metrics", {}).get("is_idle", False) or frame.get("metrics", {}).get("is_trivial_change", False)
                for frame in frame_data[1:]
            )

            # Nuanced Meeting Engine Validation via ProductivityAnalyzer
            parsed_data = self.analyzer.analyze_batch(telemetry_data, parsed_data, ss_are_same=ss_are_same)

            if parsed_data.get("is_just_notification_clearing") or parsed_data.get("idle_detected"):
                parsed_data["score"] = min(float(parsed_data.get("score", 15)), 15.0)

            # Ensure any score under 20 is strictly mapped to NO PROGRESS DETECTED
            if float(parsed_data.get("score", 0.0)) < 20.0:
                parsed_data["progress_status"] = "NO PROGRESS DETECTED"

            raw_evidence = parsed_data.get("evidence", [])
            if isinstance(raw_evidence, str):
                evidence_list = [raw_evidence]
            elif isinstance(raw_evidence, list):
                evidence_list = [str(item) for item in raw_evidence]
            else:
                evidence_list = []

            # 4. Map output to domain VLMResult class
            result = VLMResult(
                task_description=parsed_data.get("vlm_judgment", "Unknown Task"),
                progressed=parsed_data.get("progress_status") == "PROGRESS CONFIRMED",
                score=float(parsed_data.get("score", 0.0)),
                confidence=float(parsed_data.get("confidence", 0.5)),
                evidence=evidence_list,
                activity_summary=[
                    f"Structural Diff: {parsed_data.get('structural_diff', '')}",
                    f"Semantic Diff: {parsed_data.get('semantic_diff', '')}"
                ],
                raw_response={
                    "provider": "openai_compatible",
                    "raw": parsed_data,
                    "timeline_summary": [],
                    "idle_detected": parsed_data.get("idle_detected", False),
                    "model_used": active_model,
                },
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse VLM JSON output from model '{active_model}': {e}"
            )
            # Graceful fallback — keeps the pipeline alive instead of crashing
            return VLMResult(
                task_description="Model service temporarily overloaded or rate-limited.",
                progressed=False,
                score=0.0,
                confidence=0.0,
                evidence=[
                    f"What was detected: The AI model '{active_model}' returned a response "
                    "that could not be parsed as valid JSON.",
                    "Why it was classified that way: This typically occurs when the model is "
                    "rate-limited (HTTP 429), overloaded, or returns an HTML error page instead "
                    "of the expected JSON payload.",
                    f"Visual evidence: Raw output preview — {repr((raw_output if 'raw_output' in dir() else '')[:300])}",
                    "Telemetry evidence: Telemetry data was prepared but could not be evaluated "
                    "because the model did not return a usable analysis.",
                    "Recommendation: Retry the request in 30-60 seconds, or switch to a "
                    "different model (e.g., a paid Gemini fallback) if the issue persists.",
                ],
                activity_summary=[
                    "Analysis could not be completed due to an API error.",
                    "No productivity assessment is available for this batch.",
                ],
                raw_response={
                    "error": "JSONDecodeError",
                    "error_detail": str(e),
                    "raw_string": raw_output if "raw_output" in dir() else None,
                    "model_used": active_model,
                },
            )

        except Exception as e:
            logger.error(
                f"OpenAI/Groq/OpenRouter API request failed for model "
                f"'{active_model}': {e}",
                exc_info=True,
            )
            # Graceful fallback for network timeouts, connection resets, auth errors, etc.
            return VLMResult(
                task_description="Model service temporarily overloaded or rate-limited.",
                progressed=False,
                score=0.0,
                confidence=0.0,
                evidence=[
                    f"What was detected: The API request to model '{active_model}' failed "
                    f"with error: {type(e).__name__}: {e}",
                    "Why it was classified that way: This is typically caused by a network "
                    "timeout, connection reset, authentication failure, or the provider being "
                    "temporarily unavailable.",
                    "Visual evidence: No visual analysis was possible because the API call "
                    "did not complete successfully.",
                    "Telemetry evidence: Telemetry data was prepared but could not be evaluated "
                    "because the model did not return a usable analysis.",
                    "Recommendation: Check your API key and network connectivity. If the "
                    "provider is rate-limiting, wait 60 seconds before retrying or switch "
                    "to a different model.",
                ],
                activity_summary=[
                    "Analysis could not be completed due to a network or API error.",
                    "No productivity assessment is available for this batch.",
                ],
                raw_response={
                    "error": type(e).__name__,
                    "error_detail": str(e),
                    "model_used": active_model,
                },
            )