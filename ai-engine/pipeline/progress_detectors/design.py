"""
Domain-specific detector for UI/UX design and graphic editing activities.
"""

import re

# Keywords commonly found in the UI panels of design tools (Figma, Photoshop, Illustrator)
DESIGN_UI_MARKERS = [
    r"\bOpacity\b", r"\bFill\b", r"\bStroke\b", r"\bEffects\b", 
    r"\bAuto layout\b", r"\bConstraints\b", r"\bLayer\b", r"\bGroup\b",
    r"\bFrame\b", r"\bAlign\b", r"\bDrop shadow\b", r"\bExport\b"
]
DESIGN_UI_REGEX = re.compile("|".join(DESIGN_UI_MARKERS), re.IGNORECASE)

# Regex for common design property values (hex codes, pixel values, percentages, angles)
PROPERTY_VALUE_REGEX = re.compile(
    r'#[0-9a-fA-F]{3,8}\b|\b\d+(?:\.\d+)?(?:px|pt|%|deg)\b', 
    re.IGNORECASE
)


def _get_ui_marker_count(text: str) -> int:
    """Counts occurrences of specific design tool UI panel keywords."""
    if not text:
        return 0
    return len(DESIGN_UI_REGEX.findall(text))


def _get_property_value_count(text: str) -> int:
    """Counts occurrences of CSS-like properties or numeric values used in design."""
    if not text:
        return 0
    return len(PROPERTY_VALUE_REGEX.findall(text))


def _is_design_tool(text: str) -> bool:
    """
    Evaluates if the text structurally resembles a design tool's interface.
    Design tools usually have specific UI panel keywords visible on screen.
    """
    if not text or len(text) < 20:
        return False
        
    # Finding 3 or more distinct design-specific UI markers strongly suggests 
    # an inspector or layers panel is currently visible on screen.
    return _get_ui_marker_count(text) >= 3


def detect(ocr_texts: list[str]) -> dict[str, float | bool]:
    """
    Analyzes a sequence of OCR texts to determine if design/graphics activity occurred
    and measures progress based on property tweaks and text delta.

    Args:
        ocr_texts: Chronological list of extracted text from frames.

    Returns:
        A dictionary containing the domain signal metrics:
        - is_active: True if design activity was detected.
        - progress_score: A float (0.0 to 1.0) indicating the volume of changes.
    """
    if len(ocr_texts) < 2:
        return {"is_active": False, "progress_score": 0.0}

    total_progress_signal = 0.0
    active_frames = 0

    for i in range(1, len(ocr_texts)):
        prev_text = ocr_texts[i-1]
        curr_text = ocr_texts[i]

        # If the current frame resembles Figma, Sketch, Photoshop, etc.
        if _is_design_tool(curr_text):
            active_frames += 1
            
            # Progress in design tools often manifests as tweaking properties in the inspector
            # (e.g., changing a hex color from #FFF to #000, adjusting padding from 16px to 24px).
            prev_props = _get_property_value_count(prev_text)
            curr_props = _get_property_value_count(curr_text)
            
            # Also check for structural text changes (e.g., renaming layers, typing inside text boxes)
            len_diff = abs(len(curr_text) - len(prev_text))
            
            # Meaningful change: added/removed/adjusted a property value OR made a visible text edit > 10 chars
            if abs(curr_props - prev_props) > 0 or len_diff > 10:
                total_progress_signal += 1.0

    is_active = active_frames > 0
    
    # Normalize the score
    transitions = len(ocr_texts) - 1
    score = min(total_progress_signal / transitions, 1.0) if transitions > 0 else 0.0

    return {
        "is_active": is_active,
        "progress_score": score if is_active else 0.0
    }