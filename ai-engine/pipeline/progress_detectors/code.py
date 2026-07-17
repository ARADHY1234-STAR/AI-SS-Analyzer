"""
Domain-specific detector for software development and coding activities.
"""

import re

# Common syntax markers across major programming languages (Python, TS/JS, Java, HTML, etc.)
CODE_MARKERS = [
    r"\bdef\b", r"\bclass\b", r"\bimport\b", r"\bfrom\b", r"\breturn\b", 
    r"\bfunction\b", r"\bconst\b", r"\blet\b", r"\bvar\b", r"=\>", 
    r"\bpublic\b", r"\bprivate\b", r"\binterface\b", r"console\.log", 
    r"print\(", r"<div>", r"</div>", r"\{", r"\}"
]
CODE_REGEX = re.compile("|".join(CODE_MARKERS), re.IGNORECASE)


def _calculate_code_density(text: str) -> float:
    """
    Estimates how likely a piece of text is source code based on syntax markers.
    Returns a float between 0.0 (no code) and 1.0 (highly likely to be code).
    """
    if not text:
        return 0.0
        
    matches = len(CODE_REGEX.findall(text))
    # Heuristic: 5 or more distinct syntax markers in a single screen strongly indicates an IDE
    return min(matches / 5.0, 1.0)


def detect(ocr_texts: list[str]) -> dict[str, float | bool]:
    """
    Analyzes a sequence of OCR texts to determine if coding occurred and measures progress.

    Args:
        ocr_texts: Chronological list of extracted text from frames.

    Returns:
        A dictionary containing the domain signal metrics:
        - is_active: True if coding activity was detected in the session.
        - progress_score: A float (0.0 to 1.0) indicating the volume of code changes.
    """
    if len(ocr_texts) < 2:
        return {"is_active": False, "progress_score": 0.0}

    total_progress_signal = 0.0
    active_frames = 0

    for i in range(1, len(ocr_texts)):
        prev_text = ocr_texts[i-1]
        curr_text = ocr_texts[i]

        curr_density = _calculate_code_density(curr_text)
        
        # If the current frame looks like an IDE or text editor containing code
        if curr_density >= 0.4:
            active_frames += 1
            
            # Check for meaningful keystrokes/changes (ignoring trivial 1-5 char noise from OCR flutter)
            len_diff = abs(len(curr_text) - len(prev_text))
            if len_diff > 5:
                total_progress_signal += 1.0

    is_active = active_frames > 0
    
    # Normalize the score between 0.0 and 1.0 based on total possible frame transitions
    transitions = len(ocr_texts) - 1
    score = min(total_progress_signal / transitions, 1.0) if transitions > 0 else 0.0

    return {
        "is_active": is_active,
        "progress_score": score if is_active else 0.0
    }