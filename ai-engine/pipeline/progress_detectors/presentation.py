"""
Domain-specific detector for presentation and slide deck activities.
"""

import re

# Regex to identify common presentation artifacts: bullet points, numbered lists, or short title lines
BULLET_REGEX = re.compile(r'(?:^|\n)\s*(?:[•\-*○●]|(?:\d+\.))\s+', re.IGNORECASE)

def _get_bullet_point_count(text: str) -> int:
    """
    Counts the number of identifiable bullet points or list items in the text.
    """
    if not text:
        return 0
    return len(BULLET_REGEX.findall(text))


def _is_presentation_format(text: str) -> bool:
    """
    Evaluates if the text structurally resembles a presentation slide.
    Slides typically feature a low overall word count, high line break frequency, 
    and the presence of bullet points or short titles.
    """
    if not text or len(text) < 20:
        return False
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return False
        
    avg_line_length = sum(len(line) for line in lines) / len(lines)
    bullet_count = _get_bullet_point_count(text)
    
    # A standard slide often has short average line lengths (titles + bullets) 
    # and explicitly uses list formatting.
    is_short_lines = avg_line_length < 60
    has_bullets = bullet_count >= 2
    
    return is_short_lines and has_bullets


def detect(ocr_texts: list[str]) -> dict[str, float | bool]:
    """
    Analyzes a sequence of OCR texts to determine if presentation creation/editing occurred
    and measures the progress based on text deltas and slide transitions.

    Args:
        ocr_texts: Chronological list of extracted text from frames.

    Returns:
        A dictionary containing the domain signal metrics:
        - is_active: True if presentation activity was detected.
        - progress_score: A float (0.0 to 1.0) indicating the volume of changes.
    """
    if len(ocr_texts) < 2:
        return {"is_active": False, "progress_score": 0.0}

    total_progress_signal = 0.0
    active_frames = 0

    for i in range(1, len(ocr_texts)):
        prev_text = ocr_texts[i-1]
        curr_text = ocr_texts[i]

        # Check if the current frame resembles a slide deck (e.g., PowerPoint, Google Slides, Keynote)
        if _is_presentation_format(curr_text):
            active_frames += 1
            
            # Progress in presentations is usually either editing a slide (length/bullets change)
            # or moving to a completely new slide (large text delta).
            prev_bullets = _get_bullet_point_count(prev_text)
            curr_bullets = _get_bullet_point_count(curr_text)
            
            len_diff = abs(len(curr_text) - len(prev_text))
            bullet_diff = abs(curr_bullets - prev_bullets)
            
            # Meaningful change threshold: added/removed a bullet point OR changed >15 characters 
            if bullet_diff > 0 or len_diff > 15:
                total_progress_signal += 1.0

    is_active = active_frames > 0
    
    # Normalize the score
    transitions = len(ocr_texts) - 1
    score = min(total_progress_signal / transitions, 1.0) if transitions > 0 else 0.0

    return {
        "is_active": is_active,
        "progress_score": score if is_active else 0.0
    }