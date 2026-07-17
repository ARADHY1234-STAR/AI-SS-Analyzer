"""
Domain-specific detector for document writing and word processing activities.
"""

import re

def _get_word_count(text: str) -> int:
    """
    Returns the approximate number of words in a given text.
    """
    return len(re.findall(r'\b\w+\b', text))


def _is_prose_heavy(text: str) -> bool:
    """
    Evaluates if the text structurally resembles natural prose 
    (high density of alphanumeric characters and spaces relative to special symbols).
    """
    if not text or len(text) < 50:
        return False
        
    letters_spaces = sum(1 for c in text if c.isalpha() or c.isspace())
    return (letters_spaces / len(text)) > 0.85


def detect(ocr_texts: list[str]) -> dict[str, float | bool]:
    """
    Analyzes a sequence of OCR texts to determine if document writing occurred
    and measures the progress based on word count deltas.

    Args:
        ocr_texts: Chronological list of extracted text from frames.

    Returns:
        A dictionary containing the domain signal metrics:
        - is_active: True if word processing activity was detected.
        - progress_score: A float (0.0 to 1.0) indicating the volume of text changes.
    """
    if len(ocr_texts) < 2:
        return {"is_active": False, "progress_score": 0.0}

    total_progress_signal = 0.0
    active_frames = 0

    for i in range(1, len(ocr_texts)):
        prev_text = ocr_texts[i-1]
        curr_text = ocr_texts[i]

        # If the current frame looks like a word processor (heavy prose density)
        if _is_prose_heavy(curr_text):
            active_frames += 1
            
            prev_words = _get_word_count(prev_text)
            curr_words = _get_word_count(curr_text)
            
            # Meaningful change in word count (addition or significant editing/deletion)
            # Threshold set to >3 to ignore minor OCR noise/flicker
            if abs(curr_words - prev_words) > 3:
                total_progress_signal += 1.0

    is_active = active_frames > 0
    
    # Normalize the score
    transitions = len(ocr_texts) - 1
    score = min(total_progress_signal / transitions, 1.0) if transitions > 0 else 0.0

    return {
        "is_active": is_active,
        "progress_score": score if is_active else 0.0
    }