"""
Domain-specific detector for project management and ticketing activities.
"""

import re

# Common keywords found in project management tools (Jira, Linear, Trello, GitHub Issues)
TICKET_UI_MARKERS = [
    r"\bAssignee\b", r"\bReporter\b", r"\bStatus\b", r"\bPriority\b",
    r"\bTo Do\b", r"\bIn Progress\b", r"\bDone\b", r"\bSprint\b",
    r"\bBacklog\b", r"\bEpic\b", r"\bStory Points\b", r"\bLabels\b",
    r"\bPull Request\b", r"\bSub-tasks\b"
]
TICKET_UI_REGEX = re.compile("|".join(TICKET_UI_MARKERS), re.IGNORECASE)

# Regex to match standard ticket IDs (e.g., PROJ-123, ENG-4567, or GitHub-style #890)
TICKET_ID_REGEX = re.compile(r'\b[A-Z]{2,10}-\d+\b|#\d+\b')


def _get_ui_marker_count(text: str) -> int:
    """Counts occurrences of specific project management UI keywords."""
    if not text:
        return 0
    return len(TICKET_UI_REGEX.findall(text))


def _get_ticket_id_count(text: str) -> int:
    """Counts the number of unique ticket identifiers visible on screen."""
    if not text:
        return 0
    # Use a set to count unique IDs, as a single ticket page often repeats its ID multiple times
    return len(set(TICKET_ID_REGEX.findall(text)))


def _is_ticketing_tool(text: str) -> bool:
    """
    Evaluates if the text structurally resembles a ticketing or project management tool.
    """
    if not text or len(text) < 20:
        return False
        
    markers = _get_ui_marker_count(text)
    ids = _get_ticket_id_count(text)
    
    # A standard Jira/Linear screen will usually have at least one ticket ID 
    # combined with standard UI markers, or multiple UI markers for a board view.
    return markers >= 3 or (markers >= 1 and ids >= 1)


def detect(ocr_texts: list[str]) -> dict[str, float | bool]:
    """
    Analyzes a sequence of OCR texts to determine if ticket management activity occurred
    and measures progress based on text deltas and ticket navigation.

    Args:
        ocr_texts: Chronological list of extracted text from frames.

    Returns:
        A dictionary containing the domain signal metrics:
        - is_active: True if ticketing activity was detected.
        - progress_score: A float (0.0 to 1.0) indicating the volume of changes.
    """
    if len(ocr_texts) < 2:
        return {"is_active": False, "progress_score": 0.0}

    total_progress_signal = 0.0
    active_frames = 0

    for i in range(1, len(ocr_texts)):
        prev_text = ocr_texts[i-1]
        curr_text = ocr_texts[i]

        # If the current frame resembles Jira, Linear, Trello, etc.
        if _is_ticketing_tool(curr_text):
            active_frames += 1
            
            # Progress usually involves either navigating to different tickets/boards
            # or typing long descriptions/comments inside a specific ticket.
            prev_ids = _get_ticket_id_count(prev_text)
            curr_ids = _get_ticket_id_count(curr_text)
            
            len_diff = abs(len(curr_text) - len(prev_text))
            
            # Meaningful change: navigated to a different view (IDs changed) 
            # OR typed a significant comment/description (> 15 chars difference)
            if abs(curr_ids - prev_ids) > 0 or len_diff > 15:
                total_progress_signal += 1.0

    is_active = active_frames > 0
    
    # Normalize the score
    transitions = len(ocr_texts) - 1
    score = min(total_progress_signal / transitions, 1.0) if transitions > 0 else 0.0

    return {
        "is_active": is_active,
        "progress_score": score if is_active else 0.0
    }