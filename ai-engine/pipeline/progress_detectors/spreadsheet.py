"""
Domain-specific detector for spreadsheet and data entry activities.
"""

import re

# Regex to identify typical spreadsheet data: integers, decimals, currencies, percentages, and common formulas
DATA_CELL_REGEX = re.compile(
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\b|\$[\d\.,]+|[\d\.,]+%|=SUM\(|=AVERAGE\(|=VLOOKUP\(', 
    re.IGNORECASE
)


def _count_data_cells(text: str) -> int:
    """
    Estimates the number of active numerical data points or formulas in the text.
    """
    if not text:
        return 0
    return len(DATA_CELL_REGEX.findall(text))


def _is_spreadsheet_heavy(text: str) -> bool:
    """
    Evaluates if the screen structurally resembles a spreadsheet or data dashboard.
    Relies on a heuristic checking for a high absolute volume of numerical entities.
    """
    if not text or len(text) < 50:
        return False
        
    # A standard spreadsheet screen will easily have 15+ numerical data points visible
    return _count_data_cells(text) > 15


def detect(ocr_texts: list[str]) -> dict[str, float | bool]:
    """
    Analyzes a sequence of OCR texts to determine if spreadsheet activity occurred
    and measures the progress based on data entry/manipulation volume.

    Args:
        ocr_texts: Chronological list of extracted text from frames.

    Returns:
        A dictionary containing the domain signal metrics:
        - is_active: True if spreadsheet activity was detected.
        - progress_score: A float (0.0 to 1.0) indicating the volume of data changes.
    """
    if len(ocr_texts) < 2:
        return {"is_active": False, "progress_score": 0.0}

    total_progress_signal = 0.0
    active_frames = 0

    for i in range(1, len(ocr_texts)):
        prev_text = ocr_texts[i-1]
        curr_text = ocr_texts[i]

        # If the current frame looks like Excel, Google Sheets, or a data dashboard
        if _is_spreadsheet_heavy(curr_text):
            active_frames += 1
            
            prev_cells = _count_data_cells(prev_text)
            curr_cells = _count_data_cells(curr_text)
            
            # Meaningful change in the number of visible data cells.
            # Threshold set to >2 to ignore slight OCR flickering on unchanged numbers.
            if abs(curr_cells - prev_cells) > 2:
                total_progress_signal += 1.0

    is_active = active_frames > 0
    
    # Normalize the score
    transitions = len(ocr_texts) - 1
    score = min(total_progress_signal / transitions, 1.0) if transitions > 0 else 0.0

    return {
        "is_active": is_active,
        "progress_score": score if is_active else 0.0
    }