"""
Persona configurations for dynamic benchmarking of hardware metrics.
"""

PERSONA_BENCHMARKS = {
    "backend_dev": {
        "role_name": "Deep-Focus Coder",
        "keystrokes": 80,
        "clicks": 30,
        "evaluation_rules": "AI-driven: Mostly reading code, short prompts, and Tab acceptances."
    },
    "ui_designer": {
        "role_name": "UI/UX & Frontend Dev",
        "keystrokes": 40,
        "clicks": 80,
        "evaluation_rules": "AI-driven: Selecting component variants and visual layout reviews."
    },
    "architect": {
        "role_name": "System Architect / Reviewer",
        "keystrokes": 20,
        "clicks": 15,
        "evaluation_rules": "AI-driven: Deep system reading, prompt architecture, low activity."
    },
    "qa_engineer": {
        "role_name": "QA / Test Specialist",
        "keystrokes": 30,
        "clicks": 40,
        "evaluation_rules": "AI-driven: Triggering test suites and scanning test charts."
    },
    "data_analyst": {
        "role_name": "Data Analyst / Modeler",
        "keystrokes": 50,
        "clicks": 45,
        "evaluation_rules": "AI-driven: Validating auto-generated analytics dashboards."
    }
}
