"""
Free OpenRouter Multi-Model Benchmark
======================================
Loads screenshots from `../split_screenshots/`, sends them to each free
OpenRouter model via our VLMOpenAI client, and prints a side-by-side
comparison table of scores, confidence, idle detection, and task summaries.

Usage:
    cd ai-engine
    python benchmark_free_models.py

Requirements:
    - OPENROUTER_API_KEY must be set in .env (or as an environment variable)
    - Screenshots must exist in ../split_screenshots/
"""

import os
import sys
import glob
import time
import json
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the ai-engine package root is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models.vlm_openai import VLMOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("benchmark")

# ═══════════════════════════════════════════════════════════════════════════
#  FREE MODEL DICTIONARY
#  Each entry: { display_name: openrouter_model_id }
#  All models below are free-tier on OpenRouter (as of July 2025).
# ═══════════════════════════════════════════════════════════════════════════
FREE_MODELS: dict[str, str] = {
    # --- Google ---
    "Gemini 2.5 Pro (Free)":      "google/gemini-2.5-pro-exp-03-25:free",
    "Gemini 2.5 Flash (Free)":    "google/gemini-2.5-flash-preview-05-20:free",
    # --- Meta LLaMA ---
    "LLaMA 4 Maverick (Free)":    "meta-llama/llama-4-maverick:free",
    "LLaMA 4 Scout (Free)":       "meta-llama/llama-4-scout:free",
    # --- Anthropic Claude ---
    "Claude 3.5 Sonnet (Free)":   "anthropic/claude-3.5-sonnet:free",
    "Claude 3 Haiku (Free)":      "anthropic/claude-3-haiku:free",
    # --- Qwen ---
    "Qwen2.5 VL 72B (Free)":     "qwen/qwen2.5-vl-72b-instruct:free",
    "Qwen2.5 VL 32B (Free)":     "qwen/qwen2.5-vl-32b-instruct:free",
    # --- Mistral ---
    "Mistral Small 3.1 (Free)":   "mistralai/mistral-small-3.1-24b-instruct:free",
    # --- DeepSeek ---
    "DeepSeek V3 (Free)":         "deepseek/deepseek-chat-v3-0324:free",
    # --- OpenRouter Auto ---
    "OpenRouter Auto (Free)":     "openrouter/free",
}

# ═══════════════════════════════════════════════════════════════════════════
#  Load test screenshots
# ═══════════════════════════════════════════════════════════════════════════
SCREENSHOTS_DIR = SCRIPT_DIR.parent / "split_screenshots"


def load_screenshots(max_frames: int = 10) -> list[bytes]:
    """Read up to `max_frames` PNG/JPG files from the screenshots directory."""
    patterns = [str(SCREENSHOTS_DIR / "*.png"), str(SCREENSHOTS_DIR / "*.jpg")]
    files: list[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))

    if not files:
        logger.error(f"No screenshots found in {SCREENSHOTS_DIR}")
        sys.exit(1)

    files = files[:max_frames]
    images: list[bytes] = []
    for f in files:
        with open(f, "rb") as fh:
            images.append(fh.read())
    logger.info(f"Loaded {len(images)} screenshots from {SCREENSHOTS_DIR}")
    return images


# ═══════════════════════════════════════════════════════════════════════════
#  Run benchmark
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark() -> None:
    """Instantiate a single VLMOpenAI client and test each free model."""

    images = load_screenshots()

    # Minimal context payload (benchmark doesn't run the full pipeline)
    context = {
        "batch_size": len(images),
        "frame_data": [
            {"frame_index": i, "metrics": {"is_trivial_change": False, "is_idle": False}}
            for i in range(len(images))
        ],
    }

    # Create the client once — it will use the OpenRouter key from .env
    client = VLMOpenAI()

    results: list[dict] = []

    for display_name, model_id in FREE_MODELS.items():
        print(f"\n{'─' * 60}")
        print(f"  ▶  Testing: {display_name}")
        print(f"     Model ID: {model_id}")
        print(f"{'─' * 60}")

        start = time.time()
        try:
            result = client.analyze_batch(
                images=images,
                context=context,
                model_override=model_id,
            )
            elapsed = time.time() - start

            entry = {
                "model": display_name,
                "model_id": model_id,
                "score": result.score,
                "confidence": result.confidence,
                "progressed": result.progressed,
                "idle_detected": result.raw_response.get("idle_detected", False),
                "task": result.task_description[:80],
                "activity_summary": result.activity_summary,
                "time_sec": round(elapsed, 1),
                "status": "✅ OK",
            }
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(f"Model {model_id} failed: {exc}")
            entry = {
                "model": display_name,
                "model_id": model_id,
                "score": None,
                "confidence": None,
                "progressed": None,
                "idle_detected": None,
                "task": f"ERROR: {str(exc)[:60]}",
                "activity_summary": [],
                "time_sec": round(elapsed, 1),
                "status": "❌ FAIL",
            }

        results.append(entry)

    # ───────────────────────────────────────────────────────────────────
    #  Print comparison table
    # ───────────────────────────────────────────────────────────────────
    print("\n")
    print("═" * 110)
    print("  FREE MODEL BENCHMARK RESULTS")
    print("═" * 110)

    header = (
        f"{'Model':<28} │ {'Status':<8} │ {'Score':>6} │ {'Conf':>5} │ "
        f"{'Progressed':>10} │ {'Idle':>5} │ {'Time':>6} │ Task Description"
    )
    print(header)
    print("─" * 110)

    for r in results:
        score_str = f"{r['score']:.2f}" if r["score"] is not None else "  N/A"
        conf_str = f"{r['confidence']:.2f}" if r["confidence"] is not None else " N/A"
        prog_str = str(r["progressed"]) if r["progressed"] is not None else "N/A"
        idle_str = str(r["idle_detected"]) if r["idle_detected"] is not None else "N/A"

        row = (
            f"{r['model']:<28} │ {r['status']:<8} │ {score_str:>6} │ {conf_str:>5} │ "
            f"{prog_str:>10} │ {idle_str:>5} │ {r['time_sec']:>5.1f}s │ {r['task']}"
        )
        print(row)

    print("═" * 110)

    # ───────────────────────────────────────────────────────────────────
    #  Print activity summaries per model
    # ───────────────────────────────────────────────────────────────────
    print("\n")
    print("═" * 80)
    print("  ACTIVITY SUMMARIES (AI-inferred user behavior)")
    print("═" * 80)

    for r in results:
        print(f"\n  🧠 {r['model']}:")
        if r["activity_summary"]:
            for bullet in r["activity_summary"]:
                print(f"     • {bullet}")
        else:
            print("     (none returned)")

    print("\n" + "═" * 80)

    # ───────────────────────────────────────────────────────────────────
    #  Dump full JSON for deeper inspection
    # ───────────────────────────────────────────────────────────────────
    output_path = SCRIPT_DIR / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\n  📄 Full results saved to: {output_path}\n")


if __name__ == "__main__":
    run_benchmark()
