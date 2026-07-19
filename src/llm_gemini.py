"""
Gemini LLM narrator for the Self-XAI explanation.

Faithfulness contract (from the reference notebook, Cell 27/28):
  The set of detected lesion concepts is FIXED by the CBM. The LLM never adds,
  removes, or second-guesses which lesions are present; it only explains the
  fired concepts using deterministically retrieved knowledge-base excerpts.

API key handling (safe for public GitHub):
  1. Environment variable  GEMINI_API_KEY   (preferred)
  2. Local file            gemini_config.json  (git-ignored, never committed)
  The key is NEVER hard-coded in source. If neither is present, the app falls
  back to the deterministic template narrative (offline, no network).
"""

import json
import os

import src.xai

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "gemini_config.json")

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def load_config():
    """Return (api_key, model). Env var wins; then local git-ignored file."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    if not api_key and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            api_key = str(cfg.get("api_key", "")).strip()
            model = str(cfg.get("model", "")).strip() or model
        except Exception:
            pass
    return api_key, model


def is_configured():
    key, _ = load_config()
    return bool(key)


SYSTEM_MSG = (
    "You are a clinical explainability assistant for a diabetic retinopathy "
    "concept-bottleneck model (CBM). Be FAITHFUL: the set of detected lesion "
    "concepts is fixed by the model - never add, remove, or second-guess which "
    "lesions are present. Use ONLY the provided knowledge-base excerpts for "
    "clinical facts. Write in English, concise and clinically precise."
)


def _build_user_msg(result, kb, ctx):
    grade = result["grade"]
    return f"""Model outputs for the loaded fundus image:
- Detected lesion concepts (fired, with confidence): {ctx['fired_str']}
- Predicted ICDR grade: {grade} - {result['grade_name']} (softmax {result['grade_probability']:.2f})
- ICDR definition of the predicted grade: {ctx['grade_desc']}

Knowledge-base excerpts retrieved for the fired concepts:
{ctx['kb_context']}

Available full citations (use these for the Sources section):
{ctx['refs_context']}

Write an explainable narrative with exactly these sections:
1. Findings: for each detected concept, one sentence on what it is and how it looks.
2. Concept -> grade reasoning: explain WHY the predicted ICDR grade follows from the
   detected concepts, referencing the grading rules (e.g. the 4-2-1 rule where relevant).
   If no concept was detected, explain that the absence of lesions supports the grade.
3. Sources: list the cited sources.
Do not introduce any lesion that is not in the detected list."""


def generate_narrative(result, kb):
    """Generate the narrative. Returns (text, source).

    On any failure (no key, no network, bad model, quota) returns the
    deterministic template narrative with a source string explaining the reason.
    """
    ctx = xai.build_rag_context(result, kb)
    api_key, model = load_config()

    if not api_key:
        return xai.template_narrative_text(result, kb), "template KB (no API key)"

    try:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return (
                xai.template_narrative_text(result, kb),
                "template KB (google-genai not installed)",
            )

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=_build_user_msg(result, kb, ctx),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_MSG,
                temperature=0.2,
            ),
        )
        text = (resp.text or "").strip()
        if not text:
            return xai.template_narrative_text(result, kb), "template KB (empty LLM reply)"
        return text, f"Gemini ({model})"
    except Exception as e:
        return (
            xai.template_narrative_text(result, kb),
            f"template KB (LLM failed: {type(e).__name__})",
        )
