# src/judge.py
import json
import re
from .openai_safe import chat_create_safe, ModelCaps

JUDGE_SYSTEM = "You are a strict JSON-only grader."
JUDGE_TEMPLATE = """
Grade the answer to the question using this rubric (partial credit allowed):
- correctness (0..1)
- completeness (0..1)
- clarity (0..1)

Return ONLY valid JSON exactly in this schema:
{{
  "score": 0.0,
  "pass": false,
  "reasons": ["..."],
  "rubric_breakdown": {{
    "correctness": 0.0,
    "completeness": 0.0,
    "clarity": 0.0
  }},
  "confidence": 0.0
}}

Rules:
- Use intermediate values (e.g., 0.2, 0.7) when partially correct.
- "score" must equal the average of correctness, completeness, clarity.
- "pass" is true if score >= 0.7.
- "confidence" is your confidence in your grading (0..1).
- Output JSON ONLY (no markdown, no extra text).

Question: {question}
Answer: {answer}
"""

FIX_JSON_SYSTEM = "You fix JSON. Output JSON only."
FIX_JSON_TEMPLATE = """
The following text was intended to be valid JSON, but it is invalid.
Fix it so it becomes valid JSON matching this schema exactly:

{{
  "score": 0.0,
  "pass": false,
  "reasons": ["..."],
  "rubric_breakdown": {{
    "correctness": 0.0,
    "completeness": 0.0,
    "clarity": 0.0
  }},
  "confidence": 0.0
}}

Rules:
- Output JSON ONLY.
- Do not add extra keys.
- Ensure numbers are floats in [0,1].
- Ensure reasons is a list of strings.

Invalid text:
{text}
"""

def _extract_json_object(text: str) -> str:
    """Best-effort extraction of the first JSON object from a messy output."""
    t = text.strip()
    # Remove code fences if any
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end+1]
    return t

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _normalize(data: dict) -> dict:
    rb = data.get("rubric_breakdown", {}) or {}
    rb = {
        "correctness": _clamp01(rb.get("correctness", 0.0)),
        "completeness": _clamp01(rb.get("completeness", 0.0)),
        "clarity": _clamp01(rb.get("clarity", 0.0)),
    }
    score = data.get("score", (rb["correctness"] + rb["completeness"] + rb["clarity"]) / 3.0)
    score = _clamp01(score)
    conf = _clamp01(data.get("confidence", 0.0))

    reasons = data.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons][:5]

    return {
        "score": score,
        "pass": bool(data.get("pass", score >= 0.7)),
        "reasons": reasons,
        "rubric_breakdown": rb,
        "confidence": conf,
    }

def _repair_json_once(client, caps: ModelCaps, model: str, raw_text: str) -> dict:
    """Ask the model to fix invalid JSON once, then parse."""
    fix_prompt = FIX_JSON_TEMPLATE.format(text=raw_text)
    resp = chat_create_safe(
        client, caps,
        model=model,
        messages=[
            {"role": "system", "content": FIX_JSON_SYSTEM},
            {"role": "user", "content": fix_prompt},
        ],
        temperature=0.0,
    )
    fixed = resp.choices[0].message.content
    fixed_obj = _extract_json_object(fixed)
    return json.loads(fixed_obj)

def judge_answer(client, caps: ModelCaps, model: str, question: str, answer: str) -> dict:
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)

    resp = chat_create_safe(
        client, caps,
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # omitted if unsupported
    )

    raw = resp.choices[0].message.content
    obj = _extract_json_object(raw)

    try:
        data = json.loads(obj)
    except json.JSONDecodeError:
        data = _repair_json_once(client, caps, model, raw)

    return _normalize(data)
