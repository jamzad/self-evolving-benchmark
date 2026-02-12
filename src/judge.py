import json
from .openai_safe import chat_create_safe, ModelCaps

JUDGE_SYSTEM = "You are a strict grader. Output JSON only."

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
- score MUST equal the average of correctness, completeness, clarity.
- pass is true if score >= 0.7.
- confidence is your confidence in your grading (0..1).
- Output JSON ONLY (no markdown, no extra text).

Question: {question}
Answer: {answer}
"""

FIX_SYSTEM = "You repair invalid JSON. Output JSON only."
FIX_TEMPLATE = """
Fix the following text into valid JSON matching exactly this schema:

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
- Ensure numbers are floats in [0,1] (except pass boolean).
- reasons must be a list of strings.

Text:
{text}
"""

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()

def _clamp01(x) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))

def _normalize(d: dict) -> dict:
    rb = d.get("rubric_breakdown") or {}
    c1 = _clamp01(rb.get("correctness", 0.0))
    c2 = _clamp01(rb.get("completeness", 0.0))
    c3 = _clamp01(rb.get("clarity", 0.0))
    score = _clamp01(d.get("score", (c1 + c2 + c3) / 3.0))
    conf = _clamp01(d.get("confidence", 0.0))
    reasons = d.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons][:5]
    return {
        "score": score,
        "pass": bool(d.get("pass", score >= 0.7)),
        "reasons": reasons,
        "rubric_breakdown": {"correctness": c1, "completeness": c2, "clarity": c3},
        "confidence": conf,
    }

def judge_answer(client, caps: ModelCaps, model: str, question: str, answer: str) -> dict:
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)
    resp = chat_create_safe(
        client, caps,
        model=model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = _strip_fences(resp.choices[0].message.content)

    try:
        return _normalize(json.loads(raw))
    except json.JSONDecodeError:
        fix_prompt = FIX_TEMPLATE.format(text=raw)
        fix = chat_create_safe(
            client, caps,
            model=model,
            messages=[{"role": "system", "content": FIX_SYSTEM},
                      {"role": "user", "content": fix_prompt}],
            temperature=0.0,
        )
        fixed = _strip_fences(fix.choices[0].message.content)
        return _normalize(json.loads(fixed))
