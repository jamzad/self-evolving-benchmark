# src/judge.py
import json
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

Question: {question}
Answer: {answer}
"""

def judge_answer(client, caps: ModelCaps, model: str, question: str, answer: str) -> dict:
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)

    resp = chat_create_safe(
        client, caps,
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # if unsupported, wrapper omits it
    )

    txt = resp.choices[0].message.content
    data = json.loads(txt)

    # Clamp key values
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    data["confidence"] = clamp01(data.get("confidence", 0.0))
    data["score"] = clamp01(data.get("score", 0.0))
    return data
