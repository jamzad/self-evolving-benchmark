# src/judge.py
import json

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

Scoring rules:
- Use intermediate values (e.g., 0.2, 0.7) when partially correct.
- "score" should be the average of the three rubric fields.
- "pass" is true if score >= 0.7.

Question: {question}
Answer: {answer}
"""

def judge_answer(client, model: str, question: str, answer: str) -> dict:
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        # temperature=0.0,
    )
    txt = resp.choices[0].message.content
    data = json.loads(txt)

    # minimal sanity clamps
    score = float(data.get("score", 0.0))
    conf = float(data.get("confidence", 0.0))
    data["score"] = max(0.0, min(1.0, score))
    data["confidence"] = max(0.0, min(1.0, conf))
    return data
