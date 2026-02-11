# src/generate.py
import json
from typing import List, Dict
from .utils import new_id, now_iso, sha256_text

GEN_SYSTEM = "You generate novel benchmark questions for evaluating LLMs."
GEN_USER_TEMPLATE = """
Generate {n} novel questions for an LLM benchmark.

Constraints:
- Every question must be meaningfully different from prior questions.
- Return ONLY valid JSON: a list of objects, each with keys:
  - category: one of ["reasoning","math","logic","factual","instruction_following"]
  - difficulty: integer 1..5
  - prompt: string

Prior questions to avoid (do not repeat or paraphrase):
{prior_prompts}
"""

def fetch_prior_prompts(con, limit: int = 200) -> List[str]:
    rows = con.execute(
        "SELECT prompt FROM questions ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [r["prompt"] for r in rows]

def generate_questions(client, con, model: str, n: int, domain: str = "general") -> List[Dict]:
    prior = fetch_prior_prompts(con, limit=200)
    prior_block = "\n".join([f"- {p}" for p in prior]) if prior else "(none)"

    user_prompt = GEN_USER_TEMPLATE.format(n=n, prior_prompts=prior_block)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0.7,
    )
    raw = resp.choices[0].message.content
    items = json.loads(raw)

    inserted = []
    for it in items:
        prompt = it["prompt"].strip()
        h = sha256_text(prompt)

        # novelty check: hash must be new
        exists = con.execute("SELECT 1 FROM questions WHERE prompt_hash=?", (h,)).fetchone()
        if exists:
            continue

        qid = new_id()
        con.execute(
            """
            INSERT INTO questions(question_id, created_at, domain, category, difficulty, prompt, prompt_hash)
            VALUES(?,?,?,?,?,?,?)
            """,
            (qid, now_iso(), domain, it["category"], int(it["difficulty"]), prompt, h)
        )
        inserted.append({"question_id": qid, **it})

    con.commit()
    return inserted
