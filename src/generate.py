# src/generate.py
import json
from typing import List, Dict, Optional
from .utils import new_id, now_iso, sha256_text
from .openai_safe import chat_create_safe, ModelCaps
from .evolve import CATEGORIES, category_means, category_weights, sample_categories

GEN_SYSTEM = "You generate novel benchmark questions for evaluating LLMs."

GEN_USER_TEMPLATE = """
Generate {n} novel questions for an LLM benchmark.

Return ONLY valid JSON: a list of objects, each with keys:
- category: one of {categories}
- difficulty: integer 1..5
- prompt: string

Hard constraints:
- Every question must be novel vs the prior list below (no repeats, no paraphrases).
- Avoid using the same template with different numbers/entities.
- Keep prompts self-contained and answerable without external tools.

Requested category mix (approximate): {requested_mix}

Prior questions to avoid:
{prior_prompts}
"""

def fetch_prior_prompts(con, limit: int = 200) -> List[str]:
    rows = con.execute(
        "SELECT prompt FROM questions ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [r["prompt"] for r in rows]

def _requested_mix_from_history(con, n: int) -> str:
    means = category_means(con)
    weights = category_weights(means, CATEGORIES)
    sampled = sample_categories(weights, n)
    # Make a readable target mix string
    counts = {c: 0 for c in CATEGORIES}
    for s in sampled:
        counts[s] += 1
    parts = [f"{c}:{counts[c]}" for c in CATEGORIES if counts[c] > 0]
    return ", ".join(parts) if parts else "balanced"

def generate_questions(
    client,
    caps: ModelCaps,
    con,
    model: str,
    n: int,
    domain: str = "general",
    max_attempts: int = 6
) -> List[Dict]:
    inserted: List[Dict] = []
    attempts = 0

    while len(inserted) < n and attempts < max_attempts:
        attempts += 1
        need = n - len(inserted)

        prior = fetch_prior_prompts(con, limit=200)
        prior_block = "\n".join([f"- {p}" for p in prior]) if prior else "(none)"

        requested_mix = _requested_mix_from_history(con, need)

        user_prompt = GEN_USER_TEMPLATE.format(
            n=need,
            categories=CATEGORIES,
            requested_mix=requested_mix,
            prior_prompts=prior_block
        )

        resp = chat_create_safe(
            client, caps,
            model=model,
            messages=[
                {"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            # if model supports temp, allow diversity; otherwise auto-omitted
            temperature=1.0,
        )

        raw = resp.choices[0].message.content
        items = json.loads(raw)

        for it in items:
            prompt = it["prompt"].strip()
            h = sha256_text(prompt)

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

    if len(inserted) < n:
        raise RuntimeError(f"Only generated {len(inserted)}/{n} novel questions after {max_attempts} attempts.")

    return inserted
