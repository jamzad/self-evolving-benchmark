import json
from typing import List, Dict
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
- Avoid the same template with different numbers/entities.
- Keep prompts self-contained.

Difficulty scale (must follow these anchors):
- 1: single-step, obvious, minimal reasoning (but still non-trivia).
- 2: two steps or light reasoning.
- 3: multi-step reasoning; needs careful thinking; no hidden tricks.
- 4: multi-step + constraints/edge cases/traps; requires structured reasoning.
- 5: hardest: multi-step + strict constraints + plausible pitfalls; cannot be solved by a single fact or a single arithmetic step; still self-contained and judgeable.

Target difficulty: {target_difficulty}/5

Difficulty distribution rule (independent of n):
- At least 60% of the questions must have difficulty >= {target_difficulty}.
- The remaining questions can be any difficulty 1..5 to preserve diversity.

Requested category mix (approximate): {requested_mix}

Common failure themes to target (optional inspiration):
{failure_themes}

Prior questions to avoid:
{prior_prompts}
"""

def _get_state(con, key: str, default: str) -> str:
    row = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default

def fetch_prior_prompts(con, limit: int = 200) -> List[str]:
    rows = con.execute("SELECT prompt FROM questions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [r["prompt"] for r in rows]

def fetch_failure_themes(con, k: int = 8) -> List[str]:
    # Pull worst results and extract first reason lines
    rows = con.execute("""
        SELECT r.judge_json
        FROM results r
        ORDER BY r.score ASC
        LIMIT ?
    """, (k,)).fetchall()

    themes = []
    for row in rows:
        try:
            j = json.loads(row["judge_json"])
            rs = j.get("reasons", [])
            if isinstance(rs, list) and rs:
                themes.append(str(rs[0]))
        except Exception:
            continue

    # de-dup lightly
    out = []
    seen = set()
    for t in themes:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out[:k]

def _requested_mix_from_history(con, n: int) -> str:
    means = category_means(con)
    weights = category_weights(means, CATEGORIES)
    sampled = sample_categories(weights, n)
    counts = {c: 0 for c in CATEGORIES}
    for s in sampled:
        counts[s] += 1
    parts = [f"{c}:{counts[c]}" for c in CATEGORIES if counts[c] > 0]
    return ", ".join(parts) if parts else "balanced"

def generate_questions(client, caps: ModelCaps, con, model: str, n: int, domain: str = "general", max_attempts: int = 6) -> List[Dict]:
    inserted: List[Dict] = []
    attempts = 0

    target_difficulty = int(_get_state(con, "target_difficulty", "2"))
    requested_mix = _requested_mix_from_history(con, n)
    failures = fetch_failure_themes(con, k=8)
    failure_block = "\n".join([f"- {t}" for t in failures]) if failures else "(none yet)"

    while len(inserted) < n and attempts < max_attempts:
        attempts += 1
        need = n - len(inserted)

        prior = fetch_prior_prompts(con, limit=200)
        prior_block = "\n".join([f"- {p}" for p in prior]) if prior else "(none)"

        user_prompt = GEN_USER_TEMPLATE.format(
            n=need,
            categories=CATEGORIES,
            target_difficulty=target_difficulty,
            requested_mix=requested_mix,
            failure_themes=failure_block,
            prior_prompts=prior_block
        )

        resp = chat_create_safe(
            client, caps,
            model=model,
            messages=[{"role": "system", "content": GEN_SYSTEM},
                      {"role": "user", "content": user_prompt}],
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
                """INSERT INTO questions(question_id, created_at, domain, category, difficulty, prompt, prompt_hash)
                   VALUES(?,?,?,?,?,?,?)""",
                (qid, now_iso(), domain, it["category"], int(it["difficulty"]), prompt, h)
            )
            inserted.append({"question_id": qid, **it})

        con.commit()

    if len(inserted) < n:
        raise RuntimeError(f"Only generated {len(inserted)}/{n} novel questions after {max_attempts} attempts.")

    return inserted
