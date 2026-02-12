import json
import time
from .utils import new_id, now_iso
from .judge import judge_answer
from .openai_safe import chat_create_safe, ModelCaps

SOLVER_SYSTEM = "Answer the user's question as accurately and clearly as possible."

def get_state(con, key: str, default: str) -> str:
    row = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default

def set_state(con, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value)
    )
    con.commit()

def update_ema(prev_ema: float, batch_mean: float, alpha: float) -> float:
    return alpha * batch_mean + (1.0 - alpha) * prev_ema

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))

def adapt_difficulty(prev_diff: int, prev_ema: float, new_ema: float, step: float = 0.02) -> int:
    # simple, explainable adaptive testing
    if new_ema > prev_ema + step:
        return _clamp_int(prev_diff + 1, 1, 5)
    if new_ema < prev_ema - step:
        return _clamp_int(prev_diff - 1, 1, 5)
    return _clamp_int(prev_diff, 1, 5)

# def sample_questions_weighted(con, n: int):
#     rows = con.execute("""
#         SELECT q.question_id, q.prompt,
#                COALESCE(cs.mean_score, 0.5) AS mean_score
#         FROM questions q
#         LEFT JOIN (
#           SELECT q2.category AS category, AVG(r.score) AS mean_score
#           FROM results r JOIN questions q2 ON q2.question_id=r.question_id
#           GROUP BY q2.category
#         ) cs ON cs.category = q.category
#         ORDER BY mean_score ASC, q.created_at DESC
#         LIMIT ?
#     """, (n,)).fetchall()
#     return [(r["question_id"], r["prompt"]) for r in rows]

from .evolve import CATEGORIES

def sample_questions_with_coverage(con, n: int, min_per_category: int = 1):
    """
    Exploration + exploitation sampler with a 'new-first' preference.

    - Exploration: ensure coverage across categories (when n allows).
      Within each category, prefer questions that are unevaluated, then least-evaluated,
      then most recently generated.

    - Exploitation: fill remaining slots by weaker categories (lower mean score),
      again preferring unevaluated / least-evaluated questions first.

    Notes on terminology:
    - "unevaluated" = no rows in results for that question_id (not the same as 'recently generated').
    """
    k = len(CATEGORIES)
    if n <= 0:
        return []

    # Compute category means (lower = weaker). Unseen/N.A. treated as 0.5
    rows = con.execute("""
        SELECT q.category, AVG(r.score) AS mean_score
        FROM questions q
        LEFT JOIN results r ON r.question_id = q.question_id
        GROUP BY q.category
    """).fetchall()

    mean = {c: 0.5 for c in CATEGORIES}
    for r in rows:
        if r["category"] in mean and r["mean_score"] is not None:
            mean[r["category"]] = float(r["mean_score"])

    cats_sorted_weak = sorted(CATEGORIES, key=lambda c: mean.get(c, 0.5))

    # Decide per-category quota
    per_cat = 0
    if n >= k:
        per_cat = min_per_category
        if per_cat * k > n:
            per_cat = max(1, n // k)

    selected = []
    selected_ids = set()

    # Helper query: for a given category, prefer unevaluated -> least evaluated -> newest generated
    def pick_from_category(cat: str, limit: int):
        return con.execute("""
            SELECT q.question_id, q.prompt
            FROM questions q
            LEFT JOIN (
                SELECT question_id, COUNT(*) AS cnt
                FROM results
                GROUP BY question_id
            ) rc ON rc.question_id = q.question_id
            WHERE q.category = ?
            ORDER BY
                CASE WHEN rc.cnt IS NULL THEN 0 ELSE 1 END ASC,  -- unevaluated first
                COALESCE(rc.cnt, 0) ASC,                         -- then least evaluated
                q.created_at DESC                                 -- then most recently generated
            LIMIT ?
        """, (cat, limit)).fetchall()

    # 1) Exploration: pick per_cat from each category (or weakest categories if n < k)
    cats_to_cover = CATEGORIES if per_cat > 0 else cats_sorted_weak[:min(n, k)]

    for c in cats_to_cover:
        rows = pick_from_category(c, per_cat if per_cat > 0 else 1)
        for r in rows:
            if r["question_id"] not in selected_ids:
                selected.append((r["question_id"], r["prompt"]))
                selected_ids.add(r["question_id"])
            if len(selected) >= n:
                return selected

    remaining = n - len(selected)
    if remaining <= 0:
        return selected

    # 2) Exploitation: fill remaining by weakness (lowest category mean first),
    # but still prefer unevaluated / least-evaluated within those categories.
    # We'll loop categories from weakest to strongest and pull candidates until filled.
    for c in cats_sorted_weak:
        if len(selected) >= n:
            break

        # Pull a bit extra per category to avoid collisions with already selected items
        rows = pick_from_category(c, limit=max(remaining, 3))

        for r in rows:
            if r["question_id"] in selected_ids:
                continue
            selected.append((r["question_id"], r["prompt"]))
            selected_ids.add(r["question_id"])
            if len(selected) >= n:
                break

    # If still not enough (e.g., sparse categories), fallback to global unevaluated/least-evaluated
    if len(selected) < n:
        rows = con.execute("""
            SELECT q.question_id, q.prompt
            FROM questions q
            LEFT JOIN (
                SELECT question_id, COUNT(*) AS cnt
                FROM results
                GROUP BY question_id
            ) rc ON rc.question_id = q.question_id
            ORDER BY
                CASE WHEN rc.cnt IS NULL THEN 0 ELSE 1 END ASC,
                COALESCE(rc.cnt, 0) ASC,
                q.created_at DESC
            LIMIT ?
        """, (n * 3,)).fetchall()

        for r in rows:
            if r["question_id"] in selected_ids:
                continue
            selected.append((r["question_id"], r["prompt"]))
            selected_ids.add(r["question_id"])
            if len(selected) >= n:
                break

    return selected


def run_benchmark(
    client,
    caps: ModelCaps,
    con,
    *,
    base_url: str,
    solve_model: str,
    judge_model: str,
    n: int,
    alpha: float = 0.2,
    rejudge_conf_threshold: float = 0.6
):
    run_id = new_id()

    prev_ema = float(get_state(con, "ema_value", "0.0"))
    prev_diff = int(get_state(con, "target_difficulty", "2"))

    con.execute(
        "INSERT INTO runs(run_id, run_at, base_url, solve_model, judge_model, n_questions) VALUES(?,?,?,?,?,?)",
        (run_id, now_iso(), base_url, solve_model, judge_model, int(n))
    )
    con.commit()

    # qs = sample_questions_weighted(con, n)
    k = len(CATEGORIES)
    min_per_category = max(1, int( 0.2 * n/len(CATEGORIES) ))
    qs = sample_questions_with_coverage(con, n, min_per_category=min_per_category)

    if not qs:
        raise RuntimeError("No questions in DB. Run `generate` first.")

    scores = []

    for qid, q in qs:
        # Solve
        t0 = time.time()
        resp = chat_create_safe(
            client, caps,
            model=solve_model,
            messages=[{"role": "system", "content": SOLVER_SYSTEM},
                      {"role": "user", "content": q}],
            temperature=1.0,
        )
        answer = resp.choices[0].message.content.strip()
        latency_ms = int((time.time() - t0) * 1000)

        # Judge (with uncertainty proxy)
        try:
            j1 = judge_answer(client, caps, model=judge_model, question=q, answer=answer)
        except Exception as e:
            j1 = {
                "score": 0.0, "pass": False,
                "reasons": [f"Judge failed: {type(e).__name__}"],
                "rubric_breakdown": {"correctness": 0.0, "completeness": 0.0, "clarity": 0.0},
                "confidence": 0.0,
            }

        score = float(j1.get("score", 0.0))
        conf = float(j1.get("confidence", 0.0))

        # Rejudge if low confidence (self-consistency)
        j2 = None
        disagreement = 0.0
        if conf < rejudge_conf_threshold:
            try:
                j2 = judge_answer(client, caps, model=judge_model, question=q, answer=answer)
                score2 = float(j2.get("score", score))
                disagreement = abs(score - score2)
                score = 0.5 * (score + score2)  # average
                conf = max(conf, float(j2.get("confidence", conf)))
            except Exception:
                pass

        j_out = dict(j1)
        j_out["rejudged"] = bool(j2 is not None)
        j_out["disagreement"] = float(disagreement)
        if j2 is not None:
            j_out["judge2"] = j2

        scores.append(score)

        con.execute(
            """
            INSERT INTO results(result_id, run_id, question_id, answer, judge_json, score, confidence, latency_ms, created_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (new_id(), run_id, qid, answer, json.dumps(j_out), float(score), float(conf), latency_ms, now_iso())
        )
        con.commit()

    batch_mean = sum(scores) / max(1, len(scores))
    ema = update_ema(prev_ema, batch_mean, alpha)

    # adaptive difficulty update
    new_diff = adapt_difficulty(prev_diff, prev_ema, ema)

    set_state(con, "ema_value", str(ema))
    set_state(con, "ema_alpha", str(alpha))
    set_state(con, "ema_last_run_id", run_id)
    set_state(con, "last_batch_mean", str(batch_mean))
    set_state(con, "target_difficulty", str(new_diff))

    # persist run-level summaries (columns added via migration)
    con.execute("UPDATE runs SET batch_mean=?, ema_after=?, target_difficulty=? WHERE run_id=?",
                (float(batch_mean), float(ema), int(new_diff), run_id))
    con.commit()

    return {"run_id": run_id, "batch_mean": batch_mean, "ema": ema, "n": len(scores), "target_difficulty": new_diff}
