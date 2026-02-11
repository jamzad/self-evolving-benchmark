# src/run.py
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

def sample_questions_weighted(con, n: int):
    rows = con.execute("""
        SELECT q.question_id, q.prompt,
               COALESCE(cs.mean_score, 0.5) AS mean_score
        FROM questions q
        LEFT JOIN (
          SELECT q2.category AS category, AVG(r.score) AS mean_score
          FROM results r JOIN questions q2 ON q2.question_id=r.question_id
          GROUP BY q2.category
        ) cs ON cs.category = q.category
        ORDER BY mean_score ASC, q.created_at DESC
        LIMIT ?
    """, (n,)).fetchall()
    return [(r["question_id"], r["prompt"]) for r in rows]

def run_benchmark(
    client,
    caps: ModelCaps,
    con,
    *,
    base_url: str,
    solve_model: str,
    judge_model: str,
    n: int,
    alpha: float = 0.2
):
    run_id = new_id()
    con.execute(
        "INSERT INTO runs(run_id, run_at, base_url, solve_model, judge_model, n_questions) VALUES(?,?,?,?,?,?)",
        (run_id, now_iso(), base_url, solve_model, judge_model, int(n))
    )
    con.commit()

    qs = sample_questions_weighted(con, n)
    if not qs:
        raise RuntimeError("No questions in DB. Run `generate` first.")

    scores = []

    for qid, q in qs:
        t0 = time.time()
        resp = chat_create_safe(
            client, caps,
            model=solve_model,
            messages=[
                {"role": "system", "content": SOLVER_SYSTEM},
                {"role": "user", "content": q},
            ],
            temperature=1.0,  # omitted if unsupported
        )
        answer = resp.choices[0].message.content.strip()
        latency_ms = int((time.time() - t0) * 1000)

        j = judge_answer(client, caps, model=judge_model, question=q, answer=answer)
        score = float(j["score"])
        conf = float(j.get("confidence", 0.0))
        scores.append(score)

        con.execute(
            """
            INSERT INTO results(result_id, run_id, question_id, answer, judge_json, score, confidence, latency_ms, created_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (new_id(), run_id, qid, answer, json.dumps(j), score, conf, latency_ms, now_iso())
        )
        con.commit()

    batch_mean = sum(scores) / max(1, len(scores))
    prev_ema = float(get_state(con, "ema_value", "0.0"))
    ema = update_ema(prev_ema, batch_mean, alpha)

    set_state(con, "ema_value", str(ema))
    set_state(con, "ema_alpha", str(alpha))
    set_state(con, "ema_last_run_id", run_id)
    set_state(con, "last_batch_mean", str(batch_mean))

    return {"run_id": run_id, "batch_mean": batch_mean, "ema": ema, "n": len(scores)}

