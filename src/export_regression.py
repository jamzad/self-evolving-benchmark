import json

def export_regression(con, out_path: str, k: int = 20) -> int:
    rows = con.execute("""
        SELECT q.question_id, q.category, q.difficulty, q.prompt, MIN(r.score) AS worst_score
        FROM questions q
        JOIN results r ON r.question_id=q.question_id
        GROUP BY q.question_id
        ORDER BY worst_score ASC
        LIMIT ?
    """, (k,)).fetchall()

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    return len(rows)
