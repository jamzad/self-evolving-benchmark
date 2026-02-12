import json
from .evolve import CATEGORIES

def analyze(con) -> str:
    lines = []

    # -----------------------------
    # Run history
    # -----------------------------
    runs = con.execute("""
        SELECT run_at, n_questions, batch_mean, ema_after, target_difficulty
        FROM runs
        ORDER BY run_at ASC
    """).fetchall()

    lines.append("Run history (time | n | batch_mean | ema | target_difficulty):")
    if runs:
        for r in runs[-10:]:
            lines.append(
                f"  {r['run_at']} | n={r['n_questions']} | "
                f"mean={r['batch_mean']:.3f} | ema={r['ema_after']:.3f} | "
                f"d={r['target_difficulty']}"
            )
    else:
        lines.append("  (no runs yet)")

    # -----------------------------
    # Category means
    # -----------------------------
    lines.append("")
    lines.append("Category mean scores:")

    cat_means = con.execute("""
        SELECT q.category, AVG(r.score) AS mean_score, COUNT(*) AS n
        FROM results r
        JOIN questions q ON q.question_id = r.question_id
        GROUP BY q.category
    """).fetchall()

    cat_dict = {row["category"]: (row["mean_score"], row["n"]) for row in cat_means}

    for c in CATEGORIES:
        if c in cat_dict:
            mean, n = cat_dict[c]
            lines.append(f"  - {c}: {mean:.3f} (n={n})")
        else:
            lines.append(f"  - {c}: N/A")

    # -----------------------------
    # Category × Difficulty matrix
    # -----------------------------
    lines.append("")
    lines.append("Category × Difficulty matrix (mean score (count)):")

    matrix = con.execute("""
        SELECT q.category, q.difficulty,
               AVG(r.score) AS mean_score,
               COUNT(*) AS n
        FROM results r
        JOIN questions q ON q.question_id = r.question_id
        GROUP BY q.category, q.difficulty
        ORDER BY q.category, q.difficulty
    """).fetchall()

    # organize
    grid = {}
    for row in matrix:
        cat = row["category"]
        diff = row["difficulty"]
        mean = row["mean_score"]
        n = row["n"]
        grid.setdefault(cat, {})[diff] = (mean, n)

    diffs = [1, 2, 3, 4, 5]

    header = "               " + "  ".join([f"d={d}" for d in diffs])
    lines.append(header)

    for c in CATEGORIES:
        row_str = f"{c:<15}"
        for d in diffs:
            if c in grid and d in grid[c]:
                mean, n = grid[c][d]
                row_str += f"{mean:.2f}({n})  "
            else:
                row_str += "   -      "
        lines.append(row_str)

    # -----------------------------
    # Worst failures
    # -----------------------------
    rows = con.execute("""
        SELECT q.prompt, r.score, r.confidence, r.judge_json
        FROM results r
        JOIN questions q ON q.question_id = r.question_id
        ORDER BY r.score ASC
        LIMIT 5
    """).fetchall()

    if rows:
        lines.append("")
        lines.append("Worst examples:")
        for row in rows:
            try:
                j = json.loads(row["judge_json"])
                dis = float(j.get("disagreement", 0.0))
                rs = j.get("reasons", [])
                reason = rs[0] if isinstance(rs, list) and rs else "(no reason)"
            except Exception:
                dis = 0.0
                reason = "(parse error)"

            lines.append(
                f"  score={row['score']:.3f} | "
                f"conf={row['confidence']:.2f} | "
                f"dis={dis:.3f} | "
                f"{reason}"
            )

    # -----------------------------
    # Uncertainty summary
    # -----------------------------
    u = con.execute("""
        SELECT AVG(ABS(json_extract(judge_json, '$.disagreement'))) AS avg_dis
        FROM results
    """).fetchone()

    if u and u["avg_dis"] is not None:
        lines.append("")
        lines.append(
            f"Uncertainty proxy: avg judge disagreement = {u['avg_dis']:.4f}"
        )

    return "\n".join(lines)
