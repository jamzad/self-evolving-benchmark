from .evolve import category_means, CATEGORIES

def report(con) -> str:
    q_count = con.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
    r_count = con.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
    last_mean = con.execute("SELECT value FROM state WHERE key='last_batch_mean'").fetchone()
    ema = con.execute("SELECT value FROM state WHERE key='ema_value'").fetchone()

    means = category_means(con)

    lines = []
    lines.append(f"Questions: {q_count} | Results: {r_count}")
    lines.append(f"Last batch mean: {float(last_mean['value']) if last_mean else None}")
    lines.append(f"EMA: {float(ema['value']) if ema else None}")
    lines.append("")
    lines.append("Category mean scores:")
    for c in CATEGORIES:
        val = means.get(c, None)
        lines.append(f"  - {c}: {val if val is not None else 'N/A'}")

    # show 3 worst failures
    rows = con.execute("""
        SELECT q.prompt, r.score
        FROM results r JOIN questions q ON q.question_id=r.question_id
        ORDER BY r.score ASC
        LIMIT 3
    """).fetchall()
    if rows:
        lines.append("")
        lines.append("Worst 3 examples:")
        for i, row in enumerate(rows, 1):
            lines.append(f"  {i}) score={row['score']:.3f} | {row['prompt'][:120]}...")

    return "\n".join(lines)
