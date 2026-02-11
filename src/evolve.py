from typing import Dict, List
import random

CATEGORIES = ["reasoning", "math", "logic", "factual", "instruction_following"]

def category_means(con) -> Dict[str, float]:
    rows = con.execute("""
        SELECT q.category AS category, AVG(r.score) AS mean_score, COUNT(*) AS n
        FROM results r
        JOIN questions q ON q.question_id = r.question_id
        GROUP BY q.category
    """).fetchall()

    means = {}
    for row in rows:
        means[row["category"]] = float(row["mean_score"])
    return means

def category_weights(means: Dict[str, float], categories: List[str] = CATEGORIES) -> Dict[str, float]:
    # Higher weight for weaker categories
    w = {}
    for c in categories:
        m = means.get(c, 0.5)  # unseen categories treated as medium
        w[c] = max(0.05, 1.0 - m)
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}

def sample_categories(weights: Dict[str, float], n: int) -> List[str]:
    cats = list(weights.keys())
    probs = [weights[c] for c in cats]
    return random.choices(cats, weights=probs, k=n)

def format_weights(means: Dict[str, float]) -> str:
    w = category_weights(means, CATEGORIES)
    parts = []
    for c in CATEGORIES:
        m = means.get(c, None)
        parts.append(f"{c}: mean={m if m is not None else 'N/A'} weight={w[c]:.2f}")
    return " | ".join(parts)
