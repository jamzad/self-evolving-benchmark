# src/plots.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from .evolve import CATEGORIES, category_means, category_weights


def _ensure_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_evolution(con, out_dir: str = "docs/figs") -> str:
    """
    Plot batch mean vs EMA vs target difficulty over run history.

    Uses: runs(batch_mean, ema_after, target_difficulty, run_at)
    """
    outp = _ensure_dir(out_dir) / "evolution.png"

    rows = con.execute("""
        SELECT run_at, batch_mean, ema_after, target_difficulty
        FROM runs
        WHERE batch_mean IS NOT NULL AND ema_after IS NOT NULL
        ORDER BY run_at ASC
    """).fetchall()

    if not rows:
        raise RuntimeError("No run history found (need at least one `run`).")

    xs = list(range(1, len(rows) + 1))
    batch = [float(r["batch_mean"]) for r in rows]
    ema = [float(r["ema_after"]) for r in rows]
    diff = [int(r["target_difficulty"]) if r["target_difficulty"] is not None else None for r in rows]

    plt.figure()
    plt.plot(xs, batch, label="Batch mean")
    plt.plot(xs, ema, label="EMA")

    # Overlay difficulty on secondary axis (clean + informative)
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(xs, diff, linestyle="--", label="Target difficulty")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Score")
    ax2.set_ylabel("Target difficulty (1–5)")
    ax.set_title("Benchmark evolution: performance and adaptive difficulty")

    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    return str(outp)


def plot_category_pressure(con, out_dir: str = "docs/figs") -> str:
    """
    Bar chart showing category mean score vs evolution weight (pressure).
    Uses: category_means() and category_weights() (already in evolve.py)
    """
    outp = _ensure_dir(out_dir) / "category_pressure.png"

    means = category_means(con)  # {cat: mean_score}
    weights = category_weights(means, CATEGORIES)  # {cat: weight}

    cats = CATEGORIES[:]
    mean_vals = [float(means.get(c, 0.0)) if means.get(c) is not None else 0.0 for c in cats]
    w_vals = [float(weights.get(c, 0.0)) for c in cats]

    # Side-by-side bars
    x = list(range(len(cats)))
    width = 0.4

    plt.figure()
    plt.bar([i - width/2 for i in x], mean_vals, width=width, label="Mean score")
    plt.bar([i + width/2 for i in x], w_vals, width=width, label="Evolution weight")

    plt.xticks(x, cats, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.xlabel("Category")
    plt.title("Self-evolution signal: weakness → pressure")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    return str(outp)


def plot_uncertainty_over_time(con, out_dir: str = "docs/figs") -> str:
    """
    Plot average judge disagreement per run (your uncertainty proxy).
    Uses: results.judge_json -> $.disagreement (already used in analyze.py)
    """
    outp = _ensure_dir(out_dir) / "uncertainty_over_time.png"

    rows = con.execute("""
        SELECT
          run_id,
          AVG(ABS(json_extract(judge_json, '$.disagreement'))) AS avg_dis
        FROM results
        GROUP BY run_id
        ORDER BY MIN(created_at) ASC
    """).fetchall()

    if not rows:
        raise RuntimeError("No results found (need at least one `run`).")

    xs = list(range(1, len(rows) + 1))
    ys = [float(r["avg_dis"]) if r["avg_dis"] is not None else 0.0 for r in rows]

    plt.figure()
    plt.plot(xs, ys, marker="o", label="Avg disagreement")
    plt.xlabel("Run index")
    plt.ylabel("Avg |score₁ - score₂|")
    plt.title("Uncertainty proxy over time (judge self-consistency)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    return str(outp)


def plot_category_difficulty_heatmap(con, out_dir: str = "docs/figs") -> str:
    """
    Heatmap of mean score by Category × Difficulty.
    Categories are fixed by design (CATEGORIES).
    Difficulty levels are inferred dynamically.
    """
    outp = _ensure_dir(out_dir) / "category_difficulty_heatmap.png"

    categories = CATEGORIES[:]  # design-level taxonomy

    # Infer difficulty levels dynamically
    diff_rows = con.execute("""
        SELECT DISTINCT difficulty FROM questions
        WHERE difficulty IS NOT NULL
        ORDER BY difficulty ASC
    """).fetchall()

    difficulties = [int(r["difficulty"]) for r in diff_rows]
    if not difficulties:
        raise RuntimeError("No difficulty levels found in questions table.")

    # Fetch mean scores
    rows = con.execute("""
        SELECT q.category AS category,
               q.difficulty AS difficulty,
               AVG(r.score) AS mean_score
        FROM results r
        JOIN questions q ON q.question_id = r.question_id
        GROUP BY q.category, q.difficulty
    """).fetchall()

    if not rows:
        raise RuntimeError("No results found (need at least one `run`).")

    # Build index maps
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    diff_to_idx = {d: i for i, d in enumerate(difficulties)}

    grid = [
        [float("nan") for _ in difficulties]
        for _ in categories
    ]

    for r in rows:
        c = r["category"]
        d = int(r["difficulty"])
        if c in cat_to_idx and d in diff_to_idx:
            grid[cat_to_idx[c]][diff_to_idx[d]] = float(r["mean_score"])

    plt.figure()
    plt.imshow(grid, aspect="auto", cmap="inferno")
    plt.colorbar(label="Mean score")
    # plt.clim(0.0, 1.0)

    plt.xticks(
        list(range(len(difficulties))),
        [f"{d}" for d in difficulties]
    )
    plt.yticks(
        list(range(len(categories))),
        categories
    )

    plt.xlabel("Difficulty")
    plt.ylabel("Category")
    plt.title("Mean score heatmap")
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()

    return str(outp)


def visualize_all(con, out_dir: str = "docs/figs") -> Dict[str, str]:
    """
    Generate the main figures (fast, high-signal) and return paths.
    """
    outputs = {}
    outputs["evolution"] = plot_evolution(con, out_dir=out_dir)
    outputs["category_pressure"] = plot_category_pressure(con, out_dir=out_dir)
    outputs["uncertainty_over_time"] = plot_uncertainty_over_time(con, out_dir=out_dir)
    outputs["cat_diff_heatmap"] = plot_category_difficulty_heatmap(con, out_dir=out_dir)
    return outputs
