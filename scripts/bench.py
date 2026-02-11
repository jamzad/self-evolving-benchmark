# scripts/bench.py
import os
import argparse
from dotenv import load_dotenv

from src.client import make_client
from src.store import connect, init_db
from src.openai_safe import ModelCaps
from src.generate import generate_questions
from src.run import run_benchmark
from src.report import report as make_report
from src.export_regression import export_regression
from src.evolve import category_means, format_weights


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Self-evolving benchmark generator (MVP+)")
    parser.add_argument("--db", default="data/bench.sqlite", help="Path to SQLite DB")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                        help="Default model for generate/solve/judge unless overridden.")
    parser.add_argument("--gen-model", default=None)
    parser.add_argument("--solve-model", default=None)
    parser.add_argument("--judge-model", default=None)

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize database")

    p_gen = sub.add_parser("generate", help="Generate novel questions (self-evolving by weakness profile)")
    p_gen.add_argument("--n", type=int, default=10)
    p_gen.add_argument("--domain", default="general")

    p_run = sub.add_parser("run", help="Run benchmark (answer + judge + EMA)")
    p_run.add_argument("--n", type=int, default=10)
    p_run.add_argument("--alpha", type=float, default=0.2)

    sub.add_parser("report", help="Print summary report")

    p_all = sub.add_parser("all", help="Generate -> Run -> Report in one command")
    p_all.add_argument("--n-gen", type=int, default=10)
    p_all.add_argument("--n-run", type=int, default=10)
    p_all.add_argument("--domain", default="general")
    p_all.add_argument("--alpha", type=float, default=0.2)

    p_exp = sub.add_parser("export-regression", help="Export worst-K questions to JSONL")
    p_exp.add_argument("--k", type=int, default=20)
    p_exp.add_argument("--out", default="data/regression.jsonl")

    args = parser.parse_args()

    con = connect(args.db)
    init_db(con)

    client = make_client(base_url=args.base_url)
    caps = ModelCaps()

    gen_model = args.gen_model or args.model
    solve_model = args.solve_model or args.model
    judge_model = args.judge_model or args.model

    if args.cmd == "init":
        print(f"Initialized DB at {args.db}")
        return

    if args.cmd == "generate":
        means = category_means(con)
        print("Evolve weights:", format_weights(means))
        items = generate_questions(client, caps, con, model=gen_model, n=args.n, domain=args.domain)
        print(f"Inserted {len(items)} novel questions.")
        if items:
            print("Example:", items[0]["prompt"])
        return

    if args.cmd == "run":
        out = run_benchmark(
            client, caps, con,
            base_url=args.base_url,
            solve_model=solve_model,
            judge_model=judge_model,
            n=args.n,
            alpha=args.alpha
        )
        print(f"Run {out['run_id']}: mean={out['batch_mean']:.3f} | EMA={out['ema']:.3f} | n={out['n']}")
        return

    if args.cmd == "report":
        print(make_report(con))
        return

    if args.cmd == "all":
        means = category_means(con)
        print("Evolve weights:", format_weights(means))
        items = generate_questions(client, caps, con, model=gen_model, n=args.n_gen, domain=args.domain)
        print(f"Inserted {len(items)} novel questions.")
        out = run_benchmark(
            client, caps, con,
            base_url=args.base_url,
            solve_model=solve_model,
            judge_model=judge_model,
            n=args.n_run,
            alpha=args.alpha
        )
        print(f"Run {out['run_id']}: mean={out['batch_mean']:.3f} | EMA={out['ema']:.3f} | n={out['n']}")
        print("")
        print(make_report(con))
        return

    if args.cmd == "export-regression":
        n = export_regression(con, out_path=args.out, k=args.k)
        print(f"Exported {n} questions to {args.out}")
        return

if __name__ == "__main__":
    main()
