# scripts/bench.py
import os
import argparse
from dotenv import load_dotenv

from src.client import make_client
from src.store import connect, init_db
from src.generate import generate_questions
from src.run import run_benchmark

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Self-evolving benchmark (MVP)")
    parser.add_argument("--db", default="data/bench.sqlite", help="Path to SQLite DB")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_init = sub.add_parser("init", help="Initialize database")

    sub_gen = sub.add_parser("generate", help="Generate new questions")
    sub_gen.add_argument("--n", type=int, default=10)

    sub_run = sub.add_parser("run", help="Run benchmark on latest questions")
    sub_run.add_argument("--n", type=int, default=10)
    sub_run.add_argument("--temp", type=float, default=0.2)
    sub_run.add_argument("--alpha", type=float, default=0.2)

    args = parser.parse_args()

    con = connect(args.db)
    init_db(con)

    client = make_client()
    model = os.getenv("OPENAI_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if args.cmd == "init":
        print(f"Initialized DB at {args.db}")
        return

    if args.cmd == "generate":
        items = generate_questions(client=client, con=con, model=model, n=args.n, domain="general")

        print(f"Inserted {len(items)} new questions.")
        if items:
            print("Example:", items[0]["prompt"])
        return

    if args.cmd == "run":
        out = run_benchmark(
            client=client,
            con=con,
            model=model,
            base_url=base_url,
            temperature=args.temp,
            n=args.n,
            alpha=args.alpha
        )
        print(f"Run {out['run_id']}: mean={out['batch_mean']:.3f} | EMA={out['ema']:.3f} | n={out['n']}")
        return

if __name__ == "__main__":
    main()
