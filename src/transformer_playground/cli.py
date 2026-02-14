from __future__ import annotations

import argparse
from pathlib import Path

from transformer_playground.config import apply_overrides, load_config
from transformer_playground.infer import compare_runs, generate_text, resolve_run_path
from transformer_playground.train import run_training


def _common_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=str, default="best_val.pt")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", type=str, default="configs/default.py")
    p_train.add_argument("--set", action="append", default=[])

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--run-id", type=str, required=True)
    p_gen.add_argument("--runs-dir", type=str, default="runs")
    p_gen.add_argument("--prompt", type=str, required=True)
    _common_sampling_args(p_gen)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--run-ids", type=str, required=True, help="Comma-separated run ids")
    p_cmp.add_argument("--runs-dir", type=str, default="runs")
    p_cmp.add_argument("--prompt", type=str, required=True)
    p_cmp.add_argument("--n-samples", type=int, default=3)
    _common_sampling_args(p_cmp)

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args.set)
        run_dir = run_training(cfg)
        print(run_dir)
        return

    if args.cmd == "generate":
        run_path = resolve_run_path(args.runs_dir, args.run_id)
        text = generate_text(
            run_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            checkpoint_name=args.checkpoint,
        )
        print(text)
        return

    if args.cmd == "compare":
        run_paths = [resolve_run_path(args.runs_dir, rid.strip()) for rid in args.run_ids.split(",") if rid.strip()]
        rows = compare_runs(
            [str(p) for p in run_paths],
            prompt=args.prompt,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            checkpoint_name=args.checkpoint,
        )
        for row in rows:
            print(f"=== {Path(row['run']).name} (params={row['params']}) ===")
            for i, s in enumerate(row["samples"], start=1):
                print(f"[{i}] {s}")


if __name__ == "__main__":
    main()
