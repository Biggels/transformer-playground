from __future__ import annotations

import argparse
from pathlib import Path

from transformer_playground.config import apply_overrides, load_config
from transformer_playground.infer import compare_runs, generate_text, resolve_run_path
from transformer_playground.report import build_report
from transformer_playground.train import run_training


def _common_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
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
    p_gen.add_argument("--prompt", type=str, default=None)
    p_gen.add_argument("--unconditional", action="store_true")
    _common_sampling_args(p_gen)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--run-ids", type=str, required=True, help="Comma-separated run ids")
    p_cmp.add_argument("--runs-dir", type=str, default="runs")
    p_cmp.add_argument("--prompt", type=str, default=None)
    p_cmp.add_argument("--unconditional", action="store_true")
    p_cmp.add_argument("--n-samples", type=int, default=3)
    _common_sampling_args(p_cmp)

    p_report = sub.add_parser("report")
    p_report.add_argument("--run-id", type=str, required=True)
    p_report.add_argument("--runs-dir", type=str, default="runs")
    p_report.add_argument("--prompt", type=str, default=None)
    p_report.add_argument("--unconditional", action="store_true")
    p_report.add_argument("--n-samples", type=int, default=200)
    p_report.add_argument("--eval-batches", type=int, default=32)
    _common_sampling_args(p_report)

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args.set)
        run_dir = run_training(cfg)
        print(run_dir)
        return

    if args.cmd == "generate":
        run_path = resolve_run_path(args.runs_dir, args.run_id)
        prompt = None if args.unconditional else args.prompt
        if prompt is None and not args.unconditional:
            parser.error("generate requires --prompt or --unconditional")
        text = generate_text(
            run_path,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            checkpoint_name=args.checkpoint,
        )
        print(text)
        return

    if args.cmd == "compare":
        prompt = None if args.unconditional else args.prompt
        if prompt is None and not args.unconditional:
            parser.error("compare requires --prompt or --unconditional")
        run_paths = [resolve_run_path(args.runs_dir, rid.strip()) for rid in args.run_ids.split(",") if rid.strip()]
        rows = compare_runs(
            [str(p) for p in run_paths],
            prompt=prompt,
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
        return

    if args.cmd == "report":
        run_path = resolve_run_path(args.runs_dir, args.run_id)
        prompt = None if args.unconditional else args.prompt
        if prompt is None and not args.unconditional:
            parser.error("report requires --prompt or --unconditional")
        report, out_path = build_report(
            run_path=run_path,
            prompt=prompt,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            checkpoint_name=args.checkpoint,
            eval_batches=args.eval_batches,
            save=True,
        )
        metrics = report["metrics"]
        eval_block = report["eval"]
        print(f"run: {Path(run_path).name}")
        print(f"checkpoint: {args.checkpoint}")
        print(
            "eval: "
            f"train_loss={eval_block['train_loss']:.4f} "
            f"val_loss={eval_block['val_loss']:.4f} "
            f"val_ppl={eval_block['val_ppl']:.2f}"
        )
        print(
            "samples: "
            f"novel_ratio={metrics['novel_ratio']:.3f} "
            f"exact_match_any_ratio={metrics['exact_match_any_ratio']:.3f} "
            f"unique_ratio={metrics['unique_ratio']:.3f} "
            f"distinct_1={metrics['distinct_1']:.3f} "
            f"distinct_2={metrics['distinct_2']:.3f}"
        )
        if out_path is not None:
            print(f"saved_report: {out_path}")


if __name__ == "__main__":
    main()
