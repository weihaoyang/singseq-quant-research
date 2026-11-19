"""Command-line interface for jq2quant strategy transpiler."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jq2quant.runtime import run_jq_strategy
from jq2quant.transpiler import transpile_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jq2quant",
        description="Standalone Open-Source JoinQuant-to-Native Strategy Transpiler",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Transpile subcommand
    trans_p = subparsers.add_parser("transpile", help="Transpile a JoinQuant strategy into clean modern Strategy code")
    trans_p.add_argument("input_file", help="Path to input JoinQuant python file (.py)")
    trans_p.add_argument("-o", "--output", dest="output_file", help="Path to output python file (.py)")

    # Run subcommand
    run_p = subparsers.add_parser("run", help="Run a JoinQuant strategy directly via decoupled adapter")
    run_p.add_argument("input_file", help="Path to JoinQuant python file")
    run_p.add_argument("--start", default="20230101", help="Start date (YYYYMMDD)")
    run_p.add_argument("--end", default="20231231", help="End date (YYYYMMDD)")
    run_p.add_argument("--cash", type=float, default=1_000_000.0, help="Initial cash (default: 1,000,000)")
    run_p.add_argument("--benchmark", default="000300.SH", help="Benchmark code (default: 000300.SH)")

    args = parser.parse_args(argv)

    if args.command == "transpile":
        in_path = Path(args.input_file)
        if not in_path.exists():
            print(f"Error: Input file '{in_path}' does not exist.", file=sys.stderr)
            return 1
        out_path = args.output_file or in_path.with_name(f"{in_path.stem}_modern.py")
        transpile_file(in_path, out_path)
        print(f"[OK] Successfully transpiled '{in_path}' -> '{out_path}'.")
        return 0

    if args.command == "run":
        in_path = Path(args.input_file)
        if not in_path.exists():
            print(f"Error: Input file '{in_path}' does not exist.", file=sys.stderr)
            return 1
        print(f"[*] Running JoinQuant strategy '{in_path}' ({args.start} - {args.end})...")
        result = run_jq_strategy(
            in_path,
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.cash,
            benchmark=args.benchmark,
        )
        print(result.summary())
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
