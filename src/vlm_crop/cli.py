"""Command line entry points. Live calls only occur under the explicit run command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .api import APIClient
from .budget import Ledger
from .core import Config
from .dataset import prepare
from .report import build_report
from .runner import execute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download and freeze seeded benchmark samples")
    prep.add_argument("--data", type=Path, default=Path("artifacts/data"))
    prep.add_argument("--per-task", type=int, default=20)
    prep.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("run", help="Run/resume API episodes with a persistent $10 ledger")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--data", type=Path, default=Path("artifacts/data"))
    run.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report", help="Export an API-free static viewer, PNGs and GIFs")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path, default=Path("site-output"))
    sub.add_parser("budget", help="Print settled and unresolved nominal spending")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.data, args.per_task, args.seed)
            result = {"samples": len(result["samples"]), "revision": result["revision"]}
        elif args.command == "report":
            result = build_report(args.run, args.output)
        else:
            # Always the same project ledger, even when run output directories differ.
            project = Path(__file__).resolve().parents[2]
            ledger = Ledger(project / "artifacts" / "budget.json")
            if args.command == "budget":
                result = ledger.summary()
            else:
                load_dotenv(project / ".env", override=False)
                cfg = Config.load(args.config)
                result = execute(args.data, args.output, cfg, APIClient(ledger), ledger)
        print(json.dumps(result, indent=2))
        if result.get("errors") or result.get("existing_errors") or result.get("stopped"):
            parser.exit(1)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
