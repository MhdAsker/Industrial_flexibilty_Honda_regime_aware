from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import (
    capture_issue_snapshot,
    cohort_metrics,
    issue_recommendation,
    readiness_report,
    record_operator_decision,
    settle_shadow,
)


def _json_vector(path: Path) -> list[float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON numeric list in {path}")
    return [float(item) for item in value]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="shadow-scheduler", description="Human-in-the-loop industrial shadow scheduler")
    commands = result.add_subparsers(dest="command", required=True)
    readiness = commands.add_parser("readiness", help="evaluate the partner input and approval gate")
    readiness.add_argument("--partner", type=Path, required=True)
    snapshot = commands.add_parser("snapshot", help="capture an immutable issue-time input snapshot")
    snapshot.add_argument("--issue", type=Path, required=True); snapshot.add_argument("--partner", type=Path, required=True); snapshot.add_argument("--output-root", type=Path, required=True)
    recommend = commands.add_parser("recommend", help="issue a recommendation and deterministic fallback")
    recommend.add_argument("--snapshot", type=Path, required=True)
    operator = commands.add_parser("operator-decision", help="record approval, rejection, or modification")
    operator.add_argument("--snapshot", type=Path, required=True); operator.add_argument("--disposition", required=True); operator.add_argument("--operator-role", required=True)
    operator.add_argument("--reason-code"); operator.add_argument("--note"); operator.add_argument("--modified-schedule", type=Path)
    settle = commands.add_parser("settle", help="compare recommendation with realized operation")
    settle.add_argument("--snapshot", type=Path, required=True); settle.add_argument("--realized-baseline", type=Path, required=True); settle.add_argument("--realized-schedule", type=Path, required=True)
    metrics = commands.add_parser("metrics", help="aggregate prospective shadow-run metrics")
    metrics.add_argument("--root", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "readiness":
            value = readiness_report(args.partner)
        elif args.command == "snapshot":
            value = {"snapshot": str(capture_issue_snapshot(args.issue, args.partner, args.output_root))}
        elif args.command == "recommend":
            value = issue_recommendation(args.snapshot)
        elif args.command == "operator-decision":
            modified = None if args.modified_schedule is None else _json_vector(args.modified_schedule)
            value = record_operator_decision(args.snapshot, args.disposition, args.operator_role, args.reason_code, args.note, modified)
        elif args.command == "settle":
            value = settle_shadow(args.snapshot, _json_vector(args.realized_baseline), _json_vector(args.realized_schedule))
        else:
            value = cohort_metrics(args.root)
        print(json.dumps(value, indent=2, sort_keys=True))
    except (ValueError, RuntimeError, FileExistsError, KeyError, TypeError) as exc:
        raise SystemExit(f"Shadow scheduler error: {exc}") from exc


if __name__ == "__main__":
    main()
