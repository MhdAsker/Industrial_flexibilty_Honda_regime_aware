from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .adaptive import AdaptiveCalibrationState, load_state, write_state_once
from .protocol import ProspectiveProtocol, phase2_exit_report, switchback_assignment
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
        raise TypeError(f"Expected a JSON numeric list in {path}")
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
    adaptive_init = commands.add_parser("adaptive-init", help="create an immutable adaptive-calibration state")
    adaptive_init.add_argument("--partner-id", required=True); adaptive_init.add_argument("--model-id", required=True)
    adaptive_init.add_argument("--effective-at-utc", required=True); adaptive_init.add_argument("--output", type=Path, required=True)
    adaptive_update = commands.add_parser("adaptive-update", help="update bias and conformal state after realization")
    adaptive_update.add_argument("--state", type=Path, required=True); adaptive_update.add_argument("--prediction", type=Path, required=True)
    adaptive_update.add_argument("--realized", type=Path, required=True); adaptive_update.add_argument("--effective-at-utc", required=True)
    adaptive_update.add_argument("--output", type=Path, required=True)
    protocol = commands.add_parser("protocol", help="validate a prospective protocol and show its frozen hash")
    protocol.add_argument("--protocol", type=Path, required=True); protocol.add_argument("--block-id")
    exit_report = commands.add_parser("exit-report", help="evaluate the predeclared Phase 2 exit criteria")
    exit_report.add_argument("--protocol", type=Path, required=True); exit_report.add_argument("--metrics", type=Path, required=True)
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
        elif args.command == "metrics":
            value = cohort_metrics(args.root)
        elif args.command == "adaptive-init":
            state = AdaptiveCalibrationState.initial(args.partner_id, args.model_id, args.effective_at_utc)
            write_state_once(args.output, state)
            value = {**asdict(state), "state_hash": state.state_hash}
        elif args.command == "adaptive-update":
            state = load_state(args.state).update(
                _json_vector(args.prediction), _json_vector(args.realized), args.effective_at_utc,
            )
            write_state_once(args.output, state)
            value = {**asdict(state), "state_hash": state.state_hash}
        elif args.command == "protocol":
            protocol = ProspectiveProtocol.load(args.protocol)
            value = {"protocol_hash": protocol.protocol_hash, "valid": True}
            if args.block_id:
                value["assignment"] = switchback_assignment(protocol, args.block_id)
        else:
            protocol = ProspectiveProtocol.load(args.protocol)
            metrics_value = json.loads(args.metrics.read_text(encoding="utf-8"))
            value = phase2_exit_report(protocol, metrics_value)
        print(json.dumps(value, indent=2, sort_keys=True))
    except (ValueError, RuntimeError, FileExistsError, KeyError, TypeError) as exc:
        raise SystemExit(f"Shadow scheduler error: {exc}") from exc


if __name__ == "__main__":
    main()
