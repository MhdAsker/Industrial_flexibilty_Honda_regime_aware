from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import ConfigError
from .workflow import evaluate_experiment, reproduce_experiment, validate_experiment


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="flexlab", description="Offline Industrial Flexibility Decision Lab")
    sub = result.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate experiment configuration and input contracts")
    validate.add_argument("--experiment", type=Path, required=True)
    for name in ("run", "evaluate", "optimize"):
        command = sub.add_parser(name, help="run the complete offline decision study" if name == "run" else f"run through {name}")
        command.add_argument("--experiment", type=Path, required=True); command.add_argument("--output", type=Path, required=True)
    for name in ("forecast", "scenarios"):
        command = sub.add_parser(name, help=f"validate and summarize the configured {name} inputs")
        command.add_argument("--experiment", type=Path, required=True)
    reproduce = sub.add_parser("reproduce", help="run twice and verify deterministic artifact hashes")
    reproduce.add_argument("--experiment", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command in {"validate", "forecast", "scenarios"}:
            config, data = validate_experiment(args.experiment)
            value = {"valid": True, "experiment_id": config.experiment_id, "dataset_id": data.dataset_id, "target_day_local": data.target_day_local, "intervals": len(data.timestamps_utc), "scenarios": len(data.scenario_weights), "provenance": data.provenance}
        elif args.command == "reproduce":
            value = reproduce_experiment(args.experiment)
        else:
            value = evaluate_experiment(args.experiment, args.output)
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (ConfigError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"FlexLab error: {exc}") from exc


if __name__ == "__main__":
    main()
