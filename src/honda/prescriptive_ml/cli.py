from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluation import (
    artifact_digest,
    evaluate_point_forecast,
    paired_day_bootstrap,
    promotion_report,
    write_json,
)
from .experiment import run as run_experiment
from .scenario_experiment import run as run_scenario_experiment
from .scenario_selection import run as run_scenario_selection

ROOT = Path(__file__).resolve().parents[3]
TEST_PREDICTIONS = ROOT / "results/ml/test_predictions_2023.parquet"
DEVELOPMENT_PREDICTIONS = ROOT / "results/ml/development_predictions.parquet"
SILVER = ROOT / "data/silver/honda/honda_energy_15min.parquet"
MASK = ROOT / "data/silver/honda/provenance_masks_15min.parquet"
INDEX = ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet"


def audit() -> dict:
    required_predictions = (TEST_PREDICTIONS, DEVELOPMENT_PREDICTIONS)
    missing_predictions = [str(path.relative_to(ROOT)) for path in required_predictions if not path.exists()]
    if missing_predictions:
        raise ValueError(f"Missing frozen prediction artifacts: {', '.join(missing_predictions)}")
    frame = pd.read_parquet(TEST_PREDICTIONS)
    point = evaluate_point_forecast(frame.actual_kw.to_numpy(), frame.PRIMARY_ML.to_numpy())
    point["evaluation_exposed"] = True
    bootstrap = paired_day_bootstrap(frame, "PRIMARY_ML", "B5")
    source_paths = (SILVER, MASK, INDEX)
    missing_sources = [str(path.relative_to(ROOT)) for path in source_paths if not path.exists()]
    return {
        "schema_version": "prescriptive-ml.phase1-audit.v1",
        "current_champion": "LGBM_M1",
        "champion_metrics": point,
        "champion_vs_b5": bootstrap,
        "promotion_gate_example": promotion_report(point, bootstrap),
        "method_status": {
            "regime_aware_mixture": "DEVELOPMENT_SELECTED_EXPLORATORY_DO_NOT_PROMOTE",
            "quantile_lightgbm": "EVALUATED_EXPLORATORY_DO_NOT_PROMOTE",
            "grouped_conformal_calibration": "EVALUATED_EXPLORATORY_DO_NOT_PROMOTE",
            "conditional_coherent_ecc": "DEVELOPMENT_SELECTED_EXPLORATORY_DO_NOT_PROMOTE",
            "scenario_reduction": "DEVELOPMENT_REJECTED_RETAIN_200_RAW_PATHS",
            "execute_first_rolling_mpc_harness": "IMPLEMENTED_REAL_VINTAGES_REQUIRED",
            "decision_focused_learning": "DEFERRED_UNTIL_ROLLING_BENCHMARK",
        },
        "training_readiness": {
            "ready": not missing_sources,
            "missing_provenance_controlled_sources": missing_sources,
            "reason": "challenger training requires the frozen SILVER, provenance mask, and forecasting index",
        },
        "artifacts": {
            str(path.relative_to(ROOT)): artifact_digest(path) for path in required_predictions
        },
        "claim_boundary": "EXPLORATORY_EXPOSED_2023_HOLDOUT_NO_PROMOTION_CLAIM",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="prescriptive-ml", description="Offline prescriptive-ML challenger controls")
    commands = result.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="audit current champion and Phase 1 training readiness")
    audit_parser.add_argument("--output", type=Path)
    run_parser = commands.add_parser("run", help="train and evaluate the frozen small challenger set")
    run_parser.add_argument("--output", type=Path, default=ROOT / "results/prescriptive_ml")
    scenarios_parser = commands.add_parser("scenarios", help="run the exploratory conditional ECC comparison")
    scenarios_parser.add_argument("--output", type=Path, default=ROOT / "results/prescriptive_ml")
    scenarios_parser.add_argument("--raw-count", type=int, default=200)
    scenarios_parser.add_argument("--reduced-count", type=int, default=25)
    scenarios_parser.add_argument("--bandwidth", type=float, default=2.0)
    selection_parser = commands.add_parser("select-scenarios", help="select ECC bandwidth and retained count on development data")
    selection_parser.add_argument("--output", type=Path, default=ROOT / "results/prescriptive_ml")
    feature_parser = commands.add_parser("build-features", help="rebuild the frozen issuance-safe H5 feature cache")
    feature_parser.add_argument("--output", type=Path, default=ROOT / "results/ml/h5_feature_matrix.parquet")
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "audit":
            value = audit()
        elif args.command == "run":
            value = run_experiment(args.output)
        elif args.command == "select-scenarios":
            value = run_scenario_selection(args.output)
        elif args.command == "scenarios":
            value = run_scenario_experiment(args.output, args.raw_count, args.reduced_count, args.bandwidth)
        else:
            from honda import h5_deterministic_ml as h5

            h5.verify_checkpoint()
            frame, _families, _catalog, _audit = h5.build_features()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.output, index=False)
            value = {"rows": len(frame), "columns": len(frame.columns), "output": str(args.output.resolve())}
        if args.command == "audit" and args.output is not None:
            write_json(args.output, value)
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Prescriptive ML error: {exc}") from exc


if __name__ == "__main__":
    main()
