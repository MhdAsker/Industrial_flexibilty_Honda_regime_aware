from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .contracts import SiteRegistration
from .multisite_ml import advanced_model_readiness, hierarchy_execution_plan
from .storage import PlatformStore


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="flexops", description="Approval-gated multi-site flexibility platform")
    result.add_argument("--database", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize platform storage")
    commands.add_parser("verify-audit", help="verify the immutable audit hash chain")
    stage = commands.add_parser("stage", help="show a tenant deployment stage"); stage.add_argument("--tenant", required=True)
    register = commands.add_parser("register-site", help="register an immutable site version")
    register.add_argument("--file", type=Path, required=True); register.add_argument("--actor", required=True)
    principal = commands.add_parser("create-principal", help="create an API principal and reveal its token once")
    principal.add_argument("--tenant", required=True); principal.add_argument("--principal", required=True)
    principal.add_argument("--role", required=True); principal.add_argument("--site", action="append", default=[]); principal.add_argument("--actor", required=True)
    serve = commands.add_parser("serve", help="serve the optional FastAPI application")
    serve.add_argument("--host", default="127.0.0.1"); serve.add_argument("--port", type=int, default=8000)
    ledger = commands.add_parser("ledger-status", help="summarize prediction-label ledger completion")
    ledger.add_argument("--tenant", required=True); ledger.add_argument("--model")
    compare = commands.add_parser("compare-models", help="compare models on identical observed ledger rows")
    compare.add_argument("--tenant", required=True); compare.add_argument("--champion", required=True); compare.add_argument("--challenger", required=True)
    readiness = commands.add_parser("ml-readiness", help="evaluate multi-site and advanced-model data readiness")
    readiness.add_argument("--approved-sites", type=int, required=True); readiness.add_argument("--seasonal-days", type=int, required=True)
    readiness.add_argument("--pooled-rows", type=int, required=True)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Registry input must contain a JSON object")
    return value


def main() -> None:
    args = parser().parse_args()
    try:
        store = PlatformStore(args.database)
        if args.command == "init":
            value = {"initialized": True, "database": str(store.path)}
        elif args.command == "verify-audit":
            value = store.verify_audit_chain()
        elif args.command == "stage":
            value = {"tenant_id": args.tenant, "deployment_stage": store.deployment_stage(args.tenant).value}
        elif args.command == "register-site":
            raw = _read_json(args.file)
            allowed = {item.name for item in fields(SiteRegistration)}
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise ValueError(f"Unknown site fields: {', '.join(unknown)}")
            site = SiteRegistration(**raw)
            value = {"sha256": store.register_site(site, args.actor), "site_id": site.site_id}
        elif args.command == "create-principal":
            token = store.create_principal(args.tenant, args.principal, args.role, tuple(args.site), args.actor)
            value = {"principal_id": args.principal, "api_token": token, "warning": "Store this token now; only its salted hash is persisted."}
        elif args.command == "ledger-status":
            rows = store.ledger_rows(args.tenant, args.model)
            statuses = {status: sum(row["label_status"] == status for row in rows) for status in ("PENDING", "OBSERVED", "UNAVAILABLE")}
            value = {"tenant_id": args.tenant, "model_id": args.model, "rows": len(rows), "label_status": statuses}
        elif args.command == "compare-models":
            value = store.compare_ledger_models(args.tenant, args.champion, args.challenger)
        elif args.command == "ml-readiness":
            value = {
                "hierarchy": hierarchy_execution_plan(args.approved_sites, args.seasonal_days),
                "advanced_models": advanced_model_readiness(
                    comparable_approved_sites=args.approved_sites,
                    seasonal_days=args.seasonal_days,
                    pooled_rows=args.pooled_rows,
                ),
            }
        else:
            try:
                import uvicorn
            except ImportError as exc:
                raise RuntimeError("Install the platform extra: pip install -e '.[platform]'") from exc
            from .api import create_app
            uvicorn.run(create_app(args.database), host=args.host, port=args.port)
            return
        print(json.dumps(value, indent=2, sort_keys=True))
    except (TypeError, ValueError, RuntimeError, PermissionError) as exc:
        raise SystemExit(f"FlexOps error: {exc}") from exc


if __name__ == "__main__":
    main()
