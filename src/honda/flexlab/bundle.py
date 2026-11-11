from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def stable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def manifest_for(output: Path, *, experiment: dict[str, Any], provenance: dict[str, str], supported: list[str], unsupported: list[str]) -> dict[str, Any]:
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "manifest.json")
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "scipy", "PyYAML"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "NOT_INSTALLED"
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: commit = "UNKNOWN"
    return {
        "manifest_schema_version": "flexlab.manifest.v2",
        "experiment": experiment,
        "code_commit": commit,
        "python": platform.python_version(),
        "packages": packages,
        "input_provenance": provenance,
        "output_hashes": {str(p.relative_to(output)).replace("\\", "/"): digest(p) for p in files},
        "supported_claims": supported,
        "unsupported_claims": unsupported,
    }
