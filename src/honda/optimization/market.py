from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PriceAcquisitionContract:
    zone: str
    year: int
    source_url: str
    raw_unit: str = "EUR/MWh"
    required_columns: tuple[str, str] = ("timestamp_utc", "price_eur_per_mwh")
    publication_metadata_required: bool = True


def validate_price_source(path: Path, contract: PriceAcquisitionContract) -> pd.DataFrame:
    """Validate a supplied price file; this interface never downloads implicitly."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    missing = set(contract.required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Price source lacks required columns: {sorted(missing)}")
    timestamp = pd.to_datetime(frame[contract.required_columns[0]], utc=True, errors="raise")
    price = pd.to_numeric(frame[contract.required_columns[1]], errors="raise")
    if timestamp.duplicated().any() or price.isna().any():
        raise ValueError("Price source has duplicate timestamps or missing prices")
    result = frame.copy(); result[contract.required_columns[0]] = timestamp; result[contract.required_columns[1]] = price
    return result.sort_values(contract.required_columns[0]).reset_index(drop=True)
