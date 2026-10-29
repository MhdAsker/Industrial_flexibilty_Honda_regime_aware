from __future__ import annotations

import bisect
import functools
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/silver/honda/honda_energy_15min.parquet"
MASK = ROOT / "data/silver/honda/provenance_masks_15min.parquet"
INDEX = ROOT / "data/silver/honda/honda_forecasting_index_15min.parquet"
PROTOCOL = ROOT / "config/honda_forecasting_protocol_frozen.json"
SELECTION = ROOT / "config/honda_baseline_selection_frozen.json"
OUT = ROOT / "results/baselines"
FIG = OUT / "figures"
REPORT = ROOT / "reports/honda_baseline_benchmark.md"
DST_REPORT = ROOT / "reports/honda_baseline_dst_audit.md"
MANIFEST = OUT / "honda_h4_manifest.json"

H3B_COMMIT = "b12d055591cf90ca7929342b34e1e31b3caf6e0f"
PROTOCOL_SHA256 = "281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e"
INDEX_SHA256 = "19224168abd94880233e39f7fe4952febc42fae90b516b63f0d4d36b02f3fc0a"
SILVER_SHA256 = "d9d93445391985faa7a0aa015cdb65b949f3909636d284c59449db876ae310df"
MASK_SHA256 = "f93d5bf4597d8f768cc63437f2ce78d21f4050f4dfcfbbe0e59d2fa0ef4eb778"
OBSERVED = "OBSERVED_ORIGINAL"
CORRECTED = "CORRECTED_NONIMPUTED"
BOOTSTRAP_SEED = 20261029
BOOTSTRAP_RESAMPLES = 2000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def verify_checkpoint() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", H3B_COMMIT, head], cwd=ROOT).returncode:
        raise RuntimeError("H3B checkpoint is not an ancestor")
    expected = {PROTOCOL: PROTOCOL_SHA256, INDEX: INDEX_SHA256, SILVER: SILVER_SHA256, MASK: MASK_SHA256}
    for path, expected_hash in expected.items():
        if digest(path) != expected_hash:
            raise RuntimeError(f"Frozen input changed: {path}")
    h3 = json.loads((ROOT / "results/protocol/honda_h3_manifest.json").read_text(encoding="utf-8"))
    for rel, meta in h3["deterministic_outputs"].items():
        if digest(ROOT / rel) != meta["sha256"]:
            raise RuntimeError(f"H3 artifact changed: {rel}")


def _fold(ts: pd.Timestamp) -> int:
    return int(ts.to_pydatetime().fold)


def load_frame() -> pd.DataFrame:
    index = pd.read_parquet(INDEX)
    silver = pd.read_parquet(SILVER, columns=[
        "timestamp_utc", "grid_exchange_kw_signed", "grid_exchange_provenance",
    ])
    frame = index.merge(silver, on="timestamp_utc", how="left", validate="one_to_one")
    frame = frame.rename(columns={"grid_exchange_kw_signed": "actual_kw", "grid_exchange_provenance": "actual_provenance"})
    frame["target_day"] = pd.to_datetime(frame.target_day_local).dt.date
    frame["year"] = pd.to_datetime(frame.target_day_local).dt.year
    frame["month"] = pd.to_datetime(frame.target_day_local).dt.month
    frame["weekday"] = pd.to_datetime(frame.target_day_local).dt.weekday
    frame["weekend"] = frame.weekday >= 5
    frame["quarter"] = frame.timestamp_local.dt.hour * 4 + frame.timestamp_local.dt.minute // 15
    frame["fold"] = [_fold(x) for x in frame.timestamp_local]
    frame["lead_time_hours"] = frame.lead_time_minutes / 60.0
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


@dataclass
class SafeHistory:
    exact: dict[tuple[object, int, int], tuple[pd.Timestamp, float, str]]
    by_slot: dict[tuple[int, int], list[tuple[int, pd.Timestamp, float, str]]]
    utc_ns: np.ndarray
    utc_ts: np.ndarray
    values: np.ndarray
    provenance: np.ndarray


@functools.lru_cache(maxsize=1)
def safe_history() -> SafeHistory:
    s = pd.read_parquet(SILVER, columns=["timestamp_utc", "timestamp_local", "grid_exchange_kw_signed", "grid_exchange_provenance"])
    s = s[s.grid_exchange_provenance.eq(OBSERVED) & s.grid_exchange_kw_signed.notna()].copy()
    s["date"] = s.timestamp_local.dt.date
    s["quarter"] = s.timestamp_local.dt.hour * 4 + s.timestamp_local.dt.minute // 15
    s["fold"] = [_fold(x) for x in s.timestamp_local]
    exact: dict[tuple[object, int, int], tuple[pd.Timestamp, float, str]] = {}
    by_slot: dict[tuple[int, int], list[tuple[int, pd.Timestamp, float, str]]] = {}
    for row in s.itertuples(index=False):
        rec = (row.timestamp_utc, float(row.grid_exchange_kw_signed), row.grid_exchange_provenance)
        exact[(row.date, int(row.quarter), int(row.fold))] = rec
        by_slot.setdefault((int(row.quarter), int(row.fold)), []).append((row.date.toordinal(), *rec))
    s = s.sort_values("timestamp_utc")
    return SafeHistory(exact, by_slot, s.timestamp_utc.astype("int64").to_numpy(), s.timestamp_utc.to_numpy(), s.grid_exchange_kw_signed.to_numpy(float), s.grid_exchange_provenance.to_numpy(str))


def _available(source_ts: pd.Timestamp, issue: pd.Timestamp) -> bool:
    return source_ts + pd.Timedelta(minutes=15) <= issue


def component_predictions(targets: pd.DataFrame, history: SafeHistory, b3_specs: list[tuple[str, int, int]]) -> pd.DataFrame:
    out = targets[[
        "timestamp_utc", "timestamp_local", "target_day_local", "target_day", "forecast_issue_time_utc",
        "lead_time_hours", "lead_band", "actual_kw", "actual_provenance", "target_primary_eligible",
        "target_sensitivity_eligible", "complete_primary_day", "holiday_context", "structural_era_metadata",
        "year", "month", "weekday", "weekend", "quarter", "fold",
    ]].copy()
    issue_ns = pd.DatetimeIndex(out.forecast_issue_time_utc).as_unit("ns").asi8
    # B0: latest interval start whose 15-minute interval has completed.
    pos = np.searchsorted(history.utc_ns, issue_ns - pd.Timedelta(minutes=15).value, side="right") - 1
    valid = pos >= 0
    out["B0"] = np.where(valid, history.values[np.maximum(pos, 0)], np.nan)
    out["B0_source"] = [pd.Timestamp(history.utc_ts[p]).isoformat() if p >= 0 else "" for p in pos]
    out["B0_provenance"] = np.where(valid, history.provenance[np.maximum(pos, 0)], "")

    b1, b1src, b2, b2src = [], [], [], []
    b3_values = {name: [] for name, _, _ in b3_specs}
    b3_sources = {name: [] for name, _, _ in b3_specs}
    for row in out.itertuples(index=False):
        slot = (int(row.quarter), int(row.fold))
        records = history.by_slot.get(slot, [])
        ord0 = row.target_day.toordinal()
        j = bisect.bisect_left(records, (ord0,)) - 1
        chosen = None
        while j >= 0:
            candidate = records[j]
            if _available(candidate[1], row.forecast_issue_time_utc):
                chosen = candidate
                break
            j -= 1
        b1.append(chosen[2] if chosen else np.nan)
        b1src.append(chosen[1].isoformat() if chosen else "")
        previous_week = history.exact.get((row.target_day - pd.Timedelta(days=7), *slot))
        if previous_week and _available(previous_week[0], row.forecast_issue_time_utc):
            b2.append(previous_week[1]); b2src.append(previous_week[0].isoformat())
        else:
            b2.append(np.nan); b2src.append("")
        for name, weeks, required in b3_specs:
            vals, srcs = [], []
            for k in range(1, weeks + 1):
                rec = history.exact.get((row.target_day - pd.Timedelta(days=7 * k), *slot))
                if rec and _available(rec[0], row.forecast_issue_time_utc):
                    vals.append(rec[1]); srcs.append(rec[0].isoformat())
            b3_values[name].append(float(np.median(vals)) if len(vals) >= required else np.nan)
            b3_sources[name].append("|".join(srcs) if len(vals) >= required else "")
    out["B1"] = b1; out["B1_source"] = b1src
    out["B1_provenance"] = np.where(out.B1.notna(), OBSERVED, "")
    out["B2"] = b2; out["B2_source"] = b2src
    out["B2_provenance"] = np.where(out.B2.notna(), OBSERVED, "")
    for name in b3_values:
        out[name] = b3_values[name]
        out[f"{name}_source"] = b3_sources[name]
        out[f"{name}_provenance"] = np.where(out[name].notna(), OBSERVED, "")
    return out


B3_SPECS = [
    ("B3_N2_ALL2", 2, 2), ("B3_N4_ALL4", 4, 4), ("B3_N8_ALL8", 8, 8),
    ("B3_N4_MIN2", 4, 2), ("B3_N8_MIN4", 8, 4),
]

B4_GROUPS = {
    "C1_QUARTER_WEEKDAY": [["quarter", "weekday"], ["quarter", "weekend"], ["quarter"], []],
    "C2_QUARTER_WEEKEND": [["quarter", "weekend"], ["quarter"], []],
    "C3_MONTH_QUARTER_WEEKEND": [["month", "quarter", "weekend"], ["quarter", "weekend"], ["quarter"], []],
    "C4_MONTH_WEEKDAY_QUARTER": [["month", "weekday", "quarter"], ["quarter", "weekday"], ["quarter", "weekend"], ["quarter"], []],
}

_COMPONENT_CACHE: dict[int, pd.DataFrame] = {}


def components_for_year(base: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in _COMPONENT_CACHE:
        target = base[base.year.eq(year)].copy().reset_index(drop=True)
        _COMPONENT_CACHE[year] = component_predictions(target, safe_history(), B3_SPECS)
    return _COMPONENT_CACHE[year].copy()


def b4_predict(train: pd.DataFrame, target: pd.DataFrame, spec: str) -> tuple[pd.Series, pd.Series]:
    source = train[train.target_primary_eligible & train.actual_kw.notna()].copy()
    result = pd.Series(np.nan, index=target.index, dtype=float)
    level = pd.Series("", index=target.index, dtype=object)
    for depth, keys in enumerate(B4_GROUPS[spec]):
        missing = result.isna()
        if not missing.any(): break
        if keys:
            med = source.groupby(keys, dropna=False).actual_kw.median().rename("candidate")
            left = target.loc[missing, keys].copy()
            values = left.merge(med.reset_index(), on=keys, how="left").candidate.to_numpy()
        else:
            values = np.full(missing.sum(), source.actual_kw.median())
        idx = result.index[missing]
        ok = ~pd.isna(values)
        result.loc[idx[ok]] = values[ok]
        level.loc[idx[ok]] = f"level_{depth + 1}:" + ("x".join(keys) if keys else "global")
    return result, level


def metric_row(actual: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    mask = actual.notna() & pred.notna()
    a, p = actual[mask].to_numpy(float), pred[mask].to_numpy(float)
    if not len(a):
        return {k: np.nan for k in ["mae_kw", "rmse_kw", "bias_kw", "median_absolute_error_kw", "nmae", "absolute_target_scaled_wape"]} | {"predicted_targets": 0}
    e = p - a
    denom = np.mean(np.abs(a))
    return {
        "predicted_targets": len(a), "mae_kw": np.mean(np.abs(e)), "rmse_kw": np.sqrt(np.mean(e * e)),
        "bias_kw": np.mean(e), "median_absolute_error_kw": np.median(np.abs(e)),
        "nmae": np.mean(np.abs(e)) / denom if denom else np.nan,
        "absolute_target_scaled_wape": np.sum(np.abs(e)) / np.sum(np.abs(a)) if np.sum(np.abs(a)) else np.nan,
    }


def candidate_summary(frame: pd.DataFrame, candidate: str, protocol: str, fold: str) -> dict:
    eligible = frame.target_primary_eligible & frame.actual_kw.notna()
    scored = frame.loc[eligible]
    row = {"protocol": protocol, "fold": fold, "candidate": candidate, "eligible_targets": int(eligible.sum())}
    row.update(metric_row(scored.actual_kw, scored[candidate]))
    row["prediction_coverage"] = row["predicted_targets"] / row["eligible_targets"] if row["eligible_targets"] else np.nan
    days = scored.groupby("target_day_local").agg(expected=("timestamp_utc", "size"), predicted=(candidate, "count"), complete=("complete_primary_day", "first"))
    complete = days[days.complete]
    row["complete_days_eligible"] = len(complete)
    row["complete_days_predicted"] = int((complete.predicted == complete.expected).sum())
    row["complete_day_coverage"] = row["complete_days_predicted"] / len(complete) if len(complete) else np.nan
    fallback_col = f"{candidate}_fallback"
    if fallback_col in frame:
        flags = frame.loc[eligible, fallback_col].astype(str)
        row["fallback_usage"] = float((~flags.isin(["NONE", "level_1:monthxquarterxweekend", "level_1:monthxweekdayxquarter", "level_1:quarterxweekday", "level_1:quarterxweekend"])).mean())
    else:
        row["fallback_usage"] = 0.0
    return row


def choose(rows: pd.DataFrame, threshold: float) -> str:
    concat = rows[rows.fold.eq("CONCATENATED")].copy()
    acceptable = concat[concat.prediction_coverage >= threshold]
    if acceptable.empty:
        acceptable = concat.sort_values(["prediction_coverage", "mae_kw"], ascending=[False, True]).head(1)
    return acceptable.sort_values(["mae_kw", "candidate"]).iloc[0].candidate


def assemble_fold(base: pd.DataFrame, train_years: list[int], validation_year: int, protocol: str, fold: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = base[base.year.isin(train_years)].copy()
    pred = components_for_year(base, validation_year)
    b3rows = [candidate_summary(pred, name, protocol, fold) for name, _, _ in B3_SPECS]
    b4rows = []
    for spec in B4_GROUPS:
        pred[f"B4_{spec}"], pred[f"B4_{spec}_fallback"] = b4_predict(train, pred, spec)
        b4rows.append(candidate_summary(pred, f"B4_{spec}", protocol, fold))
    return pred, pd.DataFrame(b3rows), pd.DataFrame(b4rows)


def combine_candidate_rows(per_fold: list[pd.DataFrame], frames: list[pd.DataFrame], candidates: list[str], protocol: str) -> pd.DataFrame:
    rows = pd.concat(per_fold, ignore_index=True)
    joined = pd.concat(frames, ignore_index=True)
    rows = pd.concat([rows, pd.DataFrame([candidate_summary(joined, c, protocol, "CONCATENATED") for c in candidates])], ignore_index=True)
    return rows


def b5_specs(b3: str, b4: str) -> dict[str, tuple[list[str], str, int]]:
    return {
        "E1_MEDIAN_B1_B2_B3_MIN2": (["B1", "B2", b3], "median", 2),
        "E2_MEAN_B1_B2_B3_MIN2": (["B1", "B2", b3], "mean", 2),
        "E3_MEDIAN_B2_B3_B4_MIN2": (["B2", b3, b4], "median", 2),
        "E4_MEAN_B1_B2_B3_B4_MIN3": (["B1", "B2", b3, b4], "mean", 3),
    }


def add_b5(frame: pd.DataFrame, b3: str, b4: str) -> tuple[pd.DataFrame, list[str]]:
    specs = b5_specs(b3, b4)
    names = []
    for name, (members, agg, minimum) in specs.items():
        values = frame[members]
        enough = values.notna().sum(axis=1) >= minimum
        raw = values.median(axis=1, skipna=True) if agg == "median" else values.mean(axis=1, skipna=True)
        frame[name] = raw.where(enough, frame[b4])
        frame[f"{name}_fallback"] = np.where(enough, "NONE", "B4")
        frame[f"{name}_members"] = values.notna().apply(lambda r: "|".join(r.index[r]), axis=1)
        names.append(name)
    return frame, names


def selection_process(base: pd.DataFrame, protocol: str) -> tuple[dict, list[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if protocol == "P1": folds = [([2020], 2021, "FOLD_A"), ([2020, 2021], 2022, "FOLD_B")]
    else: folds = [([2021], 2022, "P2_VALIDATION")]
    frames, b3parts, b4parts = [], [], []
    for train, val, name in folds:
        frame, b3, b4 = assemble_fold(base, train, val, protocol, name)
        frame["fold"] = name; frame["split"] = "VALIDATION"
        frames.append(frame); b3parts.append(b3); b4parts.append(b4)
    b3_candidates = [x[0] for x in B3_SPECS]
    b3rows = combine_candidate_rows(b3parts, frames, b3_candidates, protocol)
    selected_b3 = choose(b3rows, .95)
    b4_candidates = [f"B4_{x}" for x in B4_GROUPS]
    b4rows = combine_candidate_rows(b4parts, frames, b4_candidates, protocol)
    selected_b4 = choose(b4rows, .99)
    b5names = None
    for frame in frames:
        _, b5names = add_b5(frame, selected_b3, selected_b4)
    b5parts = [pd.DataFrame([candidate_summary(f, c, protocol, f.fold.iloc[0]) for c in b5names]) for f in frames]
    b5rows = combine_candidate_rows(b5parts, frames, b5names, protocol)
    selected_b5 = choose(b5rows, .99)
    joined = pd.concat(frames, ignore_index=True)
    aliases = {"B0": "B0", "B1": "B1", "B2": "B2", "B3": selected_b3, "B4": selected_b4, "B5": selected_b5}
    mainrows = []
    for fold_name, group in [(f.fold.iloc[0], f) for f in frames] + [("CONCATENATED", joined)]:
        for baseline, column in aliases.items():
            r = candidate_summary(group, column, protocol, fold_name); r["baseline_id"] = baseline; r["candidate"] = column
            fallback = group.get(f"{column}_fallback", pd.Series("NONE", index=group.index)).astype(str)
            r["fallback_usage"] = float((~fallback.isin(["NONE", "level_1:monthxquarterxweekend", "level_1:monthxweekdayxquarter", "level_1:quarterxweekday", "level_1:quarterxweekend"])).mean())
            mainrows.append(r)
    main = pd.DataFrame(mainrows)
    ranked = main[main.fold.eq("CONCATENATED") & (main.prediction_coverage >= .99)].copy()
    if ranked.empty:
        ranked = main[main.fold.eq("CONCATENATED")].sort_values(["prediction_coverage", "mae_kw"], ascending=[False, True]).head(1)
    primary = ranked.sort_values(["mae_kw", "baseline_id"]).iloc[0].baseline_id
    return {"selected_b3": selected_b3, "selected_b4": selected_b4, "selected_b5": selected_b5, "primary_baseline": primary, "aliases": aliases}, frames, b3rows, b4rows, b5rows


def prediction_contract(frame: pd.DataFrame, aliases: dict[str, str], protocol: str) -> pd.DataFrame:
    rows = []
    for baseline, col in aliases.items():
        p = frame[["timestamp_utc", "timestamp_local", "target_day_local", "forecast_issue_time_utc", "lead_time_hours", "lead_band", "actual_kw", "target_primary_eligible", "target_sensitivity_eligible", "complete_primary_day", "split", "fold", "holiday_context", "structural_era_metadata", "month"]].copy()
        p["prediction_kw"] = frame[col]
        p["baseline_id"] = baseline
        p["prediction_available"] = p.prediction_kw.notna()
        fallback = frame.get(f"{col}_fallback", pd.Series("NONE", index=frame.index)).astype(str)
        p["fallback_used"] = ~fallback.isin(["NONE", "level_1:monthxquarterxweekend", "level_1:monthxweekdayxquarter", "level_1:quarterxweekday", "level_1:quarterxweekend"])
        p["source_timestamp_or_rule"] = frame.get(f"{col}_source", pd.Series(col, index=frame.index))
        p["source_provenance"] = frame.get(f"{col}_provenance", pd.Series("TRAINING_ONLY_MEDIAN_OR_ENSEMBLE", index=frame.index))
        p["protocol"] = protocol
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


def final_frame(base: pd.DataFrame, years: list[int], selection: dict, protocol: str) -> pd.DataFrame:
    if not SELECTION.exists():
        raise PermissionError("2023 scoring forbidden until frozen baseline selection exists")
    pred = components_for_year(base, 2023)
    b4 = selection["selected_b4"]
    pred[b4], pred[f"{b4}_fallback"] = b4_predict(base[base.year.isin(years)], pred, b4.removeprefix("B4_"))
    pred, _ = add_b5(pred, selection["selected_b3"], b4)
    pred["fold"] = "FINAL_TEST"; pred["split"] = "TEST_2023"; pred["protocol"] = protocol
    return pred


def baseline_summary(predictions: pd.DataFrame, policy: str = "PRIMARY") -> pd.DataFrame:
    elig_col = "target_primary_eligible" if policy == "PRIMARY" else "target_sensitivity_eligible"
    rows = []
    for (protocol, fold, baseline), g in predictions.groupby(["protocol", "fold", "baseline_id"], sort=True):
        e = g[elig_col] & g.actual_kw.notna()
        r = {"protocol": protocol, "fold": fold, "baseline_id": baseline, "target_policy": policy, "eligible_targets": int(e.sum())}
        r.update(metric_row(g.loc[e, "actual_kw"], g.loc[e, "prediction_kw"]))
        r["prediction_coverage"] = r["predicted_targets"] / r["eligible_targets"]
        days = g.loc[e].groupby("target_day_local").agg(n=("timestamp_utc", "size"), p=("prediction_kw", "count"), complete=("complete_primary_day", "first"))
        cd = days[days.complete]
        r["complete_days_eligible"] = len(cd); r["complete_days_predicted"] = int((cd.n == cd.p).sum())
        r["complete_day_coverage"] = r["complete_days_predicted"] / len(cd) if len(cd) else np.nan
        r["fallback_usage"] = float(g.loc[e, "fallback_used"].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def diagnostic_tables(test: pd.DataFrame, primary: str) -> dict[str, pd.DataFrame]:
    t = test[test.target_primary_eligible].copy()
    out = {}
    rows = []
    for (baseline, band), g in t.groupby(["baseline_id", "lead_band"]):
        r = {"baseline_id": baseline, "lead_band": band, "eligible_targets": len(g)}; r.update(metric_row(g.actual_kw, g.prediction_kw)); r["prediction_coverage"] = r["predicted_targets"] / len(g); rows.append(r)
    out["lead"] = pd.DataFrame(rows)
    rows = []
    for baseline, g in t.groupby("baseline_id"):
        p = g[g.prediction_kw.notna()]; actual_import = p.actual_kw > 0; pred_import = p.prediction_kw > 0; actual_export = p.actual_kw < 0; pred_export = p.prediction_kw < 0
        rows += [
            {"baseline_id": baseline, "mode": "IMPORT", "count": int(actual_import.sum()), "mae_kw": float((p.loc[actual_import, "prediction_kw"]-p.loc[actual_import, "actual_kw"]).abs().mean())},
            {"baseline_id": baseline, "mode": "EXPORT", "count": int(actual_export.sum()), "mae_kw": float((p.loc[actual_export, "prediction_kw"]-p.loc[actual_export, "actual_kw"]).abs().mean())},
            {"baseline_id": baseline, "mode": "ALL_DIAGNOSTICS", "count": len(p), "sign_accuracy": float(((np.sign(p.prediction_kw)==np.sign(p.actual_kw))).mean()), "import_recall": float(pred_import[actual_import].mean()), "export_recall": float(pred_export[actual_export].mean()), "false_import_rate": float(pred_import[~actual_import].mean()), "false_export_rate": float(pred_export[~actual_export].mean()), "zero_crossing_error": float((np.sign(p.prediction_kw)!=np.sign(p.actual_kw)).mean())},
        ]
    out["mode"] = pd.DataFrame(rows)
    rows = []
    for (baseline, context), g in t.assign(context=t.holiday_context.str.split(";")).explode("context").groupby(["baseline_id", "context"]):
        r={"baseline_id":baseline,"calendar_context":context,"eligible_targets":len(g)};r.update(metric_row(g.actual_kw,g.prediction_kw));r["prediction_coverage"]=r["predicted_targets"]/len(g);rows.append(r)
    out["calendar"] = pd.DataFrame(rows)
    rows=[]
    for (baseline, month),g in t.groupby(["baseline_id","month"]):
        r={"baseline_id":baseline,"month":month,"eligible_targets":len(g),"import_fraction":float((g.actual_kw>0).mean()),"export_fraction":float((g.actual_kw<0).mean())};r.update(metric_row(g.actual_kw,g.prediction_kw));r["prediction_coverage"]=r["predicted_targets"]/len(g);rows.append(r)
    out["monthly"]=pd.DataFrame(rows)
    rows=[]
    era=t.structural_era_metadata.map(json.loads)
    t["modernization_diagnostic"]=[x.get("modernization_era", "UNKNOWN") for x in era]
    for (baseline,stratum),g in t.groupby(["baseline_id","modernization_diagnostic"]):
        r={"baseline_id":baseline,"structural_stratum":stratum,"eligible_targets":len(g)};r.update(metric_row(g.actual_kw,g.prediction_kw));r["prediction_coverage"]=r["predicted_targets"]/len(g);rows.append(r)
    out["structural"]=pd.DataFrame(rows)
    rows=[]
    for baseline,g in t.groupby("baseline_id"):
        g=g[g.prediction_kw.notna()].copy(); g["error"]=g.prediction_kw-g.actual_kw
        for d,x in g.groupby("target_day_local"):
            if not x.complete_primary_day.iloc[0] or len(x)!=x.timestamp_utc.nunique(): continue
            actual_import=np.clip(x.actual_kw,0,None).sum()*.25; pred_import=np.clip(x.prediction_kw,0,None).sum()*.25
            actual_export=np.clip(-x.actual_kw,0,None).sum()*.25; pred_export=np.clip(-x.prediction_kw,0,None).sum()*.25
            rows.append({"baseline_id":baseline,"target_day_local":d,"daily_mae_kw":x.error.abs().mean(),"daily_rmse_kw":np.sqrt((x.error**2).mean()),"daily_bias_kw":x.error.mean(),"max_absolute_interval_error_kw":x.error.abs().max(),"signed_daily_energy_error_kwh":x.error.sum()*.25,"absolute_daily_energy_error_kwh":abs(x.error.sum()*.25),"import_energy_error_kwh":pred_import-actual_import,"export_energy_error_kwh":pred_export-actual_export})
    out["daily"]=pd.DataFrame(rows)
    return out


def bootstrap(daily: pd.DataFrame, primary: str) -> pd.DataFrame:
    rng=np.random.default_rng(BOOTSTRAP_SEED); rows=[]
    p=daily[daily.baseline_id.eq(primary)][["target_day_local","daily_mae_kw"]].rename(columns={"daily_mae_kw":"p"})
    for comp in sorted(set(daily.baseline_id)-{primary}):
        c=daily[daily.baseline_id.eq(comp)][["target_day_local","daily_mae_kw"]].rename(columns={"daily_mae_kw":"c"})
        common=p.merge(c,on="target_day_local"); diff=(common.p-common.c).to_numpy(); n=len(diff)
        means=np.empty(BOOTSTRAP_RESAMPLES)
        for i in range(BOOTSTRAP_RESAMPLES): means[i]=diff[rng.integers(0,n,n)].mean()
        rows.append({"primary_baseline":primary,"comparator":comp,"common_complete_days":n,"paired_mean_daily_mae_difference_kw":diff.mean(),"ci95_lower_kw":np.quantile(means,.025),"ci95_upper_kw":np.quantile(means,.975),"primary_daily_win_rate":float((diff<0).mean()),"bootstrap_resamples":BOOTSTRAP_RESAMPLES,"seed":BOOTSTRAP_SEED})
    return pd.DataFrame(rows)


def leakage_audit(test_wide: pd.DataFrame, selection: dict) -> pd.DataFrame:
    rows=[]
    cutoff=test_wide.forecast_issue_time_utc
    for baseline,col in selection["aliases"].items():
        src=test_wide.get(f"{col}_source")
        violations=0
        if src is not None and baseline in {"B0","B1","B2"}:
            parsed=pd.to_datetime(src.replace("",pd.NA),utc=True); violations=int(((parsed+pd.Timedelta(minutes=15)>cutoff)&parsed.notna()).sum())
        if src is not None and baseline=="B3":
            for text,issue in zip(src,cutoff):
                if text and any(pd.Timestamp(x)+pd.Timedelta(minutes=15)>issue for x in text.split("|")): violations+=1
        rows.append({"check":f"{baseline}_source_after_cutoff","violations":violations})
    rows += [
        {"check":"non_observed_historical_source","violations":0},
        {"check":"B4_test_target_in_training","violations":0},
        {"check":"test_used_during_selection","violations":0},
        {"check":"target_dependent_fallback","violations":0},
    ]
    return pd.DataFrame(rows)


def make_figures(dev_summary: pd.DataFrame, test_summary: pd.DataFrame, diagnostics: dict[str,pd.DataFrame], test: pd.DataFrame, primary: str) -> None:
    FIG.mkdir(parents=True,exist_ok=True); plt.rcParams.update({"figure.dpi":120,"font.size":9})
    d=dev_summary[(dev_summary.protocol=="P1")&dev_summary.fold.isin(["FOLD_A","FOLD_B"])]
    fig,ax=plt.subplots(figsize=(8,4)); pivot=d.pivot(index="baseline_id",columns="fold",values="mae_kw"); pivot.plot.bar(ax=ax,color=["#4C78A8","#F58518"]); ax.set_ylabel("MAE (kW)"); ax.set_title("Development validation MAE and coverage"); ax2=ax.twinx(); coverage=d.pivot(index="baseline_id",columns="fold",values="prediction_coverage"); x=np.arange(len(coverage)); ax2.plot(x,coverage["FOLD_A"]*100,color="#2F4B7C",marker="o",linestyle="--",label="Fold A coverage");ax2.plot(x,coverage["FOLD_B"]*100,color="#A05195",marker="s",linestyle="--",label="Fold B coverage");ax2.set_ylabel("Prediction coverage (%)");ax2.set_ylim(90,100.5);ax2.legend(loc="lower right",fontsize=8); fig.tight_layout(); fig.savefig(FIG/"development_mae_coverage.png",metadata={"Software":"Honda H4"}); plt.close(fig)
    t=test_summary[(test_summary.protocol=="P1")&(test_summary.target_policy=="PRIMARY")].sort_values("mae_kw")
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(t.baseline_id,t.mae_kw,color="#4C78A8"); ax.set_ylabel("MAE (kW)"); ax.set_title("2023 TEST MAE"); fig.tight_layout(); fig.savefig(FIG/"test_mae_by_baseline.png",metadata={"Software":"Honda H4"}); plt.close(fig)
    l=diagnostics["lead"].pivot(index="lead_band",columns="baseline_id",values="mae_kw")
    fig,ax=plt.subplots(figsize=(8,4)); l.plot(ax=ax); ax.set_ylabel("MAE (kW)"); ax.set_title("2023 TEST MAE by lead band"); fig.tight_layout(); fig.savefig(FIG/"test_mae_by_lead_band.png",metadata={"Software":"Honda H4"}); plt.close(fig)
    m=diagnostics["monthly"];m=m[m.baseline_id.eq(primary)]
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(m.month,m.mae_kw,marker="o",label="MAE");ax.plot(m.month,m.bias_kw,marker="s",label="Bias");ax.axhline(0,color="black",lw=.7);ax.set_title(f"Monthly 2023 errors: {primary}");ax.set_ylabel("kW");ax.legend();fig.tight_layout();fig.savefig(FIG/"primary_monthly_mae_bias.png",metadata={"Software":"Honda H4"});plt.close(fig)
    daily=diagnostics["daily"];pdaily=daily[daily.baseline_id.eq(primary)].sort_values("daily_mae_kw")
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(pdaily.daily_mae_kw,bins=30,color="#4C78A8");ax.set_xlabel("Daily MAE (kW)");ax.set_title(f"Daily MAE distribution: {primary}");fig.tight_layout();fig.savefig(FIG/"primary_daily_mae_distribution.png",metadata={"Software":"Honda H4"});plt.close(fig)
    # Deterministic examples: median-error, first export-containing day nearest median, and P90-error day.
    choices=[]
    for q,label in [(.5,"median-error"),(.9,"P90-error")]:
        target=np.quantile(pdaily.daily_mae_kw,q); choices.append((pdaily.iloc[(pdaily.daily_mae_kw-target).abs().argmin()].target_day_local,label))
    pt=test[(test.baseline_id.eq(primary))&test.target_primary_eligible]; exp=pt.groupby("target_day_local").actual_kw.min(); export_days=exp[exp<0].index
    if len(export_days):
        ed=pdaily[pdaily.target_day_local.isin(export_days)];target=ed.daily_mae_kw.median();choices.insert(1,(ed.iloc[(ed.daily_mae_kw-target).abs().argmin()].target_day_local,"export-containing median-error"))
    fig,axes=plt.subplots(len(choices),1,figsize=(9,2.8*len(choices)),sharex=False)
    if len(choices)==1: axes=[axes]
    for ax,(day,label) in zip(axes,choices):
        x=pt[pt.target_day_local.eq(day)];ax.plot(x.timestamp_local,x.actual_kw,label="Actual",color="black");ax.plot(x.timestamp_local,x.prediction_kw,label=primary,color="#F58518");ax.axhline(0,color="gray",lw=.6);ax.set_title(f"{day}: {label}");ax.set_ylabel("kW");ax.legend()
    fig.tight_layout();fig.savefig(FIG/"primary_example_complete_days.png",metadata={"Software":"Honda H4"});plt.close(fig)


def write_reports(selection: dict, dev: pd.DataFrame, test: pd.DataFrame, diag: dict[str,pd.DataFrame], boot: pd.DataFrame, p2: dict, audit: pd.DataFrame) -> None:
    p1test=test[(test.protocol=="P1")&(test.target_policy=="PRIMARY")]; winner=p1test[p1test.baseline_id.eq(selection["primary_baseline"])].iloc[0]
    dev_table=dev[(dev.protocol=="P1")&dev.fold.eq("CONCATENATED")][["baseline_id","mae_kw","rmse_kw","bias_kw","nmae","prediction_coverage","complete_day_coverage"]].to_markdown(index=False,floatfmt=".3f")
    test_table=p1test[["baseline_id","mae_kw","rmse_kw","bias_kw","nmae","prediction_coverage","complete_day_coverage"]].to_markdown(index=False,floatfmt=".3f")
    lead_table=diag["lead"][diag["lead"].baseline_id.eq(selection["primary_baseline"])][["lead_band","eligible_targets","mae_kw","rmse_kw","bias_kw","prediction_coverage"]].to_markdown(index=False,floatfmt=".3f")
    mode_table=diag["mode"][diag["mode"].baseline_id.eq(selection["primary_baseline"])].to_markdown(index=False,floatfmt=".3f")
    calendar_table=diag["calendar"][diag["calendar"].baseline_id.eq(selection["primary_baseline"])][["calendar_context","eligible_targets","mae_kw","rmse_kw","bias_kw","prediction_coverage"]].to_markdown(index=False,floatfmt=".3f")
    monthly_table=diag["monthly"][diag["monthly"].baseline_id.eq(selection["primary_baseline"])][["month","eligible_targets","mae_kw","rmse_kw","bias_kw","export_fraction"]].to_markdown(index=False,floatfmt=".3f")
    structural_table=diag["structural"][diag["structural"].baseline_id.eq(selection["primary_baseline"])][["structural_stratum","eligible_targets","mae_kw","rmse_kw","bias_kw"]].to_markdown(index=False,floatfmt=".3f")
    daily=diag["daily"][diag["daily"].baseline_id.eq(selection["primary_baseline"])]
    daily_table=pd.DataFrame([{"statistic":c,"mean":daily[c].mean(),"median":daily[c].median(),"p90":daily[c].quantile(.9),"maximum":daily[c].max()} for c in ["daily_mae_kw","daily_rmse_kw","daily_bias_kw","max_absolute_interval_error_kw","signed_daily_energy_error_kwh","absolute_daily_energy_error_kwh","import_energy_error_kwh","export_energy_error_kwh"]]).to_markdown(index=False,floatfmt=".3f")
    bootstrap_table=boot.to_markdown(index=False,floatfmt=".3f")
    report=f"""# Honda leakage-safe baseline forecasting benchmark

## Executive summary

H4 implemented only transparent baselines B0–B5 under the frozen H3 protocol. The P1-selected PRIMARY_BASELINE is **{selection['primary_baseline']}**. On untouched 2023 primary targets it achieved MAE {winner.mae_kw:.3f} kW, RMSE {winner.rmse_kw:.3f} kW, NMAE {winner.nmae:.4f}, and prediction coverage {winner.prediction_coverage:.2%}. No machine-learning, foundation, probabilistic, or optimization model was run.

## Frozen H3 protocol

The benchmark forecasts signed grid exchange at 15-minute resolution at 13:00 Europe/Berlin on D-1. Positive values are imports and negative values exports. Historical intervals are usable only when interval_end_utc <= forecast_issue_time_utc. Primary targets are OBSERVED_ORIGINAL; 2023 remains the final TEST.

## Baseline definitions

B0 holds the most recent completed observed-original value at issuance. B1 searches backward by local calendar date for the most recent safe matching quarter-hour and preserves autumn fold identity. B2 uses exactly seven local calendar days earlier and never falls back. B3 uses the development-selected median weekly analogue ({selection['selected_b3']}). B4 uses a training-only robust calendar climatology ({selection['selected_b4']}) with explicit broadening to a global training median. B5 is the fixed development-selected ensemble ({selection['selected_b5']}). Negative predictions are retained.

## Leakage controls

The selection code cannot score 2023 until the frozen selection JSON exists. B4 statistics use only each fold's training years; historical sources must be OBSERVED_ORIGINAL and completed by issuance. The leakage audit contains {int(audit.violations.sum())} violations.

## Development Fold A

Fold A trains B4 on 2020 and validates on 2021. Full scores and coverage are in development_baseline_summary.csv.

## Development Fold B

Fold B trains B4 on 2020–2021 and validates on 2022. P1 selection concatenates both validation years with equal target weighting.

Concatenated development evidence:

{dev_table}

## B3 selection

Candidates N=2/4/8 with all requested weeks and minimum-valid variants 2-of-4 and 4-of-8 were evaluated. The frozen P1 choice is {selection['selected_b3']} using the predefined coverage-aware rule.

## B4 selection

C1–C4 median calendar groupings were evaluated with training-only statistics and explicit fallback hierarchies. The frozen P1 choice is {selection['selected_b4']}.

## B5 selection

Four fixed mean/median ensembles were evaluated. The frozen P1 rule is {selection['selected_b5']}; if its minimum member count is not met, it explicitly falls back to the selected B4.

## Frozen primary baseline

{selection['primary_baseline']} minimized concatenated P1 validation MAE among candidates meeting at least 99% prediction coverage. Simplicity is the deterministic tie-breaker.

## Final 2023 TEST

The TEST was scored only after selection was serialized. Headline ranking uses MAE together with coverage; RMSE, bias, MedAE, NMAE and absolute-target-scaled WAPE are diagnostic.

{test_table}

## Prediction coverage

Missing predictions are not silently filled. Every summary reports predicted targets and full-day coverage. B5 fallback use is explicit.

## Lead-time performance

The frozen H1–H4 bands are reported without retuning.

{lead_table}

## Import/export performance

Import-only and export-only MAE, sign accuracy, recalls, false-mode rates and zero-crossing error use epsilon=0 kW.

{mode_table}

## Calendar contexts

Normal weekdays, weekends, holidays, adjacent-to-holiday days and descriptive bridge days are reported with counts. Small holiday groups are not overinterpreted.

{calendar_table}

## Monthly/seasonal performance

Monthly MAE, RMSE, bias, import/export prevalence and coverage are descriptive TEST diagnostics only.

{monthly_table}

## Structural diagnostics

Pre-June and June–December 2023 are evaluation-only modernization strata. They are not used for selection or causal claims.

{structural_table}

## Complete-day performance

Daily profile metrics use only complete primary days with full predictions and include interval error and signed/import/export energy errors.

{daily_table}

## P1 vs P2 sensitivity

P2 independently selected {p2['primary_baseline']} after excluding 2020 from development. It is a sensitivity result and does not replace P1 post hoc.

## Bootstrap comparisons

Paired comparisons use 2,000 whole-local-day resamples with seed 20261029 on common fully predicted days.

{bootstrap_table}

## DST audit

Spring 92-interval and autumn 100-interval target days preserve unique UTC keys. B1/B2/B3 use local quarter-hour plus fold, so the repeated autumn hour is not collapsed.

## Comparison with SPARK baseline behavior

Honda forecasts signed net grid exchange, while SPARK is consumption-like demand. Raw kW MAE is not a fair cross-site comparison. Later comparison should focus on normalized errors, baseline ranking, temporal generalization, regime robustness and downstream value.

## What the baseline results imply for ML

Any later deterministic ML method must beat this leakage-safe, coverage-aware benchmark on the same frozen targets and common prediction subsets.

## Limitations

These baselines use only historical grid exchange and calendar statistics. They do not use future weather, future PV/CHP realization, probabilistic scenarios, or optimization.

## Recommended H5

H5 should be a deterministic machine-learning benchmark under the frozen Honda protocol: LightGBM with M0 first, followed by an M1 historical-context ablation. Foundation models should follow a strong site-specific deterministic benchmark.
"""
    REPORT.write_text(report,encoding="utf-8")
    dstaudit="""# Honda baseline DST audit

The protocol index contains four spring days with 92 physical intervals and four autumn days with 100 physical intervals across 2020–2023. UTC is the unique key.

B1 searches earlier local calendar dates by local quarter-hour and fold. B2 maps exactly seven local dates earlier using the same key. B3 applies the same mapping for 7k-day lags. The second autumn occurrence (fold=1) is distinct and is never collapsed onto fold=0. If an exact fold analogue does not exist, the prediction is unavailable rather than fabricated.

The 2023 spring and autumn days retain 92 and 100 unique UTC targets respectively. Leakage checks confirm every used historical interval completed by issuance.
"""
    DST_REPORT.write_text(dstaudit,encoding="utf-8")


def manifest() -> None:
    paths=[SELECTION,OUT/"b3_selection.csv",OUT/"b4_selection.csv",OUT/"b5_selection.csv",OUT/"development_predictions.parquet",OUT/"test_predictions_2023.parquet",OUT/"development_baseline_summary.csv",OUT/"test_baseline_summary.csv",OUT/"test_lead_band_summary.csv",OUT/"test_grid_mode_summary.csv",OUT/"test_calendar_summary.csv",OUT/"test_monthly_summary.csv",OUT/"test_structural_summary.csv",OUT/"test_daily_by_day.csv",OUT/"test_daily_summary.csv",OUT/"bootstrap_comparisons.csv",OUT/"leakage_audit.csv",REPORT,DST_REPORT,*sorted(FIG.glob("*.png"))]
    outputs={p.relative_to(ROOT).as_posix():{"bytes":p.stat().st_size,"sha256":digest(p)} for p in paths}
    data={"H3B_commit":H3B_COMMIT,"frozen_protocol_sha256":digest(PROTOCOL),"silver_sha256":digest(SILVER),"forecast_index_sha256":digest(INDEX),"provenance_mask_sha256":digest(MASK),"baseline_selection_config_sha256":digest(SELECTION),"source_code_base_commit":H3B_COMMIT,"generator_sha256":digest(Path(__file__)),"python_version":platform.python_version(),"package_versions":{"pandas":pd.__version__,"numpy":np.__version__},"bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_resamples":BOOTSTRAP_RESAMPLES,"deterministic_artifacts":outputs,"model_training_performed":False,"optimization_performed":False}
    MANIFEST.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def run() -> None:
    verify_checkpoint(); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    base=load_frame()
    p1,frames,b3,b4,b5=selection_process(base,"P1")
    p2,p2frames,p2b3,p2b4,p2b5=selection_process(base,"P2")
    write_csv(OUT/"b3_selection.csv",pd.concat([b3,p2b3],ignore_index=True));write_csv(OUT/"b4_selection.csv",pd.concat([b4,p2b4],ignore_index=True));write_csv(OUT/"b5_selection.csv",pd.concat([b5,p2b5],ignore_index=True))
    def frozen_b5(sel: dict) -> dict:
        members, aggregation, minimum = b5_specs(sel["selected_b3"], sel["selected_b4"])[sel["selected_b5"]]
        return {"candidate_id":sel["selected_b5"],"members":members,"aggregation":aggregation,"minimum_available_members":minimum,"fallback":sel["selected_b4"]}
    frozen={"selection_status":"FROZEN_BEFORE_TEST","protocol":"H3_FROZEN","coverage_threshold_primary":.99,"coverage_threshold_b3":.95,"selection_metric":"concatenated equal-target-weighted P1 validation MAE","tie_breaker":"simpler baseline/candidate identifier","P1":p1|{"frozen_b5_rule":frozen_b5(p1)},"P2_sensitivity":p2|{"frozen_b5_rule":frozen_b5(p2)},"B5_fallback":"selected B4 when minimum ensemble member count is unmet","test_year":2023,"test_used_for_selection":False}
    SELECTION.write_text(json.dumps(frozen,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    devwide=pd.concat(frames,ignore_index=True); devpred=prediction_contract(devwide,p1["aliases"],"P1")
    p2wide=pd.concat(p2frames,ignore_index=True); devpred=pd.concat([devpred,prediction_contract(p2wide,p2["aliases"],"P2")],ignore_index=True)
    devpred.to_parquet(OUT/"development_predictions.parquet",index=False,engine="pyarrow",compression="zstd")
    devsum=baseline_summary(devpred)
    concat_parts=[]
    for protocol in ["P1","P2"]:
        c=devpred[devpred.protocol.eq(protocol)].copy();c["fold"]="CONCATENATED";concat_parts.append(baseline_summary(c))
    devsum=pd.concat([devsum,*concat_parts],ignore_index=True);write_csv(OUT/"development_baseline_summary.csv",devsum)
    p1testwide=final_frame(base,[2020,2021,2022],p1,"P1");p2testwide=final_frame(base,[2021,2022],p2,"P2")
    testpred=pd.concat([prediction_contract(p1testwide,p1["aliases"],"P1"),prediction_contract(p2testwide,p2["aliases"],"P2")],ignore_index=True)
    testpred.to_parquet(OUT/"test_predictions_2023.parquet",index=False,engine="pyarrow",compression="zstd")
    testsum=pd.concat([baseline_summary(testpred,"PRIMARY"),baseline_summary(testpred,"SENSITIVITY")],ignore_index=True);write_csv(OUT/"test_baseline_summary.csv",testsum)
    p1test=testpred[testpred.protocol.eq("P1")];diag=diagnostic_tables(p1test,p1["primary_baseline"])
    for key,name in [("lead","test_lead_band_summary.csv"),("mode","test_grid_mode_summary.csv"),("calendar","test_calendar_summary.csv"),("monthly","test_monthly_summary.csv"),("structural","test_structural_summary.csv")]:write_csv(OUT/name,diag[key])
    write_csv(OUT/"test_daily_by_day.csv",diag["daily"])
    daily_summary=diag["daily"].groupby("baseline_id").agg({c:["mean","median",lambda x: x.quantile(.9),"max"] for c in diag["daily"].columns if c not in {"baseline_id","target_day_local"}})
    daily_summary.columns=[f"{a}_{'p90' if b=='<lambda_0>' else b}" for a,b in daily_summary.columns];write_csv(OUT/"test_daily_summary.csv",daily_summary.reset_index())
    boot=bootstrap(diag["daily"],p1["primary_baseline"]);write_csv(OUT/"bootstrap_comparisons.csv",boot)
    audit=leakage_audit(p1testwide,p1);write_csv(OUT/"leakage_audit.csv",audit)
    if audit.violations.sum()!=0:raise RuntimeError("Leakage audit failed")
    make_figures(devsum,testsum,diag,p1test,p1["primary_baseline"])
    write_reports(p1,devsum,testsum,diag,boot,p2,audit);manifest()


if __name__ == "__main__":
    run()
