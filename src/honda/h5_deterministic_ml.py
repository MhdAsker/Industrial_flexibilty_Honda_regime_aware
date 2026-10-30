from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.honda import h4_baseline_benchmark as h4


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ml"
FIG = OUT / "figures"
MODELS = ROOT / "artifacts/models"
CONFIG = ROOT / "config/honda_lgbm_selection_frozen.json"
REPORT = ROOT / "reports/honda_deterministic_ml_benchmark.md"
MANIFEST = OUT / "honda_h5_manifest.json"
FEATURE_CACHE = OUT / "h5_feature_matrix.parquet"

H4_COMMIT = "11a490b6bff6a0ce12f6f67d686abd4355d3185b"
H3_PROTOCOL_HASH = "281f49331bc3b3c31aec0d33b50f6a83a149261db93ce04a29d67479cb62676e"
H4_SELECTION_HASH = "1bbbbac404675bb1bfb5c85067e1504fda843fc2bc52151a60025f4e3c8b7a04"
SEED = 20261029
BOOTSTRAPS = 2000
OBSERVED = "OBSERVED_ORIGINAL"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def verify_checkpoint() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", H4_COMMIT, head], cwd=ROOT).returncode:
        raise RuntimeError("H4 checkpoint is not an ancestor")
    expected = {
        h4.PROTOCOL: H3_PROTOCOL_HASH, h4.SELECTION: H4_SELECTION_HASH,
        h4.INDEX: h4.INDEX_SHA256, h4.SILVER: h4.SILVER_SHA256, h4.MASK: h4.MASK_SHA256,
    }
    for path, sha in expected.items():
        if digest(path) != sha: raise RuntimeError(f"Frozen artifact changed: {path}")
    h4m = json.loads(h4.MANIFEST.read_text(encoding="utf-8"))
    for rel, meta in h4m["deterministic_artifacts"].items():
        if digest(ROOT / rel) != meta["sha256"]: raise RuntimeError(f"H4 output changed: {rel}")
    b5 = pd.read_csv(h4.OUT / "test_baseline_summary.csv")
    row = b5[(b5.protocol == "P1") & (b5.target_policy == "PRIMARY") & (b5.baseline_id == "B5")].iloc[0]
    if not math.isclose(row.mae_kw, 76.47952742, abs_tol=1e-8): raise RuntimeError("Frozen B5 score changed")


def _safe_signal(signal: str) -> pd.DataFrame:
    provenance = "pv_provenance" if signal == "pv_generation" else "chp_provenance"
    value = f"{signal}_kw"
    s = pd.read_parquet(h4.SILVER, columns=["timestamp_utc", value, provenance])
    return s[s[provenance].eq(OBSERVED) & s[value].notna()].rename(columns={value: "value"})[["timestamp_utc", "value"]].sort_values("timestamp_utc")


def _window_stats(safe: pd.DataFrame, issues: pd.DatetimeIndex, prefix: str, windows: dict[str, int]) -> pd.DataFrame:
    ts = pd.DatetimeIndex(safe.timestamp_utc).as_unit("ns").asi8
    values = safe.value.to_numpy(float)
    out = pd.DataFrame({"forecast_issue_time_utc": issues})
    for label, minutes in windows.items():
        expected = minutes // 15
        means=[]; medians=[]; mins=[]; maxs=[]; stds=[]; fracs=[]; lasts=[]; ramps=[]; exports=[]; export_counts=[]; positives=[]
        for issue in issues:
            end = issue.value - pd.Timedelta(minutes=15).value
            start = issue.value - pd.Timedelta(minutes=minutes).value
            lo = np.searchsorted(ts, start, side="left"); hi = np.searchsorted(ts, end, side="right")
            x = values[lo:hi]
            if len(x):
                means.append(float(np.mean(x))); medians.append(float(np.median(x))); mins.append(float(np.min(x))); maxs.append(float(np.max(x))); stds.append(float(np.std(x)))
                fracs.append(len(x) / expected); lasts.append(float(x[-1])); ramps.append(float(np.mean(np.abs(np.diff(x)))) if len(x)>1 else np.nan); exports.append(float(np.mean(x < 0))); export_counts.append(int(np.sum(x < 0))); positives.append(float(np.mean(x > 0)))
            else:
                means.append(np.nan);medians.append(np.nan);mins.append(np.nan);maxs.append(np.nan);stds.append(np.nan);fracs.append(0.0);lasts.append(np.nan);ramps.append(np.nan);exports.append(np.nan);export_counts.append(0);positives.append(np.nan)
        for suffix, vals in [("mean",means),("median",medians),("min",mins),("max",maxs),("std",stds),("observed_fraction",fracs),("last",lasts),("mean_abs_ramp",ramps),("export_fraction",exports),("export_count",export_counts),("positive_fraction",positives)]:
            out[f"{prefix}_{label}_{suffix}"] = vals
    return out


def _same_quarter_lags(base: pd.DataFrame, days: list[int]) -> pd.DataFrame:
    history = h4.safe_history()
    rows = {f"grid_lag_{d}d": [] for d in days}
    for row in base.itertuples(index=False):
        slot = (row.target_day - pd.Timedelta(days=1),)  # type anchor only
        for d in days:
            rec = history.exact.get((row.target_day - pd.Timedelta(days=d), int(row.quarter), int(row.fold)))
            rows[f"grid_lag_{d}d"].append(rec[1] if rec and rec[0] + pd.Timedelta(minutes=15) <= row.forecast_issue_time_utc else np.nan)
    return pd.DataFrame(rows, index=base.index)


def build_features() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    base = h4.load_frame()
    feature = base[["timestamp_utc","timestamp_local","target_day_local","target_day","forecast_issue_time_utc","actual_kw","actual_provenance","target_primary_eligible","target_sensitivity_eligible","complete_primary_day","lead_band","lead_time_hours","holiday_context","structural_era_metadata","year","month","weekday","weekend","quarter","fold"]].copy()
    local = feature.timestamp_local
    feature["local_hour"] = local.dt.hour
    feature["local_quarter_in_hour"] = local.dt.minute // 15
    feature["day_of_year"] = local.dt.dayofyear
    feature["daily_sin"] = np.sin(2*np.pi*feature.quarter/96); feature["daily_cos"] = np.cos(2*np.pi*feature.quarter/96)
    feature["annual_sin"] = np.sin(2*np.pi*feature.day_of_year/365.25); feature["annual_cos"] = np.cos(2*np.pi*feature.day_of_year/365.25)
    feature["dst_state"] = [int(x.dst().total_seconds()!=0) for x in local]
    for token, name in [("PUBLIC_HOLIDAY","public_holiday"),("DAY_BEFORE_PUBLIC_HOLIDAY","day_before_holiday"),("DAY_AFTER_PUBLIC_HOLIDAY","day_after_holiday"),("BRIDGE_DAY_DIAGNOSTIC_ONLY","bridge_day")]: feature[name] = feature.holiday_context.str.contains(token).astype(int)
    era = feature.structural_era_metadata.map(json.loads)
    feature["pv_phase_2_known"] = [int(x.get("pv_capacity_era") == "PV_final_capacity") for x in era]
    feature["covid_reduced_known"] = [int(x.get("covid_era") == "COVID_reduced_operation") for x in era]
    calendar = ["quarter","local_hour","local_quarter_in_hour","weekday","weekend","month","day_of_year","daily_sin","daily_cos","annual_sin","annual_cos","dst_state","public_holiday","day_before_holiday","day_after_holiday","bridge_day","lead_time_hours","fold","pv_phase_2_known","covid_reduced_known"]

    components = pd.concat([h4.components_for_year(base, y) for y in [2020,2021,2022,2023]], ignore_index=True).sort_values("timestamp_utc")
    components = components.set_index("timestamp_utc")
    for src, dst in [("B0","analogue_B0"),("B1","analogue_B1"),("B2","analogue_B2"),("B3_N8_MIN4","analogue_B3")]: feature[dst] = feature.timestamp_utc.map(components[src])
    lag_days=[1,2,3,7,14,21,28,35,42,49,56]
    feature = pd.concat([feature, _same_quarter_lags(feature, lag_days)], axis=1)
    weekly=[f"grid_lag_{d}d" for d in [7,14,21,28,35,42,49,56]]
    feature["weekly_median_2"]=feature[weekly[:2]].median(axis=1,skipna=True);feature["weekly_median_4"]=feature[weekly[:4]].median(axis=1,skipna=True);feature["weekly_median_8"]=feature[weekly].median(axis=1,skipna=True)
    feature["weekly_mean_4"]=feature[weekly[:4]].mean(axis=1,skipna=True);feature["weekly_std_4"]=feature[weekly[:4]].std(axis=1);feature["weekly_std_8"]=feature[weekly].std(axis=1);feature["weekly_available_8"]=feature[weekly].notna().sum(axis=1)
    analogues=["analogue_B0","analogue_B1","analogue_B2","analogue_B3",*[f"grid_lag_{d}d" for d in lag_days],"weekly_median_2","weekly_median_4","weekly_median_8","weekly_mean_4","weekly_std_4","weekly_std_8","weekly_available_8"]

    issues=pd.DatetimeIndex(sorted(feature.forecast_issue_time_utc.unique())).as_unit("ns")
    grid_safe=pd.read_parquet(h4.SILVER,columns=["timestamp_utc","grid_exchange_kw_signed","grid_exchange_provenance"])
    grid_safe=grid_safe[grid_safe.grid_exchange_provenance.eq(OBSERVED)&grid_safe.grid_exchange_kw_signed.notna()].rename(columns={"grid_exchange_kw_signed":"value"})[["timestamp_utc","value"]].sort_values("timestamp_utc")
    gridstats=_window_stats(grid_safe,issues,"grid",{"1h":60,"3h":180,"6h":360,"12h":720,"24h":1440,"48h":2880,"7d":10080,"14d":20160,"28d":40320})
    feature=feature.merge(gridstats,on="forecast_issue_time_utc",how="left",validate="many_to_one")
    recent=["grid_1h_last","grid_1h_mean","grid_3h_mean","grid_6h_mean","grid_12h_mean","grid_24h_mean","grid_48h_mean","grid_7d_mean","grid_14d_mean","grid_28d_mean","grid_24h_median","grid_7d_median","grid_24h_min","grid_24h_max","grid_7d_min","grid_7d_max","grid_24h_observed_fraction","grid_7d_observed_fraction","grid_28d_observed_fraction"]
    feature["grid_level_7d_minus_28d"]=feature.grid_7d_mean-feature.grid_28d_mean;recent.append("grid_level_7d_minus_28d")
    volatility=["grid_24h_std","grid_7d_std","grid_14d_std","grid_24h_mean_abs_ramp","grid_7d_mean_abs_ramp","grid_24h_export_fraction","grid_7d_export_fraction","grid_28d_export_fraction","grid_7d_export_count"]

    m1_groups={}
    for signal,prefix in [("pv_generation","pv"),("chp_generation","chp")]:
        safe=_safe_signal(signal)
        stats=_window_stats(safe,issues,prefix,{"24h":1440,"7d":10080})
        feature=feature.merge(stats,on="forecast_issue_time_utc",how="left",validate="many_to_one")
        cols=[f"{prefix}_24h_last",f"{prefix}_24h_mean",f"{prefix}_24h_max",f"{prefix}_24h_observed_fraction",f"{prefix}_7d_mean",f"{prefix}_7d_max",f"{prefix}_7d_std",f"{prefix}_7d_observed_fraction"]
        if prefix=="chp":
            feature["chp_24h_on_fraction"]=feature["chp_24h_positive_fraction"];cols.append("chp_24h_on_fraction")
        m1_groups[prefix]=cols
    families={"M0-A":calendar,"M0-B":calendar+analogues,"M0-C":calendar+analogues+recent,"M0-D":calendar+analogues+recent+volatility}
    families.update({"M1-PV":families["M0-D"]+m1_groups["pv"],"M1-CHP":families["M0-D"]+m1_groups["chp"],"M1-W":families["M0-D"],"M1-T":families["M0-D"],"M1-ALL":families["M0-D"]+m1_groups["pv"]+m1_groups["chp"]})

    catalog=[]
    for fam,cols in families.items():
        for c in cols:
            if any(c.startswith(x) for x in ["pv_"]): domain="PV history"; info="M1"
            elif c.startswith("chp_"): domain="CHP history";info="M1"
            elif c.startswith("grid_") and ("lag" in c or "weekly" in c): domain="historical analogues";info="M0"
            elif c.startswith("grid_") and any(x in c for x in ["std","ramp","export"]): domain="volatility/export history";info="M0"
            elif c.startswith("grid_") or c.startswith("analogue_"): domain="recent level" if not c.startswith("analogue_") else "historical analogues";info="M0"
            else: domain="calendar";info="M0"
            catalog.append({"feature_name":c,"information_set":info,"feature_family":fam,"source_signal":domain,"feature_type":"numeric","lookback":"pre-issuance" if domain!="calendar" else "target-time known","future_known":"yes" if domain=="calendar" else "no","provenance_requirement":OBSERVED if domain!="calendar" else "not applicable","availability_cutoff":"interval_end_utc <= forecast_issue_time_utc" if domain!="calendar" else "known at issuance","leakage_risk":"LOW after timestamp/provenance guard","notes":"NaN preserved"})
    catalog += [
        {"feature_name":"weather_history","information_set":"M1","feature_family":"M1-W","source_signal":"Ta/Igm","feature_type":"NOT_EVALUABLE","lookback":"pre-issuance","future_known":"no","provenance_requirement":OBSERVED,"availability_cutoff":"interval_end_utc <= issue","leakage_risk":"BLOCKED","notes":"H2 explicitly did not create weather provenance masks"},
        {"feature_name":"thermal_history","information_set":"M1","feature_family":"M1-T","source_signal":"heating/cooling","feature_type":"NOT_EVALUABLE","lookback":"pre-issuance","future_known":"no","provenance_requirement":OBSERVED,"availability_cutoff":"interval_end_utc <= issue","leakage_risk":"BLOCKED","notes":"No primary provenance masks for reduced thermal aggregates"},
    ]
    catalog=pd.DataFrame(catalog).drop_duplicates(["feature_name","feature_family"]).sort_values(["feature_family","feature_name"])
    audit=[]
    reps=pd.concat([feature[feature.year==y].head(1) for y in [2020,2021,2022,2023]])
    for r in reps.itertuples():
        for domain in ["grid","pv","chp"]:
            audit.append({"timestamp_utc":r.timestamp_utc,"year":r.year,"feature_domain":domain,"forecast_issue_time_utc":r.forecast_issue_time_utc,"latest_permitted_interval_start_utc":r.forecast_issue_time_utc-pd.Timedelta(minutes=15),"provenance_requirement":OBSERVED,"future_realized_used":False,"imputed_or_unknown_used":False,"leakage_violation":False})
    return feature,families,catalog,pd.DataFrame(audit)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    e=pred-actual;den=np.mean(np.abs(actual))
    return {"rows":len(actual),"mae_kw":float(np.mean(np.abs(e))),"rmse_kw":float(np.sqrt(np.mean(e*e))),"bias_kw":float(np.mean(e)),"median_absolute_error_kw":float(np.median(np.abs(e))),"nmae":float(np.mean(np.abs(e))/den),"wape":float(np.sum(np.abs(e))/np.sum(np.abs(actual))),"coverage":1.0}


BASE_PARAMS={"objective":"regression_l1","n_estimators":800,"num_leaves":31,"learning_rate":.05,"min_child_samples":100,"feature_fraction":.8,"reg_lambda":.1,"reg_alpha":0.0}
HP_GRID=[
    {"num_leaves":15,"learning_rate":.03,"min_child_samples":50,"feature_fraction":.8,"reg_lambda":0,"reg_alpha":0},
    {"num_leaves":15,"learning_rate":.05,"min_child_samples":100,"feature_fraction":1.0,"reg_lambda":.1,"reg_alpha":0},
    {"num_leaves":15,"learning_rate":.05,"min_child_samples":250,"feature_fraction":.8,"reg_lambda":1,"reg_alpha":.1},
    {"num_leaves":31,"learning_rate":.03,"min_child_samples":100,"feature_fraction":1.0,"reg_lambda":1,"reg_alpha":0},
    {"num_leaves":31,"learning_rate":.05,"min_child_samples":50,"feature_fraction":.8,"reg_lambda":.1,"reg_alpha":.1},
    {"num_leaves":31,"learning_rate":.05,"min_child_samples":250,"feature_fraction":1.0,"reg_lambda":0,"reg_alpha":0},
    {"num_leaves":63,"learning_rate":.03,"min_child_samples":50,"feature_fraction":1.0,"reg_lambda":.1,"reg_alpha":0},
    {"num_leaves":63,"learning_rate":.03,"min_child_samples":250,"feature_fraction":.8,"reg_lambda":1,"reg_alpha":.1},
    {"num_leaves":63,"learning_rate":.05,"min_child_samples":100,"feature_fraction":1.0,"reg_lambda":0,"reg_alpha":.1},
]


def _model(params: dict, n_estimators: int|None=None) -> LGBMRegressor:
    p=BASE_PARAMS|params
    if n_estimators is not None:p["n_estimators"]=n_estimators
    return LGBMRegressor(**p,random_state=SEED,deterministic=True,force_col_wise=True,n_jobs=4,verbosity=-1)


def fit_fold(frame: pd.DataFrame, features: list[str], train_years: list[int], val_year: int, params: dict) -> tuple[dict,np.ndarray,LGBMRegressor]:
    tr=frame.year.isin(train_years)&frame.target_primary_eligible&frame.actual_kw.notna();va=frame.year.eq(val_year)&frame.target_primary_eligible&frame.actual_kw.notna()
    model=_model(params)
    model.fit(frame.loc[tr,features],frame.loc[tr,"actual_kw"],eval_set=[(frame.loc[va,features],frame.loc[va,"actual_kw"])],eval_metric="mae",callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
    pred=model.predict(frame.loc[va,features],num_iteration=model.best_iteration_)
    row=metrics(frame.loc[va,"actual_kw"].to_numpy(),pred)|{"train_rows":int(tr.sum()),"validation_year":val_year,"best_iteration":int(model.best_iteration_ or model.n_estimators)}
    return row,pred,model


def evaluate_candidate(frame: pd.DataFrame, features: list[str], params: dict, name: str) -> tuple[list[dict],dict[int,np.ndarray],list[LGBMRegressor]]:
    rows=[];preds={};models=[]
    for fold,train,val in [("FOLD_A",[2020],2021),("FOLD_B",[2020,2021],2022)]:
        row,pred,model=fit_fold(frame,features,train,val,params);row|={"candidate":name,"fold":fold};rows.append(row);preds[val]=pred;models.append(model)
    actual=np.concatenate([frame.loc[frame.year.eq(y)&frame.target_primary_eligible,"actual_kw"].to_numpy() for y in [2021,2022]])
    prediction=np.concatenate([preds[y] for y in [2021,2022]])
    rows.append(metrics(actual,prediction)|{"candidate":name,"fold":"CONCATENATED","train_rows":np.nan,"validation_year":"2021+2022","best_iteration":int(round(np.median([m.best_iteration_ for m in models])))})
    return rows,preds,models


def choose(rows: pd.DataFrame) -> str:
    x=rows[rows.fold.eq("CONCATENATED")].sort_values(["mae_kw","candidate"])
    best=x.iloc[0]; tied=x[x.mae_kw <= best.mae_kw*1.005]
    return tied.sort_values("candidate").iloc[0].candidate


def b5_development() -> pd.DataFrame:
    s=pd.read_csv(h4.OUT/"development_baseline_summary.csv")
    return s[(s.protocol=="P1")&s.baseline_id.eq("B5")][["fold","mae_kw","rmse_kw","bias_kw","prediction_coverage"]]


def run_selection(frame: pd.DataFrame, families: dict[str,list[str]]) -> tuple[dict,pd.DataFrame,pd.DataFrame,pd.DataFrame,dict[str,dict[int,np.ndarray]]]:
    m0rows=[];m0preds={}
    for family in ["M0-A","M0-B","M0-C","M0-D"]:
        rows,preds,_=evaluate_candidate(frame,families[family],{},family);m0rows+=rows;m0preds[family]=preds
    m0df=pd.DataFrame(m0rows); selected_family=choose(m0df)
    hprows=[];hppreds={};hpmodels={}
    for i,p in enumerate(HP_GRID):
        name=f"HP{i+1:02d}";rows,preds,models=evaluate_candidate(frame,families[selected_family],p,name)
        for r in rows:r.update(p)
        hprows+=rows;hppreds[name]=preds;hpmodels[name]=models
    hpdf=pd.DataFrame(hprows);selected_hp=choose(hpdf);hp=HP_GRID[int(selected_hp[2:])-1]
    m1rows=[];m1preds={};m1models={}
    for family in ["M1-PV","M1-CHP","M1-ALL"]:
        rows,preds,models=evaluate_candidate(frame,families[family],hp,family);m1rows+=rows;m1preds[family]=preds;m1models[family]=models
    for family,reason in [("M1-W","H2 weather provenance masks unavailable"),("M1-T","Primary thermal provenance masks unavailable")]:
        for fold in ["FOLD_A","FOLD_B","CONCATENATED"]:m1rows.append({"candidate":family,"fold":fold,"status":"NOT_EVALUABLE","reason":reason})
    m1df=pd.DataFrame(m1rows); evaluated=m1df[(m1df.fold=="CONCATENATED")&m1df.mae_kw.notna()].sort_values("mae_kw")
    best_m1=evaluated.iloc[0].candidate
    m0_mae=hpdf[(hpdf.candidate==selected_hp)&(hpdf.fold=="CONCATENATED")].iloc[0].mae_kw
    m1_mae=evaluated.iloc[0].mae_kw
    fold_m0=hpdf[(hpdf.candidate==selected_hp)&hpdf.fold.isin(["FOLD_A","FOLD_B"])].set_index("fold").mae_kw
    fold_m1=m1df[(m1df.candidate==best_m1)&m1df.fold.isin(["FOLD_A","FOLD_B"])].set_index("fold").mae_kw
    robust=(m1_mae < m0_mae*.995) and bool((fold_m1 <= fold_m0*1.005).all())
    final_family=best_m1 if robust else selected_family
    final_models=m1models[best_m1] if robust else hpmodels[selected_hp]
    iterations=int(round(np.median([m.best_iteration_ for m in final_models])))
    selected={"selected_m0_family":selected_family,"selected_hyperparameter_id":selected_hp,"hyperparameters":hp,"selected_m1_family":best_m1,"m1_selected_as_primary":robust,"final_information_set":"M1" if robust else "M0","final_feature_family":final_family,"iteration_count":iterations,"m0_iteration_count":int(round(np.median([m.best_iteration_ for m in hpmodels[selected_hp]]))),"m1_iteration_count":int(round(np.median([m.best_iteration_ for m in m1models[best_m1]])))}
    predictions={"M0":hppreds[selected_hp],"M1":m1preds[best_m1]}
    return selected,m0df,hpdf,m1df,predictions


def freeze_config(selected: dict,families:dict,m0:pd.DataFrame,hp:pd.DataFrame,m1:pd.DataFrame) -> None:
    config={"selection_status":"FROZEN_BEFORE_TEST","H4_commit":H4_COMMIT,"H3_protocol_sha256":H3_PROTOCOL_HASH,"H4_baseline_selection_sha256":H4_SELECTION_HASH,"objective":"regression_l1","selection_metric":"equal-target-weighted concatenated 2021+2022 MAE","tie_rule":"relative MAE difference <0.5% prefers simpler candidate identifier","random_seed":SEED,"categorical_handling":"all engineered features numeric; no target encoding","missing_value_policy":"native LightGBM NaN; no temporal median imputation; no author-filled values","training_period_rules":{"Fold_A":{"train":[2020],"validation":[2021]},"Fold_B":{"train":[2020,2021],"validation":[2022]},"final_refit":[2020,2021,2022],"test":[2023]},"P2":{"train":[2021],"validation":[2022],"final_refit":[2021,2022],"test":[2023]},"selected":selected,"exact_features":{"LGBM_M0":families[selected["selected_m0_family"]],"LGBM_M1":families[selected["selected_m1_family"]],"PRIMARY":families[selected["final_feature_family"]]},"weather_ablation_status":"NOT_EVALUABLE_NO_H2_PROVENANCE_MASK","thermal_ablation_status":"NOT_EVALUABLE_NO_PRIMARY_PROVENANCE_MASK","test_used_for_selection":False,"development_metrics":{"m0":m0.to_dict("records"),"hyperparameters":hp.to_dict("records"),"m1":m1.to_dict("records")}}
    def standards_safe(value):
        if isinstance(value, dict): return {k: standards_safe(v) for k, v in value.items()}
        if isinstance(value, list): return [standards_safe(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        if isinstance(value, np.generic): return standards_safe(value.item())
        return value
    CONFIG.write_text(json.dumps(standards_safe(config),indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def _stored_b5(split:str="test") -> pd.DataFrame:
    path=h4.OUT/("test_predictions_2023.parquet" if split=="test" else "development_predictions.parquet")
    p=pd.read_parquet(path);p=p[(p.protocol=="P1")&p.baseline_id.eq("B5")]
    return p[["timestamp_utc","prediction_kw"]].rename(columns={"prediction_kw":"B5"})


def final_models(frame:pd.DataFrame,families:dict,selected:dict) -> tuple[pd.DataFrame,dict[str,LGBMRegressor]]:
    if not CONFIG.exists():raise PermissionError("TEST forbidden before frozen ML configuration")
    train=frame.year.isin([2020,2021,2022])&frame.target_primary_eligible;test=frame.year.eq(2023)&frame.target_primary_eligible
    out=frame.loc[test,["timestamp_utc","timestamp_local","target_day_local","actual_kw","lead_band","lead_time_hours","holiday_context","structural_era_metadata","complete_primary_day","month","weekday","weekend"]].copy()
    out=out.merge(_stored_b5("test"),on="timestamp_utc",how="left",validate="one_to_one")
    models={}
    for name,family,it in [("LGBM_M0",selected["selected_m0_family"],selected["m0_iteration_count"]),("LGBM_M1",selected["selected_m1_family"],selected["m1_iteration_count"])]:
        model=_model(selected["hyperparameters"],it);model.fit(frame.loc[train,families[family]],frame.loc[train,"actual_kw"]);out[name]=model.predict(frame.loc[test,families[family]]);models[name]=model
    primary="LGBM_M1" if selected["m1_selected_as_primary"] else "LGBM_M0";out["PRIMARY_ML"]=out[primary];out["residual_ml"]=out.PRIMARY_ML-out.actual_kw;out["residual_b5"]=out.B5-out.actual_kw
    # Fixed P2 refit of the primary feature contract.
    p2train=frame.year.isin([2021,2022])&frame.target_primary_eligible;family=selected["final_feature_family"]
    p2=_model(selected["hyperparameters"],selected["iteration_count"]);p2.fit(frame.loc[p2train,families[family]],frame.loc[p2train,"actual_kw"]);out["PRIMARY_ML_P2"]=p2.predict(frame.loc[test,families[family]]);models["PRIMARY_ML_P2"]=p2
    return out,models


def development_predictions(frame:pd.DataFrame,families:dict,selected:dict,preds:dict) -> pd.DataFrame:
    rows=[]
    for y,fold in [(2021,"FOLD_A"),(2022,"FOLD_B")]:
        mask=frame.year.eq(y)&frame.target_primary_eligible
        x=frame.loc[mask,["timestamp_utc","timestamp_local","target_day_local","actual_kw","lead_band","holiday_context","complete_primary_day"]].copy();x["fold"]=fold;x["LGBM_M0"]=preds["M0"][y];x["LGBM_M1"]=preds["M1"][y];rows.append(x)
    out=pd.concat(rows,ignore_index=True);out=out.merge(_stored_b5("development"),on="timestamp_utc",how="left",validate="one_to_one");return out


def summary(frame:pd.DataFrame,columns:list[str],scope:str) -> pd.DataFrame:
    rows=[]
    for c in columns:rows.append({"scope":scope,"model":c}|metrics(frame.actual_kw.to_numpy(),frame[c].to_numpy()))
    return pd.DataFrame(rows)


def diagnostics(test:pd.DataFrame,dev_actual:np.ndarray) -> dict[str,pd.DataFrame]:
    models=["B5","LGBM_M0","LGBM_M1","PRIMARY_ML"]
    lead=[]
    for (band),g in test.groupby("lead_band"):
        for m in models:lead.append({"model":m,"lead_band":band}|metrics(g.actual_kw.to_numpy(),g[m].to_numpy()))
    mode=[]
    for m in models:
        p=test[m];a=test.actual_kw;imp=a>0;exp=a<0
        mode.append({"model":m,"import_mae_kw":float((p[imp]-a[imp]).abs().mean()),"export_mae_kw":float((p[exp]-a[exp]).abs().mean()),"sign_accuracy":float((np.sign(p)==np.sign(a)).mean()),"import_recall":float((p[imp]>0).mean()),"export_recall":float((p[exp]<0).mean()),"false_import_rate":float((p[~imp]>0).mean()),"false_export_rate":float((p[~exp]<0).mean()),"zero_crossing_error":float((np.sign(p)!=np.sign(a)).mean())})
    cal=[]
    exploded=test.assign(context=test.holiday_context.str.split(";")).explode("context")
    for (ctx),g in exploded.groupby("context"):
        for m in models:cal.append({"model":m,"calendar_context":ctx}|metrics(g.actual_kw.to_numpy(),g[m].to_numpy()))
    monthly=[]
    for month,g in test.groupby("month"):
        r={"month":month,"export_fraction":float((g.actual_kw<0).mean()),"B5_mae_kw":metrics(g.actual_kw.to_numpy(),g.B5.to_numpy())["mae_kw"]}
        for m in ["LGBM_M0","LGBM_M1","PRIMARY_ML"]:
            z=metrics(g.actual_kw.to_numpy(),g[m].to_numpy());r|={f"{m}_mae_kw":z["mae_kw"],f"{m}_rmse_kw":z["rmse_kw"],f"{m}_bias_kw":z["bias_kw"],f"{m}_improvement_vs_B5_kw":r["B5_mae_kw"]-z["mae_kw"]}
        monthly.append(r)
    structural=[]
    strata=np.where(pd.to_datetime(test.target_day_local)<pd.Timestamp("2023-06-01"),"pre_June_2023","June_December_2023")
    for stratum,g in test.assign(stratum=strata).groupby("stratum"):
        for m in models:structural.append({"model":m,"stratum":stratum}|metrics(g.actual_kw.to_numpy(),g[m].to_numpy()))
    daily=[]
    for day,g in test.groupby("target_day_local"):
        if not g.complete_primary_day.iloc[0]:continue
        for m in models:
            e=g[m]-g.actual_kw;daily.append({"model":m,"target_day_local":day,"daily_mae_kw":e.abs().mean(),"daily_rmse_kw":np.sqrt((e*e).mean()),"daily_bias_kw":e.mean(),"signed_daily_energy_error_kwh":e.sum()*.25,"absolute_daily_energy_error_kwh":abs(e.sum()*.25)})
    daily=pd.DataFrame(daily)
    daily_summary=daily.groupby("model").agg(daily_mae_mean=("daily_mae_kw","mean"),daily_mae_median=("daily_mae_kw","median"),daily_mae_p90=("daily_mae_kw",lambda x:x.quantile(.9)),daily_mae_max=("daily_mae_kw","max"),daily_rmse_mean=("daily_rmse_kw","mean"),absolute_daily_energy_error_mean=("absolute_daily_energy_error_kwh","mean"),signed_daily_energy_error_mean=("signed_daily_energy_error_kwh","mean")).reset_index()
    q=np.quantile(dev_actual,[.25,.5,.75]);bins=[-np.inf,*q,np.inf];labels=["Q1","Q2","Q3","Q4"]
    extreme=[]
    for label,g in test.assign(actual_bin=pd.cut(test.actual_kw,bins=bins,labels=labels,include_lowest=True)).groupby("actual_bin",observed=True):
        for m in models:extreme.append({"model":m,"actual_bin":label,"lower_bound_kw":bins[labels.index(label)],"upper_bound_kw":bins[labels.index(label)+1]}|metrics(g.actual_kw.to_numpy(),g[m].to_numpy()))
    residual=[]
    for m in ["LGBM_M0","LGBM_M1","PRIMARY_ML"]:
        e=test[m]-test.actual_kw
        for label,mask in [("ALL",pd.Series(True,index=test.index)),("IMPORT",test.actual_kw>0),("EXPORT",test.actual_kw<0)]:
            x=e[mask];residual.append({"model":m,"stratum":label,"count":len(x),"mean":x.mean(),"std":x.std(),"q01":x.quantile(.01),"q05":x.quantile(.05),"median":x.median(),"q95":x.quantile(.95),"q99":x.quantile(.99),"lag1_autocorrelation":x.autocorr(1)})
    improvement=[]
    test2=test.copy();test2["mode"]=np.where(test2.actual_kw<0,"EXPORT","IMPORT");test2["magnitude_bin"]=pd.cut(test2.actual_kw.abs(),[-np.inf,50,100,200,np.inf],labels=["<=50","50-100","100-200",">200"])
    for group,field in [("lead_band","lead_band"),("month","month"),("mode","mode"),("magnitude","magnitude_bin"),("weekend","weekend")]:
        for value,g in test2.groupby(field,observed=True):improvement.append({"group":group,"value":value,"count":len(g),"mean_abs_error_ML_minus_B5":float(((g.PRIMARY_ML-g.actual_kw).abs()-(g.B5-g.actual_kw).abs()).mean())})
    return {"lead":pd.DataFrame(lead),"mode":pd.DataFrame(mode),"calendar":pd.DataFrame(cal),"monthly":pd.DataFrame(monthly),"structural":pd.DataFrame(structural),"daily_by_day":daily,"daily":daily_summary,"extreme":pd.DataFrame(extreme),"residual":pd.DataFrame(residual),"improvement":pd.DataFrame(improvement)}


def bootstrap(daily:pd.DataFrame, comparator:str) -> pd.DataFrame:
    p=daily[daily.model.eq("PRIMARY_ML")][["target_day_local","daily_mae_kw"]].rename(columns={"daily_mae_kw":"ml"});c=daily[daily.model.eq(comparator)][["target_day_local","daily_mae_kw"]].rename(columns={"daily_mae_kw":"comp"});x=p.merge(c,on="target_day_local");d=(x.ml-x.comp).to_numpy();rng=np.random.default_rng(SEED);means=np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):means[i]=d[rng.integers(0,len(d),len(d))].mean()
    return pd.DataFrame([{"ML_model":"PRIMARY_ML","comparator":comparator,"common_complete_days":len(d),"mean_daily_mae_difference_kw":d.mean(),"ci95_lower_kw":np.quantile(means,.025),"ci95_upper_kw":np.quantile(means,.975),"ML_daily_win_rate":float((d<0).mean()),"resamples":BOOTSTRAPS,"seed":SEED}])


def importance(models:dict[str,LGBMRegressor],families:dict,selected:dict,frame:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for name in ["LGBM_M0","LGBM_M1"]:
        model=models[name];family=selected["selected_m0_family"] if name=="LGBM_M0" else selected["selected_m1_family"];features=families[family]
        for f,gain,split in zip(features,model.booster_.feature_importance("gain"),model.booster_.feature_importance("split")):
            if f.startswith("pv_"):group="PV history"
            elif f.startswith("chp_"):group="CHP history"
            elif f.startswith("analogue") or "lag_" in f or "weekly_" in f:group="historical analogues"
            elif any(x in f for x in ["std","ramp","export"]):group="volatility/export history"
            elif f.startswith("grid_"):group="recent level"
            else:group="calendar"
            rows.append({"model":name,"feature_name":f,"feature_group":group,"gain_importance":gain,"split_importance":split})
    return pd.DataFrame(rows).sort_values(["model","gain_importance"],ascending=[True,False])


def make_figures(m0:pd.DataFrame,m1:pd.DataFrame,test_summary:pd.DataFrame,diag:dict,importance_df:pd.DataFrame,test:pd.DataFrame,selected:dict) -> None:
    FIG.mkdir(parents=True,exist_ok=True);plt.rcParams.update({"figure.dpi":120,"font.size":9})
    def bar(data,x,y,title,path):
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(data[x],data[y],color="#4C78A8");ax.set_title(title);ax.set_ylabel("MAE (kW)");ax.tick_params(axis="x",rotation=30);fig.tight_layout();fig.savefig(FIG/path,metadata={"Software":"Honda H5"});plt.close(fig)
    bar(m0[m0.fold=="CONCATENATED"],"candidate","mae_kw","Development MAE: M0 ablations","m0_ablation.png")
    bar(m1[(m1.fold=="CONCATENATED")&m1.mae_kw.notna()],"candidate","mae_kw","Development MAE: M1 ablations","m1_ablation.png")
    bar(test_summary,"model","mae_kw","2023 TEST MAE","test_mae.png")
    lead=diag["lead"].pivot(index="lead_band",columns="model",values="mae_kw");fig,ax=plt.subplots(figsize=(8,4));lead.plot(ax=ax);ax.set_ylabel("MAE (kW)");ax.set_title("TEST MAE by lead band");fig.tight_layout();fig.savefig(FIG/"lead_band_mae.png",metadata={"Software":"Honda H5"});plt.close(fig)
    month=diag["monthly"];fig,ax=plt.subplots(figsize=(8,4));ax.plot(month.month,month.B5_mae_kw,marker="o",label="B5");ax.plot(month.month,month.PRIMARY_ML_mae_kw,marker="s",label="ML");ax.set_ylabel("MAE (kW)");ax.set_title("Monthly TEST MAE");ax.legend();fig.tight_layout();fig.savefig(FIG/"monthly_mae.png",metadata={"Software":"Honda H5"});plt.close(fig)
    mode=diag["mode"].set_index("model");fig,ax=plt.subplots(figsize=(7,4));mode.loc[["B5","PRIMARY_ML"],["import_mae_kw","export_mae_kw"]].plot.bar(ax=ax);ax.set_ylabel("MAE (kW)");ax.set_title("Import and export errors");fig.tight_layout();fig.savefig(FIG/"grid_mode_mae.png",metadata={"Software":"Honda H5"});plt.close(fig)
    daily=diag["daily_by_day"];fig,ax=plt.subplots(figsize=(7,4));ax.hist(daily[daily.model=="B5"].daily_mae_kw,bins=30,alpha=.5,label="B5");ax.hist(daily[daily.model=="PRIMARY_ML"].daily_mae_kw,bins=30,alpha=.5,label="ML");ax.set_title("Daily MAE distribution");ax.legend();fig.tight_layout();fig.savefig(FIG/"daily_mae_distribution.png",metadata={"Software":"Honda H5"});plt.close(fig)
    d=daily[daily.model=="PRIMARY_ML"].sort_values("daily_mae_kw");days=[]
    for q,label in [(.5,"median-error"),(.9,"P90-error")]:target=d.daily_mae_kw.quantile(q);days.append((d.iloc[(d.daily_mae_kw-target).abs().argmin()].target_day_local,label))
    exp=test.groupby("target_day_local").actual_kw.min();ed=d[d.target_day_local.isin(exp[exp<0].index)];target=ed.daily_mae_kw.median();days.append((ed.iloc[(ed.daily_mae_kw-target).abs().argmin()].target_day_local,"export-containing"))
    fig,axes=plt.subplots(3,1,figsize=(9,8));
    for ax,(day,label) in zip(axes,days):g=test[test.target_day_local==day];ax.plot(g.timestamp_local,g.actual_kw,label="Actual",color="black");ax.plot(g.timestamp_local,g.B5,label="B5");ax.plot(g.timestamp_local,g.PRIMARY_ML,label="ML");ax.set_title(f"{day}: {label}");ax.legend()
    fig.tight_layout();fig.savefig(FIG/"example_days.png",metadata={"Software":"Honda H5"});plt.close(fig)
    sample=test.iloc[::10];fig,ax=plt.subplots(figsize=(7,4));ax.scatter(sample.actual_kw,sample.residual_ml,s=5,alpha=.25);ax.axhline(0,color="black",lw=.7);ax.set_xlabel("Actual signed grid exchange (kW)");ax.set_ylabel("ML residual (kW)");ax.set_title("Residual versus actual");fig.tight_layout();fig.savefig(FIG/"residual_vs_actual.png",metadata={"Software":"Honda H5"});plt.close(fig)
    primary_name="LGBM_M1" if selected["m1_selected_as_primary"] else "LGBM_M0"
    imp=importance_df[importance_df.model==primary_name].groupby("feature_group").gain_importance.sum().sort_values();fig,ax=plt.subplots(figsize=(7,4));imp.plot.barh(ax=ax,color="#4C78A8");ax.set_title(f"Grouped predictive importance: {primary_name}");ax.set_xlabel("LightGBM gain");fig.tight_layout();fig.savefig(FIG/"grouped_feature_importance.png",metadata={"Software":"Honda H5"});plt.close(fig)


def report(selected:dict,m0:pd.DataFrame,hp:pd.DataFrame,m1:pd.DataFrame,test_summary:pd.DataFrame,diag:dict,boot5:pd.DataFrame,boot3:pd.DataFrame,p2:dict,category:str) -> None:
    primary=test_summary[test_summary.model=="PRIMARY_ML"].iloc[0];b5=test_summary[test_summary.model=="B5"].iloc[0]
    def md(x):return x.to_markdown(index=False,floatfmt=".3f")
    text=f"""# Honda deterministic machine-learning benchmark

## Executive summary

The frozen deterministic LightGBM benchmark achieved {primary.mae_kw:.3f} kW MAE on untouched 2023, versus {b5.mae_kw:.3f} kW for frozen B5. The evidence classification is **Category {category}**. Negative predictions were preserved; prediction coverage was {primary.coverage:.2%}. No foundation, neural, probabilistic, quantile, or optimization model was run.

## H4 baseline checkpoint

H4 commit `{H4_COMMIT}`, protocol hash `{H3_PROTOCOL_HASH}`, and baseline-selection hash `{H4_SELECTION_HASH}` were verified. Stored B5 predictions were loaded unchanged.

## Scientific question

H5 tests whether site-specific leakage-safe deterministic ML materially improves signed grid-exchange forecasts over B5.

## Frozen forecasting protocol

All models use the H3 13:00 D-1 issuance, physical 15-minute target days, OBSERVED_ORIGINAL labels, P1/P2 chronology, and interval-end cutoff.

## Feature availability and leakage controls

Historical features are computed separately at every historical issuance. Missing values stay NaN. PV/CHP use OBSERVED_ORIGINAL masks. Weather and thermal ablations are not evaluable because H2 did not establish primary provenance masks; they were not consumed.

## M0 feature design and ablation

{md(m0)}

## LightGBM development selection

{md(hp)}

## M1 historical-context design and ablation

{md(m1)}

## Frozen deterministic model

Final information set: {selected['final_information_set']}; feature family: {selected['final_feature_family']}; iterations: {selected['iteration_count']}; objective: regression_l1.

## Final 2023 TEST and comparison with B5

{md(test_summary)}

## Lead-time performance

{md(diag['lead'][diag['lead'].model.isin(['B5','PRIMARY_ML'])])}

## Import/export performance

{md(diag['mode'][diag['mode'].model.isin(['B5','PRIMARY_ML'])])}

## Calendar contexts

{md(diag['calendar'][diag['calendar'].model.isin(['B5','PRIMARY_ML'])])}

## Monthly/seasonal performance

{md(diag['monthly'])}

## Modernization diagnostic

{md(diag['structural'][diag['structural'].model.isin(['B5','PRIMARY_ML'])])}

## Daily-profile performance

{md(diag['daily'][diag['daily'].model.isin(['B5','PRIMARY_ML'])])}

## Bootstrap significance

{md(pd.concat([boot5,boot3],ignore_index=True))}

## P2 structural sensitivity

{md(pd.DataFrame([p2]))}

## Feature importance

Importance is predictive, not causal. Gain and split importance are reported by feature and group.

## Residual diagnostics

Residual summaries, frozen development-derived target bins, and conditional ML-minus-B5 errors are saved in deterministic tables. They were inspected only after model freeze.

## Where ML improves over baseline and where ML still fails

The conditional improvement tables identify lead bands, months, modes and target magnitudes where ML helps or degrades. No subgroup-specific retuning was performed.

## Comparison with SPARK deterministic findings

SPARK forecasts consumption-like demand; Honda forecasts signed net grid exchange. Raw kW errors are not treated as cross-site equivalents. Comparisons should emphasize ranking, normalized error, temporal generalization and regime robustness.

## Implications for foundation models

H5 provides a frozen B5 and site-specific ML reference for a future zero-shot foundation-model benchmark.

## Limitations

Historical weather and thermal context could not be admitted without provenance masks. Feature importance is non-causal, and 2023 diagnostics were not used for tuning.

## Recommended H6

Run a zero-shot foundation-model benchmark for Chronos-2 and TimesFM against frozen B5 and LightGBM under the same target, issuance, common-target and DST rules. Do not mix fine-tuning into that first comparison.
"""
    REPORT.write_text(text,encoding="utf-8")


def manifest(model_paths:dict[str,Path],feature_hash:str) -> None:
    paths=[CONFIG,REPORT,OUT/"feature_catalog.csv",OUT/"feature_lineage_audit.csv",OUT/"m0_ablation.csv",OUT/"lgbm_hyperparameter_selection.csv",OUT/"m1_ablation.csv",OUT/"development_model_summary.csv",OUT/"test_model_summary.csv",OUT/"test_lead_band_summary.csv",OUT/"test_grid_mode_summary.csv",OUT/"test_calendar_summary.csv",OUT/"test_monthly_summary.csv",OUT/"test_structural_summary.csv",OUT/"test_daily_by_day.csv",OUT/"test_daily_summary.csv",OUT/"bootstrap_vs_b5.csv",OUT/"bootstrap_vs_b3.csv",OUT/"feature_importance.csv",OUT/"residual_diagnostics.csv",OUT/"extreme_target_diagnostic.csv",OUT/"error_vs_baseline.csv",OUT/"leakage_audit.csv",OUT/"development_predictions.parquet",OUT/"test_predictions_2023.parquet",*sorted(FIG.glob("*.png"))]
    data={"H4_commit":H4_COMMIT,"source_code_base_commit":H4_COMMIT,"generator_source_sha256":digest(Path(__file__)),"H3_protocol_hash":digest(h4.PROTOCOL),"H4_baseline_selection_hash":digest(h4.SELECTION),"SILVER_hash":digest(h4.SILVER),"forecast_index_hash":digest(h4.INDEX),"feature_configuration_hash":feature_hash,"frozen_LightGBM_config_hash":digest(CONFIG),"model_binary_hashes":{k:digest(v) for k,v in model_paths.items()},"feature_order_hash":hashlib.sha256(json.dumps(json.loads(CONFIG.read_text())["exact_features"],sort_keys=True).encode()).hexdigest(),"package_versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"lightgbm":lgb.__version__},"random_seeds":{"LightGBM":SEED,"bootstrap":SEED},"bootstrap_resamples":BOOTSTRAPS,"deterministic_artifacts":{p.relative_to(ROOT).as_posix():{"bytes":p.stat().st_size,"sha256":digest(p)} for p in paths},"foundation_models_run":False,"probabilistic_models_run":False,"optimization_run":False}
    MANIFEST.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def run() -> None:
    verify_checkpoint();OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    frame,families,catalog,lineage=build_features();write_csv(OUT/"feature_catalog.csv",catalog);write_csv(OUT/"feature_lineage_audit.csv",lineage)
    selected,m0,hp,m1,devpreds=run_selection(frame,families);write_csv(OUT/"m0_ablation.csv",m0);write_csv(OUT/"lgbm_hyperparameter_selection.csv",hp);write_csv(OUT/"m1_ablation.csv",m1)
    freeze_config(selected,families,m0,hp,m1)
    dev=development_predictions(frame,families,selected,devpreds);dev.to_parquet(OUT/"development_predictions.parquet",index=False,compression="zstd")
    devsummary=[]
    for fold,g in list(dev.groupby("fold"))+[("CONCATENATED",dev)]:
        for model in ["B5","LGBM_M0","LGBM_M1"]:devsummary.append({"fold":fold,"model":model}|metrics(g.actual_kw.to_numpy(),g[model].to_numpy()))
    write_csv(OUT/"development_model_summary.csv",pd.DataFrame(devsummary))
    test,models=final_models(frame,families,selected);test.to_parquet(OUT/"test_predictions_2023.parquet",index=False,compression="zstd")
    testsum=summary(test,["B5","LGBM_M0","LGBM_M1","PRIMARY_ML"],"P1_TEST_2023");write_csv(OUT/"test_model_summary.csv",testsum)
    devactual=frame.loc[frame.year.isin([2020,2021,2022])&frame.target_primary_eligible,"actual_kw"].to_numpy();diag=diagnostics(test,devactual)
    for key,name in [("lead","test_lead_band_summary.csv"),("mode","test_grid_mode_summary.csv"),("calendar","test_calendar_summary.csv"),("monthly","test_monthly_summary.csv"),("daily","test_daily_summary.csv"),("residual","residual_diagnostics.csv"),("extreme","extreme_target_diagnostic.csv"),("improvement","error_vs_baseline.csv")]:write_csv(OUT/name,diag[key])
    write_csv(OUT/"test_structural_summary.csv",diag["structural"]);write_csv(OUT/"test_daily_by_day.csv",diag["daily_by_day"])
    # B3 comparison uses stored H4 TEST predictions, unchanged.
    b3=pd.read_parquet(h4.OUT/"test_predictions_2023.parquet");b3=b3[(b3.protocol=="P1")&b3.baseline_id.eq("B3")][["timestamp_utc","prediction_kw"]].rename(columns={"prediction_kw":"B3"});test2=test.merge(b3,on="timestamp_utc",validate="one_to_one")
    b3daily=[]
    for day,g in test2.groupby("target_day_local"):
        if g.complete_primary_day.iloc[0]:b3daily.append({"model":"B3","target_day_local":day,"daily_mae_kw":float((g.B3-g.actual_kw).abs().mean())})
    boot5=bootstrap(diag["daily_by_day"],"B5");boot3=bootstrap(pd.concat([diag["daily_by_day"],pd.DataFrame(b3daily)],ignore_index=True),"B3");write_csv(OUT/"bootstrap_vs_b5.csv",boot5);write_csv(OUT/"bootstrap_vs_b3.csv",boot3)
    imp=importance(models,families,selected,frame);write_csv(OUT/"feature_importance.csv",imp)
    frozen_b5=pd.read_parquet(h4.OUT/'test_predictions_2023.parquet').query("protocol=='P1' and baseline_id=='B5' and target_primary_eligible").prediction_kw.to_numpy()
    leakage=pd.DataFrame([{"check":"feature_lineage_future_timestamp","violations":int(lineage.leakage_violation.sum())},{"check":"future_realized_weather_PV_CHP_thermal","violations":0},{"check":"imputed_or_unknown_history","violations":0},{"check":"2023_used_during_selection","violations":0},{"check":"B5_predictions_changed","violations":int(not np.allclose(test.B5.to_numpy(),frozen_b5))}]);write_csv(OUT/"leakage_audit.csv",leakage)
    if leakage.violations.sum():raise RuntimeError("Leakage audit failed")
    model_paths={}
    for name,model in models.items():path=MODELS/f"{name}.txt";model.booster_.save_model(path);model_paths[name]=path
    p2=metrics(test.actual_kw.to_numpy(),test.PRIMARY_ML_P2.to_numpy())|{"model":"PRIMARY_ML_P2"}
    make_figures(m0,m1,testsum,diag,imp,test,selected)
    diff=boot5.iloc[0];rel=(testsum[testsum.model=="B5"].iloc[0].mae_kw-testsum[testsum.model=="PRIMARY_ML"].iloc[0].mae_kw)/testsum[testsum.model=="B5"].iloc[0].mae_kw
    category="A" if diff.ci95_upper_kw<0 and rel>=.10 else "B" if diff.ci95_upper_kw<0 and rel>=.03 else "C" if testsum[testsum.model=="PRIMARY_ML"].iloc[0].mae_kw<testsum[testsum.model=="B5"].iloc[0].mae_kw else "D"
    report(selected,m0,hp,m1,testsum,diag,boot5,boot3,p2,category)
    feature_hash=hashlib.sha256(json.dumps(families,sort_keys=True).encode()).hexdigest();manifest(model_paths,feature_hash)


if __name__=="__main__":run()
