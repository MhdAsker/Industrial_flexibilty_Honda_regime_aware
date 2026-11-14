# NVIDIA Kumo Forecast: Honda integration architecture and experimental design

Status: implemented and smoke-tested; full development selection and retrospective 2023 evaluation are not run.

## Scientific boundary

This extension asks whether retrieval of issuance-safe historical industrial operating patterns adds value beyond a pretrained Kumo Forecast prediction. It preserves the frozen Honda target, issue time, provenance rules, development folds, and historical benchmark artifacts. It is a retrospective post-benchmark extension because the 2023 Honda period has already been examined by earlier project stages.

The official interface audited on 2026-11-14 is `perform_forecasting(df, config=ForecastingConfig(...), context_df=None)` from NVIDIA Kumo-TS 1.0.1 at commit `ad459475d1302fc8365e7117e39837a8ecb6a7e3`. Kumo reads a fixed history and predicts 72 steps natively, then recursively extends longer horizons. DARR takes a continuous historical `context_df` and blends `alpha * direct + (1 - alpha) * retrieved`; `k`, `temperature`, and `context_stride` control retrieval. NVIDIA explicitly warns that DARR need not beat direct forecasting and that stride can create phase errors. See the [official Kumo documentation](https://docs.nvidia.com/sdgm/structured-data-models/models/kumo-forecast) and [Kumo-TS repository](https://github.com/NVIDIA/Kumo-TS).

## Forecast-path contract

The last permissible measurement ends at 13:00 Europe/Berlin on D-1. The Kumo input therefore ends at 12:45 local. The adapter requests every physical 15-minute step from 13:00 D-1 through the final interval of D, then extracts only delivery day D.

| Delivery day | Delivery intervals | Issue-to-delivery path | First target position |
|---|---:|---:|---:|
| ordinary | 96 | 140 | 44 |
| spring DST | 92 | 136 | 44 |
| autumn DST | 100 | 144 | 44 |

All construction occurs on an exact UTC grid. This prevents the common error of treating the first 96 values after issuance as the next delivery day. Kumo's 72-step native output requires one audited autoregressive extension for these 136–144-step paths.

## Components

- `data_adapter.py` creates exact UTC contexts and target positions. Only `OBSERVED_ORIGINAL` history is admitted. Missing input history is interpolated only inside the already available pre-issuance window, and the imputed count remains in every output row. Targets and retrieval continuations are never imputed.
- `retrieval.py` implements candidate eligibility, provenance metadata, raw-profile distance, external regime filtering, weighted retrieval, and a complete-run adapter for native DARR.
- `regimes.py` creates calendar families and pre-issuance load regimes. Its thresholds were fitted only on 272 usable 2020 origins; 94 origins with no observed context were excluded and counted.
- `runner.py` pins code, model, and backbone revisions; downloads ignored artifacts; runs experiments; blocks unfrozen TEST access; records hardware/runtime; freezes a final configuration only from a development selection table; and generates diagnostics.
- `evaluation.py` computes coverage, MAE, RMSE, bias, median absolute error, absolute-target NMAE, lead, calendar, structural-era, and import/export tables. Paired uncertainty resamples whole local days.
- `diagnostics.py` plots the query, retrieved inputs, observed continuations, direct prediction, retrieval-enhanced prediction, and actual trajectory. Similarity is explicitly not interpreted as causal or physical equivalence.
- `downstream.py` maps a selected, complete Kumo point forecast into the existing Honda scenario contract without manufacturing quantiles or trajectories. Incomplete days are rejected.

## Experiments

1. `DIRECT`: official pretrained Kumo with no retrieval and no Honda gradient training.
2. `NATIVE_DARR`: official DARR over one fully observed continuous archive. The public API has no per-window eligibility mask; selecting a complete run prevents internal zero-fill from fabricating retrieved continuations.
3. `REGIME_*`: a clearly labeled external retrieval wrapper around direct Kumo. It is not native DARR. Candidate windows occur at historical Honda issue times and are optionally filtered by exact calendar context, calendar family, or calendar family plus pre-issuance load regime.
4. Development ablation: sequence length 512/1024; k 8/32; direct blend weight 0/0.25/0.75/1; stride 1/4; temperature 0.05. The grid is deliberately small and fixed before inference.

For Fold A, candidates and fitted information are restricted to 2020 when forecasting 2021. Fold B may use 2020–2021 when forecasting 2022. The final 2023 archive may use 2020–2022 only after development selection has been frozen.

## Retrieval eligibility

Every external candidate must have a complete finite input and continuation, end its input exactly at its historical issue, begin its continuation at that issue, and finish its continuation before the new query issue. Candidate timestamps, availability time, calendar label, pre-issuance regime, distance, rank, and weight are retained. Deliberately injected future and nonfinite continuations are rejected by tests.

Native DARR differs: NVIDIA accepts one continuous frame and does not expose retrieved-neighbor identities. This study therefore limits native DARR to one fully observed continuous run and uses the external wrapper for auditable regime filtering and neighbor-level explanations.

## Reproducible commands

```powershell
python -m venv .venv-kumo --system-site-packages
.\.venv-kumo\Scripts\python.exe -m pip install -e "<Kumo-TS checkout>\Kumo-Forecast"
.\.venv-kumo\Scripts\python.exe -m src.honda.kumo.runner prepare
python -m src.honda.kumo.runner audit

.\.venv-kumo\Scripts\python.exe -m src.honda.kumo.runner run `
  --phase development --method DIRECT --seq-len 512 --alpha 1 `
  --k 8 --temperature 0.05 --context-stride 4 --device cpu `
  --output results/kumo/development_direct_l512.parquet `
  --runtime-output results/kumo/development_direct_l512_runtime.json

python -m src.honda.kumo.runner evaluate `
  --phase development --predictions results/kumo/development_direct_l512.parquet

python -m src.honda.kumo.runner freeze `
  --selection-table results/kumo/development_selection.csv
```

TEST remains blocked until the last command creates `config/honda_kumo_final_frozen.json`. Large weights, virtual environments, and prediction parquet files are ignored by Git.

## Hardware path

The local driver sees a GTX 1650 with 4 GB VRAM, but the installed PyTorch build is CPU-only. NVIDIA's model card reports A100/H100 evaluation hardware and does not establish 4 GB GPU compatibility. The pinned Kumo head is 1.404 GB and the pinned MOMENT backbone is 1.386 GB, before runtime tensors. The verified low-risk path here is CPU inference; GPU use must first pass a memory smoke test in a CUDA-enabled isolated environment.
