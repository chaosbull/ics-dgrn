# ICS-DGRN mechanism experiments (`result/pems_newexp`)

This folder is independent of `result/pems_st200` (main DL comparison) and `result/pems` (Ridge / ablations).  
Below: **what each file shows / what CSV columns mean / how to read plots**.

**Setup**: PEMS08 primary (multi-seed also uses PEMS03/04); hist=12 -> horizon=12; 300 epochs; SmoothL1; Adam+Cosine; subsample 800/150/250; default `layer_dims=[56,40]`, `compression_dims=[28]`, `esp_target=0.9`; conda `py312` + CUDA.

---

## Directory overview

```
result/pems_newexp/
  REPORT.md                 # this guide
  compression_ratio/        # interlayer compression-ratio sensitivity
  sparse_density/           # compression-matrix density sensitivity
  esp_state_decay/          # state-difference decay (direct ESP check)
  esp_forecast_joint/       # spectral scaling tau x forecasting
  fixed_vs_trainable/       # fixed vs trainable recurrent W
  multi_seed/               # multi-seed mean +/- std
  multi_horizon/            # 1-12 step error growth
  compute_overhead/         # params / time / GPU memory
  qualitative/              # three scenario prediction curves
```

---

## 1. `compression_ratio/` — interlayer compression ratio

Layer dims fixed at `[56,40]`; only compression dim `d_comp = round(56 * r)` changes, for `r in {1.0, 0.8, 0.6, 0.4, 0.2}`.

### Tables

| File | Meaning |
|------|---------|
| `PEMS08_compression_metrics.csv` | One row per `r`: test MSE/MAE/RMSE/MAPE/R2, train time, trainable params, compression dim, ESP fields |
| `PEMS08_compression_esp.json` | JSON backup of the same run |

**Summary**

| r | compression dim | MAE | RMSE | R2 | trainable params |
|--:|----------------:|----:|-----:|---:|-----------------:|
| 1.0 | 56 | 20.177 | 31.390 | 0.9526 | 41182 |
| 0.8 | 45 | 20.162 | 31.336 | 0.9528 | 40126 |
| 0.6 | 34 | 20.185 | 31.352 | 0.9527 | 39070 |
| 0.4 | 22 | 20.180 | 31.330 | 0.9528 | 37918 |
| 0.2 | 11 | 20.224 | 31.417 | 0.9525 | 36862 |

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_compression_curves.png` | Three panels: x=`r`, y=MAE / RMSE / R2 | Accuracy vs compression strength; lowest MAE at `r=0.8` here |
| `PEMS08_compression_params_vs_mae.png` | Left: trainable params vs `r`; right: MAE vs `r` | Capacity vs accuracy on one figure |

---

## 2. `sparse_density/` — compression matrix density

Compression dim fixed at 28; nonzero ratio of Phi is `rho in {1.0, 0.5, 0.2, 0.1, 0.05}`.

### Tables

| File | Meaning |
|------|---------|
| `PEMS08_density_metrics.csv` | One row per `rho`; includes `phi_spec_norm_l0` (spectral norm of Phi) |

**Summary**

| rho | MAE | RMSE | R2 | \|\|Phi\|\|_2 |
|----:|----:|-----:|---:|-------------:|
| 1.0 | 20.197 | 31.348 | 0.9527 | 1.97 |
| 0.5 | 20.255 | 31.508 | 0.9523 | 1.46 |
| 0.2 | 20.240 | 31.506 | 0.9523 | 1.07 |
| 0.1 | 20.301 | 31.608 | 0.9519 | 0.91 |
| 0.05 | 20.353 | 31.665 | 0.9518 | 0.72 |

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_density_curves.png` | x=`rho`, y=MAE / RMSE / R2 | Smaller `rho` = sparser; watch whether error rises |
| `PEMS08_density_params_vs_mae.png` | Params and MAE | Parameter count may stay flat if mask is not counted as fewer Parameters; focus on MAE |

---

## 3. `esp_state_decay/` — state difference decay

Same input, two random initial states; track per-layer state difference over time.  
`Dhat_l(t) = ||Xa-Xb||_F / ||Xa(0)-Xb(0)||_F` (log y-axis).

Spectral scaling `tau in {0.5, 0.7, 0.9, 1.0, 1.1}` (`esp_target`).

### Tables / JSON

| File | Meaning |
|------|---------|
| `PEMS08_esp_decay_summary.csv` / `.json` | Per `tau`: `max_kappa`, `esp_ok`, late/early ratios, final Dhat, decay label |
| `PEMS08_tau{tau}_layer{k}_Dhat.csv` | Columns `t`, `D_hat`, `D` for layer k |

**Summary**

| tau | max_kappa | esp_ok | L1 late/early | L2 late/early | final Dhat | label |
|----:|----------:|:------:|--------------:|--------------:|-----------:|:-----:|
| 0.5 | 0.500 | True | 0.0076 | 0.0232 | 0.0007 | Strong |
| 0.7 | 0.714 | True | 0.0099 | 0.0317 | 0.0012 | Strong |
| 0.9 | 0.897 | True | 0.0143 | 0.0448 | 0.0022 | Strong |
| 1.0 | 1.048 | False | 0.0164 | 0.0446 | 0.0023 | Moderate |
| 1.1 | 1.088 | False | 0.0190 | 0.0560 | 0.0032 | Moderate |

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_tau{tau}_state_decay.png` | x=time, y=Dhat (log); Layer 1 / Layer 2 | Downward curves = fading initial-state effect |
| `PEMS08_state_decay_vs_tau.png` | Final layer only; one curve per `tau` | Larger `tau` usually decays slower |

---

## 4. `esp_forecast_joint/` — spectral scaling x forecast

Full train for each `tau`; record kappa, decay label, test MAE/RMSE, 12-step error, `G_horizon`.

### Tables

| File | Meaning |
|------|---------|
| `PEMS08_esp_forecast_table.csv` | Short table: tau, max_kappa, StateDecay, MAE, RMSE, R2, MAE_h12, G_horizon, esp_ok |
| `PEMS08_esp_forecast_full.csv` | Full metrics |
| `PEMS08_tau{tau}_horizon_mae.csv` | Columns `h` (1..12), MAE |

**Summary**

| tau | max_kappa | StateDecay | MAE | MAE_12 | G_horizon |
|----:|----------:|:----------:|----:|-------:|----------:|
| 0.5 | 0.511 | Strong | 20.221 | 25.673 | 0.827 |
| 0.7 | 0.715 | Strong | 20.178 | 25.575 | 0.823 |
| 0.9 | 0.919 | Strong | 20.168 | 25.556 | 0.823 |
| 1.0 | 1.022 | Moderate | 20.177 | 25.601 | 0.828 |
| 1.1 | 1.124 | Moderate | 20.158 | 25.584 | 0.830 |

`G_horizon = (MAE_12 - MAE_1) / MAE_1`.

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_esp_forecast_joint.png` | Left: tau vs max_kappa (gray line at 1); right: tau vs test MAE | Theory side vs forecasting side |

---

## 5. `fixed_vs_trainable/` — fixed vs trainable reservoir

| Variant | Meaning |
|---------|---------|
| FixedReservoir | Sparse W fixed; train Win / Phi / leak / readout (default ICS-DGRN) |
| TrainableReservoir | Recurrent W also receives gradients |

### Tables

| File | Meaning |
|------|---------|
| `PEMS08_fixed_vs_trainable.csv` | Two rows: MAE/RMSE/R2, TrainableParams, TotalParams, TrainTimeSec, PeakGPUMemMB, esp_ok |

**Summary**

| variant | MAE | RMSE | trainable | total | train (s) | peak GPU (MB) | esp_ok |
|---------|----:|-----:|----------:|------:|----------:|--------------:|:------:|
| Fixed | 20.168 | 31.354 | 38494 | 49534 | 252.4 | 197.9 | True |
| Trainable | 20.010 | 31.106 | 43230 | 49534 | 262.2 | 198.4 | False |

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_fixed_vs_trainable.png` | Bars: MAE, trainable params, train time | Accuracy vs optimization cost |

---

## 6. `multi_seed/` — multiple seeds

ICS-DGRN: PEMS03/04/08 with seeds 42, 43, 44.  
AGCRN: PEMS08 only, same three seeds.

### Tables

| File | Meaning |
|------|---------|
| `all_seed_runs.csv` | One row per independent run |
| `multi_seed_summary.csv` | Mean/std of MAE/RMSE/R2 by dataset x model |

**Summary**

| dataset | model | MAE | RMSE | R2 |
|---------|-------|----:|-----:|---:|
| PEMS03 | ICS-DGRN | 18.764 +/- 0.246 | 29.483 +/- 0.149 | 0.9560 +/- 0.0022 |
| PEMS04 | ICS-DGRN | 25.014 +/- 0.192 | 38.258 +/- 0.378 | 0.9419 +/- 0.0012 |
| PEMS08 | ICS-DGRN | 20.261 +/- 0.084 | 31.219 +/- 0.389 | 0.9539 +/- 0.0012 |
| PEMS08 | AGCRN | 20.335 +/- 0.173 | 31.138 +/- 0.247 | 0.9542 +/- 0.0013 |

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_multiseed_mae.png` | Mean MAE bars with std error bars (ICS-DGRN vs AGCRN) | Overlap of error bars |
| `PEMS04_multiseed_mae.png` | ICS-DGRN only | Variance on that set |
| `PEMS03_multiseed_mae.png` | ICS-DGRN only | Same |

---

## 7. `multi_horizon/` — multi-step error growth

Train ICS-DGRN + STGCN/TGCN/GWNet/AGCRN on PEMS08; MAE per horizon `h=1..12`.

### Tables

| File | Meaning |
|------|---------|
| `PEMS08_horizon_mae_detail.csv` | Rows = horizons 1..12; columns = models |
| `PEMS08_horizon_growth.csv` | Per model: overall MAE, MAE_h1, MAE_h12, DeltaMAE, G_horizon |

**Definitions**

- `DeltaMAE = MAE_12 - MAE_1`
- `G_horizon = (MAE_12 - MAE_1) / MAE_1`

**Summary**

| model | MAE | MAE_1 | MAE_12 | DeltaMAE | G |
|-------|----:|------:|-------:|---------:|--:|
| ICS-DGRN | 20.168 | 14.017 | 25.556 | 11.540 | 0.823 |
| AGCRN | 20.245 | 14.444 | 25.980 | 11.536 | 0.799 |
| STGCN | 21.066 | 13.957 | 27.874 | 13.917 | 0.997 |
| TGCN | 23.749 | 16.559 | 30.865 | 14.305 | 0.864 |
| GWNet | 21.420 | 14.074 | 28.402 | 14.328 | 1.018 |

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_horizon_mae.png` | x=`h`, y=MAE; one curve per model | Higher / steeper = worse long-horizon error |
| `PEMS08_G_horizon_bar.png` | Bars of `G_horizon` | Relative growth only (not absolute MAE) |

---

## 8. `compute_overhead/` — params / time / memory

Same protocol for five models: trainable params, total params, train time, peak GPU memory.

### Tables

| File | Meaning |
|------|---------|
| `PEMS08_compute_overhead.csv` | One row per model |

**Summary**

| model | MAE | trainable | total | train (s) | peak GPU (MB) |
|-------|----:|----------:|------:|----------:|--------------:|
| ICS-DGRN | 20.168 | 38494 | 49534 | 259.2 | 198.6 |
| STGCN | 21.067 | 25740 | 25740 | 322.3 | 897.3 |
| TGCN | 23.749 | 3564 | 3564 | 162.7 | 86.3 |
| GWNet | 21.417 | 12876 | 12876 | 212.4 | 181.0 |
| AGCRN | 20.245 | 4156 | 4156 | 122.7 | 55.0 |

### Figures

| File | What is plotted | How to read |
|------|-----------------|-------------|
| `PEMS08_compute_overhead.png` | Three bar panels: trainable params, train seconds, peak GPU MB | Size / wall-clock / memory |

---

## 9. `qualitative/` — multi-scenario curves

Test set node 0, horizon 1; three auto-selected windows vs Ground Truth / ICS-DGRN / AGCRN.

| File | What is plotted | How the window is chosen |
|------|-----------------|--------------------------|
| `PEMS08_scenario_A_normal.png` | Blue=truth, orange=ICS-DGRN, green dashed=AGCRN (~80 steps) | Lowest variation (calm traffic) |
| `PEMS08_scenario_B_rapid_rise.png` | Same | Largest positive increments |
| `PEMS08_scenario_C_rapid_fall.png` | Same | Largest negative increments |
| `PEMS08_scenarios.json` | `start` / `end` indices | Window locations in the test series |

Read for lag / overshoot qualitatively; not a primary accuracy proof.

---

## 10. Root files

| File | Meaning |
|------|---------|
| `REPORT.md` | This guide |

---

## Common CSV columns

| Column | Meaning |
|--------|---------|
| MAE / RMSE / R2 / MSE / MAPE | Denormalized test metrics (MAPE can explode near zero flow) |
| TrainTimeSec / InferTimeSec | Wall-clock train / test inference seconds |
| TrainableParams / TotalParams | Trainable count / approx total float storage |
| PeakGPUMemMB | Peak `torch.cuda.max_memory_allocated` (MB) |
| MAE_h1 / MAE_h12 | MAE at horizon 1 / 12 |
| DeltaMAE / G_horizon | See section 7 |
| max_kappa / esp_ok / esp_target | max_l \|\|S\|\|_2\|\|W\|\|_2; whether <1; chosen tau |
| compression_ratio / compression_dim | r and integer compression width |
| rho / phi_density | Nonzero ratio of compression matrix |
| train_recurrent | True if W is trainable |
| seed | Random seed |
