# Multi-seed traffic forecasting comparison

Checked-in numbers are mean ± std over seeds 7, 21, and 42.
History 12 → horizon 12. Metrics are inverse-normalized flow (MAE, RMSE, R²).
Train/val/test subsample: 1600 / 400 / 500.

ICS-DGRN uses fixed recurrent weights and fixed interlayer compression, `layer_dims=[48, 36, 24]`, `compression_dims=[24, 16]`, `esp_target=0.9`.
Trainable baselines in the main table ran for 30 epochs (SmoothL1, Adam, cosine schedule, early stopping on validation MAE).
DeepESN is a ridge readout on a fixed deep reservoir (one fit, not 30 epochs of backprop).

An extra ICS-DGRN-only run at 80 epochs is in `ics_long_runs.csv`. Baselines were not extended to 80 epochs, so that file is not a matched comparison.

## Main table (30-epoch trainable models; DeepESN ridge)

| Dataset | Model | MAE | RMSE | R² | Train (s) |
|---------|-------|----:|-----:|---:|----------:|
| PEMS03 | ICS-DGRN | 20.854 ± 0.296 | 32.227 ± 0.584 | 0.949 | 93.2 |
| PEMS03 | DeepESN | 21.959 ± 0.302 | 39.625 ± 5.927 | 0.921 | 31.9 |
| PEMS03 | STGCN | 21.993 ± 0.432 | 34.160 ± 0.773 | 0.942 | 82.2 |
| PEMS03 | AGCRN | 23.042 ± 0.652 | 36.932 ± 0.879 | 0.933 | 21.3 |
| PEMS04 | ICS-DGRN | 26.182 ± 0.173 | 39.915 ± 0.184 | 0.936 | 81.1 |
| PEMS04 | DeepESN | 27.325 ± 0.161 | 41.809 ± 0.108 | 0.930 | 28.7 |
| PEMS04 | STGCN | 27.421 ± 0.173 | 42.018 ± 0.165 | 0.929 | 71.5 |
| PEMS04 | AGCRN | 28.049 ± 0.255 | 42.237 ± 0.311 | 0.928 | 19.0 |
| PEMS08 | ICS-DGRN | 21.641 ± 0.237 | 32.924 ± 0.185 | 0.949 | 54.2 |
| PEMS08 | DeepESN | 22.221 ± 0.093 | 33.859 ± 0.238 | 0.946 | 15.3 |
| PEMS08 | STGCN | 22.548 ± 0.207 | 34.263 ± 0.378 | 0.945 | 119.6 |
| PEMS08 | AGCRN | 23.467 ± 0.229 | 35.233 ± 0.427 | 0.942 | 14.1 |

Full rows, including GWNet, DCRNN, and TGCN: `main_summary.csv`. Per-seed rows: `all_metrics.csv`.

Relative MAE reduction versus the best baseline on each set (`improvement_vs_best.csv`): PEMS03 5.03% (vs DeepESN), PEMS04 4.19% (vs DeepESN), PEMS08 2.61% (vs DeepESN).

ICS-DGRN at 80 epochs (mean MAE): PEMS03 20.463, PEMS04 25.858, PEMS08 21.268. See `ics_long_runs.csv`.

## Figures

| File | Content |
|------|---------|
| `F_main_metrics.png` | MAE / RMSE across datasets |
| `F_improve_heatmap.png` | Relative MAE change vs the best baseline |
| `F_horizon.png` | MAE at horizons 1–12 |

PEMS08 trainable parameter count for ICS-DGRN is 39531 (51339 total, including fixed reservoir weights). AGCRN is 4156 and STGCN is 25740. ICS-DGRN is not the smallest or the fastest model in this table.
