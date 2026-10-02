# ICS-DGRN Experiment Report

## 1. Method
- **ICS-DGRN**: Deep Graph Reservoir (ESP-constrained) + interlayer Gaussian compression (ICS-DESN style).
- Training: fixed reservoir + ridge readout (RC / classical ML level, same as ICS-DESN).
- ESP condition enforced per layer: `||W_l||_2 * ||S||_2 < 1`.

## 2. Spatiotemporal (PEMS) results

Baselines: HA, LastRepeat, Ridge, ESN, DeepESN, ICS-DESN, GraphESN, plus ablations DGRN-NoICS / DGRN-NoGraph.

### PEMS03 (sorted by MAE)

| model | MAE | RMSE | MSE | R2 | TimeSec |
| --- | --- | --- | --- | --- | --- |
| ICS-DGRN | 21.9844 | 34.449 | 1186.74 | 0.941 | 7.8 |
| DGRN-NoICS | 22.1545 | 40.716 | 1657.79 | 0.9175 | 7.768 |
| GraphESN | 22.224 | 39.8316 | 1586.56 | 0.9211 | 4.1395 |
| DGRN-NoGraph | 22.3836 | 43.6458 | 1904.96 | 0.9052 | 7.6807 |
| LastRepeat | 24.5503 | 38.5881 | 1489.04 | 0.9259 | 0.0471 |
| Ridge | 25.9673 | 39.9163 | 1593.31 | 0.9207 | 0.1829 |
| HA | 30.8904 | 48.637 | 2365.56 | 0.8823 | 0.0437 |
| ESN | 49.7891 | 74.9559 | 5618.38 | 0.7204 | 0.3333 |
| ICS-DESN | 53.3055 | 80.4397 | 6470.55 | 0.678 | 0.3455 |
| DeepESN | 53.4331 | 80.6215 | 6499.82 | 0.6766 | 0.3575 |

- ICS-DGRN MAE=21.9844, ICS-DESN MAE=53.3055, relative MAE reduction=58.76%.
- Best on this split: **ICS-DGRN**.

### PEMS04 (sorted by MAE)

| model | MAE | RMSE | MSE | R2 | TimeSec |
| --- | --- | --- | --- | --- | --- |
| DGRN-NoICS | 27.7012 | 42.6512 | 1819.12 | 0.9277 | 9.5995 |
| ICS-DGRN | 27.7575 | 42.7745 | 1829.66 | 0.9273 | 9.6848 |
| DGRN-NoGraph | 27.7576 | 42.764 | 1828.76 | 0.9273 | 9.9077 |
| GraphESN | 28.0385 | 43.2192 | 1867.9 | 0.9258 | 5.4913 |
| Ridge | 31.6533 | 48.3111 | 2333.97 | 0.9073 | 0.1702 |
| LastRepeat | 31.9551 | 49.4435 | 2444.66 | 0.9029 | 0.0485 |
| ESN | 33.1856 | 51.5192 | 2654.22 | 0.8946 | 0.3875 |
| DeepESN | 33.4913 | 52.1051 | 2714.94 | 0.8921 | 0.4466 |
| ICS-DESN | 33.4919 | 52.077 | 2712.02 | 0.8923 | 0.4677 |
| HA | 39.3575 | 58.6603 | 3441.03 | 0.8633 | 0.0496 |

- ICS-DGRN MAE=27.7575, ICS-DESN MAE=33.4919, relative MAE reduction=17.12%.
- Best on this split: **DGRN-NoICS**.

### PEMS08 (sorted by MAE)

| model | MAE | RMSE | MSE | R2 | TimeSec |
| --- | --- | --- | --- | --- | --- |
| DGRN-NoGraph | 22.2584 | 34.5104 | 1190.97 | 0.9427 | 5.1293 |
| ICS-DGRN | 22.2636 | 34.6752 | 1202.37 | 0.9422 | 5.1626 |
| DGRN-NoICS | 22.2843 | 34.5899 | 1196.46 | 0.9425 | 5.1133 |
| GraphESN | 22.4279 | 35.0129 | 1225.9 | 0.941 | 2.382 |
| ICS-DESN | 24.3885 | 36.6199 | 1341.02 | 0.9355 | 0.4829 |
| DeepESN | 24.4086 | 36.6808 | 1345.48 | 0.9353 | 0.4796 |
| ESN | 24.5421 | 36.8943 | 1361.19 | 0.9345 | 0.318 |
| LastRepeat | 25.4043 | 39.8421 | 1587.39 | 0.9236 | 0.0289 |
| Ridge | 27.9975 | 41.5735 | 1728.36 | 0.9169 | 0.1066 |
| HA | 31.9806 | 48.6183 | 2363.74 | 0.8863 | 0.0304 |

- ICS-DGRN MAE=22.2636, ICS-DESN MAE=24.3885, relative MAE reduction=8.71%.
- Best on this split: **DGRN-NoGraph**.

## 3. Time-series (ICS-DESN paper datasets)

### ETTh1

| model | MSE | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| ICS-DGRN | 0.000599 | 0.017497 | 0.024464 | 0.933353 |
| ESN | 0.000618 | 0.017949 | 0.024855 | 0.931209 |
| Ridge | 0.000716 | 0.020031 | 0.026752 | 0.920751 |
| DeepESN | 0.003256 | 0.030038 | 0.057058 | 0.637463 |
| ICS-DESN | 0.070516 | 0.08406 | 0.265549 | -6.85235 |

### logistic

| model | MSE | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| ICS-DESN | 0.000954 | 0.014722 | 0.030886 | 0.99261 |
| ESN | 0.001041 | 0.023178 | 0.032269 | 0.991934 |
| DeepESN | 0.0014 | 0.019406 | 0.03741 | 0.989158 |
| ICS-DGRN | 0.008985 | 0.080701 | 0.094791 | 0.930434 |
| Ridge | 0.131277 | 0.326158 | 0.362322 | -0.014951 |

### lorenz

| model | MSE | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| ICS-DESN | 0 | 4.1e-05 | 5.5e-05 | 1 |
| DeepESN | 0 | 4.7e-05 | 8.5e-05 | 1 |
| ESN | 0 | 0.000102 | 0.000178 | 0.999999 |
| ICS-DGRN | 1e-06 | 0.000526 | 0.000723 | 0.999982 |
| Ridge | 8.2e-05 | 0.005781 | 0.009071 | 0.99716 |

### sunspot

| model | MSE | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| ICS-DESN | 0.003576 | 0.046956 | 0.0598 | 0.946953 |
| DeepESN | 0.003614 | 0.047246 | 0.060116 | 0.946391 |
| ESN | 0.003702 | 0.048009 | 0.060847 | 0.945079 |
| ICS-DGRN | 0.003902 | 0.049154 | 0.062466 | 0.941895 |
| Ridge | 0.004115 | 0.050622 | 0.064145 | 0.938616 |

### weather

| model | MSE | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| DeepESN | 3e-06 | 0.001586 | 0.001586 | -1.76039e+09 |
| ICS-DESN | 4e-06 | 0.001925 | 0.001925 | -2.59469e+09 |
| ESN | 1.1e-05 | 0.003242 | 0.003242 | -7.35543e+09 |
| Ridge | 0.000125 | 0.007196 | 0.011187 | -8.71092e+10 |
| ICS-DGRN | 0.000143 | 0.008169 | 0.01195 | -9.99534e+10 |

## 4. Figure / table locations

- `result/pems/<DATASET>/` : bars, radar, heatmap, horizon curves, ESP decay, adjacency, predictions
- `result/timeseries/<DATASET>/` : ICS-DESN-style curves and metric tables
- `result/REPORT.md` : this file
