# ICS-DGRN

Interlayer Sparse Compression Deep Graph Reservoir Network for short-term traffic flow forecasting.

| | |
|---|---|
| Author | [chaosbull](https://github.com/chaosbull) |
| License | [Apache License 2.0](LICENSE) |
| Raw PEMS data | not included; prepared as in [STLGRU](https://github.com/Kishor-Bhaumik/STLGRU). See [data/README.md](data/README.md) |

ICS-DGRN stacks graph reservoir layers and compresses each layer with a fixed Gaussian map before the next layer. Recurrent weights are scaled once so that the echo-state sufficient condition holds (`esp_target=0.9` by default), then frozen. Input maps, leak, and the readout are trained. The checked-in model uses three reservoir widths `[48, 36, 24]` and compression widths `[24, 16]`.

---

## Repository layout

```
ics-dgrn/
|-- LICENSE
|-- NOTICE
|-- AUTHORS
|-- README.md
|-- requirements.txt
|-- main.py
|-- configs/default.yaml
|-- model/                  # ICS-DGRN, ICS-DESN, ESN, graph operators
|-- baselines/              # STGCN, TGCN, GWNet, AGCRN, DCRNN, ML baselines
|-- data/                   # loaders only; place PEMS npz/pkl here
|-- experiments/            # training and evaluation scripts
|-- utils/                  # metrics, plots, ESP helpers
|-- checkpoints/            # optional weights (none shipped)
`-- result/
    |-- pems_st200/         # multi-seed DL comparison
    |-- pems_newexp/        # ablation, compression, ESP, efficiency
    |-- pems/               # ridge reservoir comparison
    |-- timeseries/         # synthetic and public sequence runs
    `-- REPORT.md
```

No paper sources and no raw traffic tensors are included.

---

## Environment

```bash
conda create -n py312 python=3.12 -y
conda activate py312
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

PyTorch is not pinned in `requirements.txt`. Use a CUDA build that matches the driver.

---

## Data

Download PEMS03, PEMS04, PEMS07, and PEMS08 using the links in [STLGRU](https://github.com/Kishor-Bhaumik/STLGRU) (Baidu Drive password is given in that README). Place the DCRNN-style splits here:

```
data/PEMS03/train.npz  val.npz  test.npz  adj_*.pkl
data/PEMS04/...
data/PEMS08/...
```

Loaders read `x` and `y` with shape `(B, T, N, C)` and keep channel 0. Details: [data/README.md](data/README.md).

---

## Reproduce

```bash
conda activate py312

# Multi-seed comparison. Fresh files go to result/p0_support/
python -u experiments/run_p0_bundle.py --stage all

# ICS-DGRN only, 80 epochs (does not retrain baselines)
python -u experiments/run_ics_longer.py

# Ridge reservoir comparison -> result/pems/
python -u experiments/run_pems.py

# Sequence benchmarks -> result/timeseries/
python -u experiments/run_timeseries.py

# Earlier single-seed DL loop -> result/pems_st200/ when run as-is
python -u experiments/run_pems_200ep.py --epochs 300 --datasets PEMS08 PEMS04 PEMS03
```

`run_pems_200ep.py` overwrites `result/pems_st200/`. The tables already in that folder are the multi-seed run described below, not a 300-epoch single seed. Copy the folder aside before launching the older script.

Default multi-seed protocol: history 12, horizon 12, seeds 7 / 21 / 42, subsample 1600 / 400 / 500, SmoothL1, Adam with cosine decay. Trainable baselines in the main table used 30 epochs. DeepESN fits a ridge readout on a frozen reservoir.

---

## Results

Numbers below are mean ± standard deviation over three seeds, on inverse-normalized flow. Source tables: [`result/pems_st200/main_summary.csv`](result/pems_st200/main_summary.csv) and [`result/pems_st200/REPORT.md`](result/pems_st200/REPORT.md).

### PEMS08

| Model | MAE | RMSE | R² | Train (s) | Trainable params |
|-------|----:|-----:|---:|----------:|-----------------:|
| ICS-DGRN | 21.641 ± 0.237 | 32.924 ± 0.185 | 0.949 | 54.2 | 39531 |
| DeepESN | 22.221 ± 0.093 | 33.859 ± 0.238 | 0.946 | 15.3 | 2616 |
| STGCN | 22.548 ± 0.207 | 34.263 ± 0.378 | 0.945 | 119.6 | 25740 |
| AGCRN | 23.467 ± 0.229 | 35.233 ± 0.427 | 0.942 | 14.1 | 4156 |
| GWNet | 23.765 ± 0.355 | 35.620 ± 0.485 | 0.940 | 36.7 | 12876 |
| DCRNN | 24.286 ± 0.304 | 36.293 ± 0.362 | 0.938 | 34.7 | 9996 |
| TGCN | 40.826 ± 0.476 | 63.065 ± 0.849 | 0.813 | 19.5 | 3564 |

### PEMS04

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| ICS-DGRN | 26.182 ± 0.173 | 39.915 ± 0.184 | 0.936 |
| DeepESN | 27.325 ± 0.161 | 41.809 ± 0.108 | 0.930 |
| STGCN | 27.421 ± 0.173 | 42.018 ± 0.165 | 0.929 |
| AGCRN | 28.049 ± 0.255 | 42.237 ± 0.311 | 0.928 |
| GWNet | 28.260 ± 0.274 | 42.979 ± 0.441 | 0.926 |
| DCRNN | 29.230 ± 0.160 | 43.983 ± 0.025 | 0.922 |
| TGCN | 35.762 ± 0.354 | 54.623 ± 0.785 | 0.880 |

### PEMS03

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| ICS-DGRN | 20.854 ± 0.296 | 32.227 ± 0.584 | 0.949 |
| DeepESN | 21.959 ± 0.302 | 39.625 ± 5.927 | 0.921 |
| STGCN | 21.993 ± 0.432 | 34.160 ± 0.773 | 0.942 |
| AGCRN | 23.042 ± 0.652 | 36.932 ± 0.879 | 0.933 |
| GWNet | 23.441 ± 0.367 | 35.464 ± 0.640 | 0.938 |
| DCRNN | 23.487 ± 0.498 | 37.448 ± 0.751 | 0.931 |
| TGCN | 39.732 ± 0.915 | 57.671 ± 1.296 | 0.835 |

Against the lowest-MAE baseline on each dataset (DeepESN in all three), ICS-DGRN reduces MAE by 5.03% (PEMS03), 4.19% (PEMS04), and 2.61% (PEMS08). A separate 80-epoch ICS-DGRN run, without retraining the baselines, reaches mean MAE 20.463 / 25.858 / 21.268 on PEMS03 / PEMS04 / PEMS08.

On PEMS08, a single reservoir layer is worse than the three-layer model (MAE 21.874 vs 21.641). Removing graph propagation or compression does not increase MAE on that split, and a rewired graph is not worse than the sensor graph. Training time is lower than STGCN on PEMS08 (54 s vs 120 s) and higher than AGCRN (14 s). Fixed reservoir weights make the stored parameter count larger than the trainable count (51339 vs 39531 on PEMS08).

ESP scaling with `tau <= 0.9` meets the spectral sufficient condition, and the measured state difference grows as `tau` increases. That sweep does not show a matching rise in forecast MAE.

---

## Citation

```
Author: chaosbull
Project: ICS-DGRN
https://github.com/chaosbull/ics-dgrn
License: Apache-2.0
```

For the PEMS files, also cite [STLGRU](https://github.com/Kishor-Bhaumik/STLGRU) and the original PEMS providers.

---

## License

Copyright 2024-2026 chaosbull

Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
