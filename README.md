# ICS-DGRN

**Interlayer Sparse Compression Deep Graph Reservoir Network** for short-term traffic flow forecasting.

| | |
|---|---|
| Author | [chaosbull](https://github.com/chaosbull) |
| License | [Apache License 2.0](LICENSE) |
| Code + results | this repository |
| Raw PEMS data | **not** shipped; see [data/README.md](data/README.md) |

ICS-DGRN combines deep graph reservoirs with interlayer Gaussian compression under an ESP (Echo State Property) spectral constraint. Recurrent weights are fixed after ESP scaling by default; input maps, compression, leak, and readout are trained.

---

## Repository layout

```
ics-dgrn/
|-- LICENSE
|-- NOTICE
|-- README.md
|-- requirements.txt
|-- main.py
|-- configs/default.yaml
|-- model/                  # ICS-DGRN / ICS-DESN / ESN / graph ops
|-- baselines/              # STGCN, TGCN, GWNet, AGCRN (+ ML baselines)
|-- data/                   # loaders only; place PEMS npz/pkl here
|-- experiments/            # training and evaluation scripts
|-- utils/                  # metrics, viz, ESP helpers
|-- checkpoints/            # optional *.pt (none exported in this release)
`-- result/                 # figures and CSV tables
    |-- pems_st200/         # main DL comparison (300 epoch)
    |-- pems_newexp/        # mechanism experiments
    |-- pems/               # Ridge reservoir + ablations
    |-- timeseries/         # synthetic / ETTh1 / weather RC runs
    `-- REPORT.md
```

No PDF papers and no raw traffic tensors are included.

---

## Results included

| Path | Content |
|------|---------|
| [`result/pems_st200/`](result/pems_st200/) | ICS-DGRN vs STGCN / TGCN / GWNet / AGCRN on PEMS03/04/08 (300 epoch, CUDA) |
| [`result/pems_newexp/`](result/pems_newexp/) | Compression ratio, sparse density, ESP state decay, ESP-forecast joint, fixed vs trainable, multi-seed, multi-horizon, compute, qualitative |
| [`result/pems/`](result/pems/) | Ridge / classical RC comparison and graph-ICS ablations |
| [`result/timeseries/`](result/timeseries/) | Non-PEMS sequence experiments |

Figure guide for the mechanism suite: [`result/pems_newexp/REPORT.md`](result/pems_newexp/REPORT.md).  
Main DL table: [`result/pems_st200/REPORT.md`](result/pems_st200/REPORT.md).

**Checkpoints (`.pt`)**: this release does not contain trained weight files. Metrics and figures were saved during training. Retrain with the scripts below, or add `torch.save` under `checkpoints/` if needed.

---

## Environment

```bash
conda create -n py312 python=3.12 -y
conda activate py312
# install a CUDA build of PyTorch matching your driver, then:
pip install -r requirements.txt
```

Verified with PyTorch 2.5.x + CUDA 12.x on an NVIDIA GPU.

---

## Data

Raw PEMS tensors are **not** in this repo. Prepare them following the PEMS layout used by [STLGRU](https://github.com/Kishor-Bhaumik/STLGRU) (PEMS03 / PEMS04 / PEMS07 / PEMS08; DCRNN-style `train/val/test.npz` + adjacency pickle). See that repository and its linked Baidu Drive / Google Drive sources for downloads.

Place files under `data/`:

```
data/PEMS03/train.npz  val.npz  test.npz  adj_*.pkl
data/PEMS04/...
data/PEMS08/...
```

Details: [`data/README.md`](data/README.md).

---

## Reproduce experiments

```bash
conda activate py312
cd ics-dgrn

# Main DL comparison -> result/pems_st200/
python -u experiments/run_pems_200ep.py --epochs 300 --datasets PEMS08 PEMS04 PEMS03

# Mechanism suite -> result/pems_newexp/
python -u experiments/run_new_experiments.py --exps all --epochs 300 --datasets PEMS08

# Ridge / ablation -> result/pems/
python -u experiments/run_pems.py

# Optional timeseries RC -> result/timeseries/
python -u experiments/run_timeseries.py

# Or via main.py
python main.py --task pems_st --epochs 300
```

Default DL protocol: history 12 -> horizon 12; SmoothL1; Adam + CosineAnnealing; subsample train/val/test = 800/150/250; ICS-DGRN `layer_dims=[56,40]`, `compression_dims=[28]`, `esp_target=0.9`.

---

## Citation

If you use this code or results, please credit:

```
Author: chaosbull
Project: ICS-DGRN
License: Apache-2.0
```

Traffic datasets: please also cite / acknowledge [STLGRU](https://github.com/Kishor-Bhaumik/STLGRU) and the original PEMS data providers as appropriate.

---

## License

Copyright 2024-2026 chaosbull

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
