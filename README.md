# ICS-DGRN

**Interlayer Sparse Compression Deep Graph Reservoir Network** for short-term traffic flow forecasting.

| | |
|---|---|
| Author | [chaosbull](https://github.com/chaosbull) |
| License | [Apache License 2.0](LICENSE) |
| Code + results | this repository |
| Raw PEMS data | **not** shipped 鈥?see [data/README.md](data/README.md) |

ICS-DGRN combines deep graph reservoirs with interlayer Gaussian compression, under an ESP (Echo State Property) spectral constraint. Recurrent weights are fixed after ESP scaling by default; input maps, compression, leak, and readout are trained.

---

## Repository layout

```
ics-dgrn/
鈹溾攢鈹€ LICENSE                 # Apache-2.0
鈹溾攢鈹€ NOTICE
鈹溾攢鈹€ README.md               # this file
鈹溾攢鈹€ requirements.txt
鈹溾攢鈹€ main.py                 # unified entry
鈹溾攢鈹€ configs/default.yaml
鈹溾攢鈹€ model/                  # ICS-DGRN / ICS-DESN / ESN / graph ops
鈹溾攢鈹€ baselines/              # STGCN, TGCN, GWNet, AGCRN (+ ML baselines)
鈹溾攢鈹€ data/                   # loaders only; place PEMS npz/pkl here (see data/README.md)
鈹溾攢鈹€ experiments/            # training & evaluation scripts
鈹溾攢鈹€ utils/                  # metrics, viz, ESP helpers
鈹溾攢鈹€ checkpoints/            # optional *.pt (none exported in this release)
鈹斺攢鈹€ result/                 # figures + CSV tables from runs
    鈹溾攢鈹€ pems_st200/         # main DL comparison (300 epoch)
    鈹溾攢鈹€ pems_newexp/        # mechanism experiments (compression / ESP / multi-seed / 鈥?
    鈹溾攢鈹€ pems/               # Ridge reservoir + ablations
    鈹溾攢鈹€ timeseries/         # synthetic / ETTh1 / weather style RC runs
    鈹斺攢鈹€ REPORT.md
```

No PDF papers and no raw traffic tensors are included.

---

## Results included

| Path | Content |
|------|---------|
| [`result/pems_st200/`](result/pems_st200/) | ICS-DGRN vs STGCN / TGCN / GWNet / AGCRN on PEMS03/04/08 (300 epoch, CUDA) |
| [`result/pems_newexp/`](result/pems_newexp/) | Compression ratio, sparse density, ESP state decay, ESP鈥揻orecast joint, fixed vs trainable, multi-seed, multi-horizon, compute, qualitative |
| [`result/pems/`](result/pems/) | Ridge / classical RC comparison and graph鈥揑CS ablations |
| [`result/timeseries/`](result/timeseries/) | Non-PEMS sequence experiments |

Figure-by-figure notes for the mechanism suite: [`result/pems_newexp/REPORT.md`](result/pems_newexp/REPORT.md).  
Main DL table draft: [`result/pems_st200/REPORT.md`](result/pems_st200/REPORT.md).

**Checkpoints (`.pt`)**: this release does not contain trained weight files. Models were evaluated during training and only metrics/figures were saved. Use the scripts below to retrain; you may add `torch.save` under `checkpoints/` if needed.

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

## Data (local path only)

Place DCRNN-style PEMS files under `data/`:

```
data/PEMS03/train.npz  val.npz  test.npz  adj_*.pkl
data/PEMS04/...
data/PEMS08/...
```

Details and expected keys: [`data/README.md`](data/README.md).  
Local development copy of the tensors (not in git): `f:/PythonProject4/feifa2/data/PEMS0X/`.

---

## Reproduce experiments

```bash
conda activate py312
cd ics-dgrn-release

# Main DL comparison 鈫?result/pems_st200/
python -u experiments/run_pems_200ep.py --epochs 300 --datasets PEMS08 PEMS04 PEMS03

# Mechanism suite 鈫?result/pems_newexp/
python -u experiments/run_new_experiments.py --exps all --epochs 300 --datasets PEMS08

# Ridge / ablation 鈫?result/pems/
python -u experiments/run_pems.py

# Optional timeseries RC 鈫?result/timeseries/
python -u experiments/run_timeseries.py

# Or via main.py
python main.py --task pems_st --epochs 300
```

Default protocol for DL runs: history 12 鈫?horizon 12; SmoothL1; Adam + CosineAnnealing; subsample train/val/test = 800/150/250; ICS-DGRN `layer_dims=[56,40]`, `compression_dims=[28]`, `esp_target=0.9`.

---

## Citation

If you use this code or results, please cite the associated paper and credit:

```
Author: chaosbull
Project: ICS-DGRN
License: Apache-2.0
```

---

## License

Copyright 2024-2026 chaosbull  

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
