# Data layout (not shipped)

Author: chaosbull  
License: Apache-2.0  
Project: ICS-DGRN

This repository **does not** include raw PEMS tensors.

## Source

Use the PEMS03 / PEMS04 / PEMS07 / PEMS08 preparation instructions from:

- [STLGRU (Kishor-Bhaumik/STLGRU)](https://github.com/Kishor-Bhaumik/STLGRU)

That project documents download links (Baidu Drive / Google Drive) for PEMS and related traffic datasets in DCRNN-style format.

## Expected paths

```
data/
  PEMS03/
    train.npz
    val.npz
    test.npz
    adj_mx.pkl          # or adj_*.pkl (DCRNN-style pickle; adjacency is the 3rd element)
  PEMS04/
    train.npz
    val.npz
    test.npz
    adj_*.pkl
  PEMS08/
    train.npz
    val.npz
    test.npz
    adj_*.pkl
```

## Array conventions

- `train.npz` / `val.npz` / `test.npz` keys: `x`, `y`
- Shapes: `x,y` -> `(B, T, N, C)`; loaders use channel `0` (flow) -> `(B, T, N)`
- Default window: `T=12` history, `T=12` horizon (same as DCRNN / common ST baselines)

## Loader entry

Python API: `data.pems_loader.load_pems(name, max_train=..., max_val=..., max_test=..., seed=...)`.
