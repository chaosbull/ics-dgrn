# Data layout (not shipped)

Author: [chaosbull](https://github.com/chaosbull)  
License: Apache-2.0  
Project: ICS-DGRN

This repository does not include raw PEMS tensors.

## Source

PEMS03, PEMS04, PEMS07, and PEMS08 follow the layout used by [STLGRU](https://github.com/Kishor-Bhaumik/STLGRU) (Kishor Kumar Bhaumik et al.). That repository documents the Baidu Drive / Google Drive download links and the DCRNN-style `train.npz` / `val.npz` / `test.npz` plus adjacency pickle.

METR-LA and PEMS-BAY, if needed, are the files linked from the same README (DCRNN Google Drive / Baidu Yun).

## Expected paths

```
data/
  PEMS03/
    train.npz
    val.npz
    test.npz
    adj_mx.pkl          # or adj_*.pkl; DCRNN pickle, adjacency is the 3rd element
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
- Shapes: `x, y` -> `(B, T, N, C)`. Loaders use channel `0` (flow) -> `(B, T, N)`
- Default window: 12 history steps, 12-step horizon

## Loader entry

`data.pems_loader.load_pems(name, max_train=..., max_val=..., max_test=..., seed=...)`.
