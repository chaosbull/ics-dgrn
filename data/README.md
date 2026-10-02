# Data layout (not shipped)

Author: chaosbull  
License: Apache-2.0  
Project: ICS-DGRN

This repository **does not** include raw PEMS tensors. Download / prepare them yourself, then place files as below.

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
- Shapes: `x,y` → `(B, T, N, C)`; loaders use channel `0` (flow) → `(B, T, N)`
- Default window: `T=12` history, `T=12` horizon (same as DCRNN / common ST baselines)

## Local development path (reference only)

If you already keep data under the private working tree:

```
f:/PythonProject4/feifa2/data/PEMS03/
f:/PythonProject4/feifa2/data/PEMS04/
f:/PythonProject4/feifa2/data/PEMS08/
```

You can symlink or copy those folders into this repo’s `data/` directory before running experiments.

## Loader entry

Python API: `data.pems_loader.load_pems(name, max_train=..., max_val=..., max_test=..., seed=...)`.
