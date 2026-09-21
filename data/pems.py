# Copyright 2026 ZengWenquan
# https://github.com/chaosbull
# SPDX-License-Identifier: Apache-2.0

"""PEMS03 / PEMS04 / PEMS08, the npz split used by STSGCN and ASTGCN."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def list_pems() -> Tuple[str, ...]:
    return ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def _load_adj(path: str) -> np.ndarray:
    # DCRNN pickle: [sensor_ids, id_to_index, adj]
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        adj = np.asarray(obj[2], dtype=np.float64)
    elif isinstance(obj, dict) and "adj" in obj:
        adj = np.asarray(obj["adj"], dtype=np.float64)
    else:
        adj = np.asarray(obj, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"bad adjacency shape: {getattr(adj, 'shape', None)}")
    return adj


def load_pems(
    name: str,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    name = name.upper()
    folder = os.path.join(DATA_ROOT, name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"{folder} is missing. Put train.npz, val.npz, test.npz and adj_{name}.pkl there."
        )

    def _load(split: str):
        z = np.load(os.path.join(folder, f"{split}.npz"))
        x = z["x"][..., 0].astype(np.float32)  # (B, T, N)
        y = z["y"][..., 0].astype(np.float32)
        return x, y

    x_tr, y_tr = _load("train")
    x_va, y_va = _load("val")
    x_te, y_te = _load("test")

    rng = np.random.default_rng(seed)

    def _sub(x, y, m):
        if m is None or m >= len(x):
            return x, y
        idx = rng.choice(len(x), size=m, replace=False)
        idx.sort()
        return x[idx], y[idx]

    x_tr, y_tr = _sub(x_tr, y_tr, max_train)
    x_va, y_va = _sub(x_va, y_va, max_val)
    x_te, y_te = _sub(x_te, y_te, max_test)

    mean = float(x_tr.mean())
    std = float(x_tr.std()) + 1e-6

    def norm(a):
        return (a - mean) / std

    def denorm(a):
        return a * std + mean

    adj = _load_adj(os.path.join(folder, f"adj_{name}.pkl"))
    return {
        "name": name,
        "x_train": norm(x_tr),
        "y_train": norm(y_tr),
        "x_val": norm(x_va),
        "y_val": norm(y_va),
        "x_test": norm(x_te),
        "y_test": norm(y_te),
        "adj": adj,
        "n_nodes": int(x_tr.shape[-1]),
        "hist_len": int(x_tr.shape[1]),
        "horizon": int(y_tr.shape[1]),
        "mean": mean,
        "std": std,
        "denorm": denorm,
    }
