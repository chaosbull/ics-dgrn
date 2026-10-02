# Copyright 2024-2026 chaosbull
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: chaosbull
# Project: ICS-DGRN

"""PEMS03/04/07/08 loaders (DCRNN / STGCN-style windowed traffic data)."""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.graph_ops import load_pems_adjacency

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

def pems_dir(name: str) -> str:
    return os.path.join(DATA_ROOT, name.upper())

def load_pems(
    name: str,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 2025,
) -> Dict:
    """
    Returns dict with:
      x_train/y_train/... : (B, 12, N)
      adj : raw adjacency
      mean, std : scaler fitted on train
    """
    name = name.upper()
    d = pems_dir(name)
    if not os.path.isdir(d):
        raise FileNotFoundError(d)

    def _load(split: str):
        z = np.load(os.path.join(d, f"{split}.npz"))
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

    adj_path = os.path.join(d, f"adj_{name}.pkl")
    adj = load_pems_adjacency(adj_path)

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

def list_pems() -> Tuple[str, ...]:
    return ("PEMS03", "PEMS04", "PEMS07", "PEMS08")
