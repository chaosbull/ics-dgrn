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

"""Unified data loading for DGRN experiments (ICS-DESN style datasets)."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .prepare_synthetic import logistic_map, lorenz_system, synthetic_sunspot

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")

def list_datasets() -> Dict[str, str]:
    return {
        "logistic": "Chaotic logistic map (synthetic)",
        "lorenz": "Lorenz attractor xyz (synthetic)",
        "sunspot": "Sunspot proxy / local csv",
        "ETTh1": "Electricity Transformer Temperature (hourly)",
        "ETTh2": "ETT hour #2",
        "ETTm1": "ETT minute #1",
        "ETTm2": "ETT minute #2",
        "weather": "Weather multivariate (Jena-style)",
    }

def _minmax(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    return (x - mn) / (mx - mn + 1e-12), mn, mx

def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")

def load_raw(name: str) -> np.ndarray:
    """
    Return float array:
      univariate -> (T, 1)
      multivariate -> (T, C)
    """
    name = name.lower()
    if name == "logistic":
        path = os.path.join(DATA_DIR, "logistic.csv")
        if os.path.exists(path):
            return _read_csv("logistic.csv")[["value"]].to_numpy(float)
        return logistic_map().reshape(-1, 1)

    if name == "lorenz":
        path = os.path.join(DATA_DIR, "lorenz.csv")
        if os.path.exists(path):
            return _read_csv("lorenz.csv")[["x", "y", "z"]].to_numpy(float)
        return lorenz_system()

    if name == "sunspot":
        path = os.path.join(DATA_DIR, "sunspot.csv")
        if os.path.exists(path):
            df = _read_csv("sunspot.csv")
            col = "sunspot" if "sunspot" in df.columns else df.columns[-1]
            return df[[col]].to_numpy(float)
        return synthetic_sunspot().reshape(-1, 1)

    if name.startswith("ett"):
        # ETTh1 / ETTh2 / ETTm1 / ETTm2
        key = {"etth1": "ETTh1.csv", "etth2": "ETTh2.csv", "ettm1": "ETTm1.csv", "ettm2": "ETTm2.csv"}[name]
        df = _read_csv(key)
        cols = [c for c in df.columns if c.lower() != "date"]
        return df[cols].to_numpy(float)

    if name == "weather":
        df = _read_csv("weather.csv")
        cols = [c for c in df.columns if c.lower() != "date"]
        # Keep a manageable subset of channels for reservoir experiments
        if len(cols) > 12:
            prefer = [c for c in cols if any(k in c.lower() for k in ["t (", "rh", "rain", "wv", "p ("])]
            cols = prefer[:10] if prefer else cols[:10]
        return df[cols].to_numpy(float)

    raise ValueError(f"Unknown dataset: {name}. Options: {list(list_datasets())}")

def make_forecast_tensors(
    series: np.ndarray,
    window: int = 24,
    horizon: int = 1,
    stride: int = 1,
    target_col: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window supervised pairs (ICS / LTSF style).

    Returns
    -------
    X : (F, num_windows)  flattened window features (C*window, samples)
    Y : (horizon * n_targets, num_windows)
        For multivariate default: predict all channels at next step => (C, samples) if horizon=1
        Here we predict `target_col` only when series is multi and horizon>=1 — actually
        for fair ICS-DESN ETTh1 match we predict OT (last col) one-step.
    """
    T, C = series.shape
    n = (T - window - horizon + 1) // stride
    X = np.zeros((C * window, n))
    Y = np.zeros((horizon, n))
    for i in range(n):
        s = i * stride
        e = s + window
        X[:, i] = series[s:e, :].reshape(-1)
        # one-step / multi-step univariate target (default last channel)
        Y[:, i] = series[e : e + horizon, target_col]
    return X, Y

def load_dataset(
    name: str,
    window: int = 24,
    horizon: int = 1,
    train_ratio: float = 0.8,
    normalize: bool = True,
    target_col: int = -1,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Unified loader returning arrays ready for reservoir models.

    Also returns `node_series` (T, N) for graph construction:
      - multivariate: use channels as nodes
      - univariate: delay-embedding nodes of size `window`
    """
    raw = load_raw(name)
    if max_samples is not None and raw.shape[0] > max_samples:
        raw = raw[:max_samples]

    if normalize:
        raw, mn, mx = _minmax(raw)
    else:
        mn = mx = None

    T, C = raw.shape
    X, Y = make_forecast_tensors(raw, window=window, horizon=horizon, target_col=target_col)
    n = X.shape[1]
    n_train = int(n * train_ratio)
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, n)

    if C == 1:
        # delay-embedding graph nodes
        node_series = np.zeros((T - window + 1, window))
        for i in range(node_series.shape[0]):
            node_series[i] = raw[i : i + window, 0]
        n_nodes = window
    else:
        node_series = raw
        n_nodes = C

    return {
        "name": name,
        "raw": raw,
        "X": X,
        "Y": Y,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_nodes": n_nodes,
        "n_channels": C,
        "window": window,
        "horizon": horizon,
        "node_series": node_series,
        "minmax": (mn, mx),
    }
