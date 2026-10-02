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

"""Graph operators for Deep Graph Reservoir (DGRN / ICS-DGRN)."""

from __future__ import annotations

import pickle
from typing import Optional, Tuple, Union

import numpy as np

def load_pems_adjacency(path: str) -> np.ndarray:
    """
    Load DCRNN-style PEMS adjacency pickle:
      [sensor_ids, sensor_id_to_ind, adj_mx]
    Returns dense float64 adj_mx (N, N).
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        adj = np.asarray(obj[2], dtype=np.float64)
    elif isinstance(obj, dict) and "adj" in obj:
        adj = np.asarray(obj["adj"], dtype=np.float64)
    else:
        adj = np.asarray(obj, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Invalid adjacency shape: {adj.shape}")
    return adj

def normalize_adjacency(
    adj: np.ndarray,
    add_self_loop: bool = True,
    mode: str = "rw",
) -> np.ndarray:
    """
    Build graph propagation operator S.

    Modes
    -----
    rw   : D^{-1} A  (row-stochastic / random-walk)
    sym  : D^{-1/2} A D^{-1/2}
    none : raw A (optionally with self-loops)
    """
    a = adj.astype(np.float64).copy()
    np.fill_diagonal(a, 0.0)
    if add_self_loop:
        a = a + np.eye(a.shape[0], dtype=np.float64)

    if mode == "none":
        return a

    deg = a.sum(axis=1)
    deg = np.maximum(deg, 1e-12)

    if mode == "rw":
        return (1.0 / deg)[:, None] * a

    if mode == "sym":
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]

    raise ValueError(f"Unknown normalize mode: {mode}")

def spectral_norm(mat: np.ndarray) -> float:
    """Largest singular value (induced 2-norm)."""
    # For moderate N use SVD; for large N approximate with power iteration
    n = mat.shape[0]
    if n <= 512:
        return float(np.linalg.svd(mat, compute_uv=False)[0])
    v = np.random.randn(n)
    v /= np.linalg.norm(v) + 1e-12
    m = mat.astype(np.float64)
    for _ in range(30):
        v = m.T @ (m @ v)
        nrm = np.linalg.norm(v)
        if nrm < 1e-18:
            return 0.0
        v /= nrm
    return float(np.linalg.norm(m @ v))

def scale_operator_for_esp(
    s: np.ndarray,
    w: np.ndarray,
    target: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Scale W so that ||W||_2 * ||S||_2 <= target (2.pdf ESP condition).
    Returns (S, W_scaled, ||S||, ||W_scaled||).
    """
    ns = spectral_norm(s)
    nw = spectral_norm(w)
    if ns < 1e-12 or nw < 1e-12:
        return s, w, ns, nw
    scale = target / (ns * nw)
    w2 = w * scale
    return s, w2, ns, spectral_norm(w2)

def correlation_graph(series: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Build a correlation graph from multivariate series (T, C).
    Used when no physical adjacency is available.
    """
    x = series - series.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-12
    x = x / std
    corr = (x.T @ x) / max(series.shape[0] - 1, 1)
    np.fill_diagonal(corr, 0.0)
    adj = np.abs(corr)
    adj[adj < threshold] = 0.0
    return adj

def delay_embedding_graph(window: int) -> np.ndarray:
    """Chain graph for delay-embedding nodes (univariate ICS-style)."""
    adj = np.zeros((window, window), dtype=np.float64)
    for i in range(window - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj
