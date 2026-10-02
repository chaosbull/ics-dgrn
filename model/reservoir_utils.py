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

"""Shared reservoir utilities: random weights, ridge readout, compression."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

def sparse_random_matrix(
    n_row: int,
    n_col: int,
    density: float = 0.1,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    density = float(np.clip(density, 1e-6, 1.0))
    w = rng.uniform(-scale, scale, size=(n_row, n_col))
    mask = rng.random((n_row, n_col)) < density
    w *= mask
    return w.astype(np.float64)

def spectral_radius(w: np.ndarray) -> float:
    if w.size == 0:
        return 0.0
    # Use eigenvalues for square matrices of moderate size
    if w.shape[0] != w.shape[1]:
        return float(np.linalg.svd(w, compute_uv=False)[0])
    n = w.shape[0]
    if n <= 256:
        ev = np.linalg.eigvals(w)
        return float(np.max(np.abs(ev)))
    # Power iteration on |W| approximation via SVD largest singular value proxy
    v = np.random.randn(n)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(40):
        v = w @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-18:
            return 0.0
        v /= nrm
    return float(np.linalg.norm(w @ v))

def scale_spectral_radius(w: np.ndarray, rho: float = 0.9) -> np.ndarray:
    r = spectral_radius(w)
    if r < 1e-12:
        return w
    return (rho / r) * w

def gaussian_measurement(
    n_in: int,
    n_out: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """ICS-DESN style Gaussian observation matrix Phi in R^{n_out x n_in}."""
    rng = rng or np.random.default_rng(0)
    phi = rng.normal(0.0, 1.0 / np.sqrt(max(n_out, 1)), size=(n_out, n_in))
    return phi.astype(np.float64)

def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    """
    Solve min ||W X - Y||_F^2 + ridge ||W||_F^2
    X: (F, N), Y: (O, N)  -> W: (O, F)
    """
    # W = Y X^T (X X^T + ridge I)^{-1}
    gram = x @ x.T
    d = gram.shape[0]
    gram.flat[:: d + 1] += ridge
    # Solve (gram) Z^T = X Y^T  => Z = W
    rhs = x @ y.T
    try:
        z = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        z = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return z.T

def ridge_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return w @ x

def add_bias(x: np.ndarray) -> np.ndarray:
    """X (F, N) -> (F+1, N) with ones row."""
    return np.vstack([x, np.ones((1, x.shape[1]), dtype=x.dtype)])
