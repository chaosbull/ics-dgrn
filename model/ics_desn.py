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

"""
Interlayer Sparse Compression DeepESN (ICS-DESN)
following Wang et al., IEEE TNNLS 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .reservoir_utils import (
    add_bias,
    gaussian_measurement,
    ridge_fit,
    ridge_predict,
    scale_spectral_radius,
    sparse_random_matrix,
)

@dataclass
class ICSDESNConfig:
    layer_sizes: tuple = (100, 80, 60)
    compression_dims: tuple = (40, 30)  # length = L-1
    spectral_radius: float = 0.9
    leak: float = 0.5
    input_scale: float = 0.5
    density: float = 0.1
    ridge: float = 1e-4
    washout: int = 50
    seed: int = 42
    use_bias: bool = True
    concatenate_layers: bool = True

class ICSDESN:
    """
    DeepESN with Gaussian compressive sensing between layers:
      x_new = Phi @ x_old ,  Phi ~ N(0, 1/m)
    """

    def __init__(self, n_inputs: int, n_outputs: int, cfg: Optional[ICSDESNConfig] = None):
        self.cfg = cfg or ICSDESNConfig()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        rng = np.random.default_rng(self.cfg.seed)
        sizes = list(self.cfg.layer_sizes)
        self.sizes = sizes
        comps = list(self.cfg.compression_dims)
        if len(comps) < len(sizes) - 1:
            comps = comps + [max(sizes[-1] // 2, 8)] * (len(sizes) - 1 - len(comps))
        self.comps = comps[: max(len(sizes) - 1, 0)]

        self.wins: List[np.ndarray] = []
        self.ws: List[np.ndarray] = []
        self.phis: List[Optional[np.ndarray]] = [None]
        prev_dim = n_inputs
        for i, n in enumerate(sizes):
            win = rng.uniform(-self.cfg.input_scale, self.cfg.input_scale, size=(n, prev_dim))
            w = sparse_random_matrix(n, n, density=self.cfg.density, rng=rng)
            w = scale_spectral_radius(w, self.cfg.spectral_radius)
            self.wins.append(win)
            self.ws.append(w)
            if i < len(sizes) - 1:
                m = self.comps[i]
                phi = gaussian_measurement(n, m, rng=rng)
                self.phis.append(phi)
                prev_dim = m
            else:
                prev_dim = n
        self.states = [np.zeros(n, dtype=np.float64) for n in sizes]
        self.wout: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.states = [np.zeros(n, dtype=np.float64) for n in self.sizes]

    def compress(self, layer_idx: int, x: np.ndarray) -> np.ndarray:
        phi = self.phis[layer_idx + 1] if layer_idx + 1 < len(self.phis) else None
        if phi is None:
            return x
        return phi @ x

    def step(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        inp = u
        outs = []
        for i, n in enumerate(self.sizes):
            pre = self.wins[i] @ inp + self.ws[i] @ self.states[i]
            x = (1.0 - self.cfg.leak) * self.states[i] + self.cfg.leak * np.tanh(pre)
            self.states[i] = x
            outs.append(x)
            if i < len(self.sizes) - 1:
                inp = self.compress(i, x)
        if self.cfg.concatenate_layers:
            return np.concatenate(outs)
        return outs[-1]

    def collect_states(self, u_seq: np.ndarray, washout: Optional[int] = None) -> np.ndarray:
        washout = self.cfg.washout if washout is None else washout
        self.reset()
        states = []
        for t in range(len(u_seq)):
            x = self.step(u_seq[t])
            if t >= washout:
                states.append(x.copy())
        return np.stack(states, axis=1)

    def fit(self, u_seq: np.ndarray, y_seq: np.ndarray) -> "ICSDESN":
        wash = self.cfg.washout
        x = self.collect_states(u_seq, washout=wash)
        y = y_seq[wash:].T
        if self.cfg.use_bias:
            x = add_bias(x)
        self.wout = ridge_fit(x, y, ridge=self.cfg.ridge)
        return self

    def predict(self, u_seq: np.ndarray, washout: int = 0) -> np.ndarray:
        x = self.collect_states(u_seq, washout=washout)
        if self.cfg.use_bias:
            x = add_bias(x)
        return ridge_predict(self.wout, x).T
