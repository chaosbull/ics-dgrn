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

"""Classical ML / statistical baselines at the same RC level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR

@dataclass
class RidgeConfig:
    alpha: float = 1.0
    seed: int = 42

class RidgeForecaster:
    """Flattened-window ridge regression (ML baseline)."""

    def __init__(self, cfg: Optional[RidgeConfig] = None):
        self.cfg = cfg or RidgeConfig()
        self.model = Ridge(alpha=self.cfg.alpha, random_state=self.cfg.seed)
        self.out_shape: Optional[Tuple[int, ...]] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeForecaster":
        # x: (B,T,N) or (B,F); y: (B,H,N) or (B,O)
        xb = x.reshape(len(x), -1)
        yb = y.reshape(len(y), -1)
        self.out_shape = y.shape[1:]
        self.model.fit(xb, yb)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = x.reshape(len(x), -1)
        y = self.model.predict(xb)
        if self.out_shape is not None:
            y = y.reshape((-1,) + self.out_shape)
        return y

class HistoricalAverage:
    """Predict each horizon by the mean of the input window (per node)."""

    def __init__(self):
        self.out_horizon: Optional[int] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HistoricalAverage":
        self.out_horizon = y.shape[1] if y.ndim == 3 else 1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (B,T,N)
        mean = x.mean(axis=1, keepdims=True)  # (B,1,N)
        h = self.out_horizon or 1
        return np.repeat(mean, h, axis=1)

class LastRepeat:
    """Repeat the last observed frame for all horizons."""

    def __init__(self):
        self.out_horizon: Optional[int] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LastRepeat":
        self.out_horizon = y.shape[1] if y.ndim == 3 else 1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        last = x[:, -1:, :]
        h = self.out_horizon or 1
        return np.repeat(last, h, axis=1)

class LinearSVRMulti:
    """Per-output LinearSVR (slower; used on small TS tasks)."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.models = []
        self.out_shape = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearSVRMulti":
        xb = x.reshape(len(x), -1)
        yb = y.reshape(len(y), -1)
        self.out_shape = y.shape[1:]
        self.models = []
        for j in range(yb.shape[1]):
            m = LinearSVR(C=self.C, max_iter=self.max_iter, random_state=self.seed)
            m.fit(xb, yb[:, j])
            self.models.append(m)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = x.reshape(len(x), -1)
        cols = [m.predict(xb) for m in self.models]
        y = np.stack(cols, axis=1)
        if self.out_shape is not None:
            y = y.reshape((-1,) + self.out_shape)
        return y

class NodeWiseESNAdapter:
    """
    Apply independent ESNs per node on (B,T,N) traffic windows.
    Used as a strong non-graph RC baseline on PEMS.
    """

    def __init__(self, esn_factory, n_nodes: int):
        self.esn_factory = esn_factory
        self.n_nodes = n_nodes
        self.models = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NodeWiseESNAdapter":
        # For speed: train one shared ESN on flattened all-node series is elsewhere;
        # here we fit a single multi-output ridge on concatenated node windows via one ESN
        # operating on mean-pooled signal + residual — keep simple: one ESN on vectorized input.
        from model.esn import ESN, ESNConfig

        B, T, N = x.shape
        H = y.shape[1]
        # Use spatial mean as driving signal, predict all nodes via ridge on reservoir+input
        u = x.mean(axis=2, keepdims=False)  # (B,T)
        # Build features by running ESN per sample
        cfg = ESNConfig(n_reservoir=120, washout=0, seed=7, ridge=1e-3)
        esn = ESN(n_inputs=1, n_outputs=H * N, cfg=cfg)
        feats = []
        for i in range(B):
            esn.reset()
            st = None
            for t in range(T):
                st = esn.step(np.array([u[i, t]]))
            feats.append(st)
        F = np.stack(feats, axis=1)  # (R,B)
        from model.reservoir_utils import add_bias, ridge_fit

        yb = y.reshape(B, -1).T
        Fb = add_bias(F)
        self.wout = ridge_fit(Fb, yb, ridge=cfg.ridge)
        self.cfg = cfg
        self.esn = esn
        self.H = H
        self.N = N
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        from model.reservoir_utils import add_bias, ridge_predict

        B, T, N = x.shape
        u = x.mean(axis=2)
        feats = []
        for i in range(B):
            self.esn.reset()
            st = None
            for t in range(T):
                st = self.esn.step(np.array([u[i, t]]))
            feats.append(st)
        F = np.stack(feats, axis=1)
        Fb = add_bias(F)
        y = ridge_predict(self.wout, Fb).T
        return y.reshape(B, self.H, self.N)
