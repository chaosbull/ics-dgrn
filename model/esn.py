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

"""Classic leaky Echo State Network (reservoir computing baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .reservoir_utils import (
    add_bias,
    ridge_fit,
    ridge_predict,
    scale_spectral_radius,
    sparse_random_matrix,
)

@dataclass
class ESNConfig:
    n_reservoir: int = 100
    spectral_radius: float = 0.9
    leak: float = 0.5
    input_scale: float = 0.5
    density: float = 0.1
    ridge: float = 1e-4
    washout: int = 50
    seed: int = 42
    use_bias: bool = True

class ESN:
    """Single-layer leaky ESN with ridge readout."""

    def __init__(self, n_inputs: int, n_outputs: int, cfg: Optional[ESNConfig] = None):
        self.cfg = cfg or ESNConfig()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        rng = np.random.default_rng(self.cfg.seed)
        self.win = rng.uniform(
            -self.cfg.input_scale,
            self.cfg.input_scale,
            size=(self.cfg.n_reservoir, n_inputs),
        )
        w = sparse_random_matrix(
            self.cfg.n_reservoir,
            self.cfg.n_reservoir,
            density=self.cfg.density,
            rng=rng,
        )
        self.w = scale_spectral_radius(w, self.cfg.spectral_radius)
        self.wout: Optional[np.ndarray] = None
        self._state = np.zeros(self.cfg.n_reservoir, dtype=np.float64)

    def reset(self) -> None:
        self._state = np.zeros(self.cfg.n_reservoir, dtype=np.float64)

    def step(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        pre = self.win @ u + self.w @ self._state
        x = (1.0 - self.cfg.leak) * self._state + self.cfg.leak * np.tanh(pre)
        self._state = x
        return x

    def collect_states(self, u_seq: np.ndarray, washout: Optional[int] = None) -> np.ndarray:
        """
        u_seq: (T, n_inputs)
        returns states (n_reservoir, T_eff) after washout
        """
        washout = self.cfg.washout if washout is None else washout
        self.reset()
        states = []
        for t in range(len(u_seq)):
            x = self.step(u_seq[t])
            if t >= washout:
                states.append(x.copy())
        if not states:
            raise ValueError("Sequence shorter than washout")
        return np.stack(states, axis=1)

    def fit(self, u_seq: np.ndarray, y_seq: np.ndarray) -> "ESN":
        """
        u_seq: (T, n_in), y_seq: (T, n_out) aligned with full sequence;
        washout applied to both.
        """
        wash = self.cfg.washout
        x = self.collect_states(u_seq, washout=wash)
        y = y_seq[wash:].T  # (n_out, T_eff)
        if self.cfg.use_bias:
            x = add_bias(x)
        self.wout = ridge_fit(x, y, ridge=self.cfg.ridge)
        return self

    def predict(self, u_seq: np.ndarray, washout: Optional[int] = None) -> np.ndarray:
        wash = 0 if washout is None else washout
        x = self.collect_states(u_seq, washout=wash)
        if self.cfg.use_bias:
            x = add_bias(x)
        y = ridge_predict(self.wout, x).T  # (T, n_out)
        return y

    def fit_xy(self, x_feat: np.ndarray, y: np.ndarray) -> "ESN":
        """Direct ridge on provided features (F, N) / (O, N)."""
        if self.cfg.use_bias:
            x_feat = add_bias(x_feat)
        self.wout = ridge_fit(x_feat, y, ridge=self.cfg.ridge)
        return self
