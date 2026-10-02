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

"""Deep Graph Reservoir Network with Interlayer Sparse Compression (ICS-DGRN)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .graph_ops import normalize_adjacency, scale_operator_for_esp, spectral_norm
from .reservoir_utils import (
    add_bias,
    gaussian_measurement,
    ridge_fit,
    ridge_predict,
    sparse_random_matrix,
)

@dataclass
class DGRNConfig:
    layer_dims: tuple = (32, 24)  # feature dims d_l per layer
    compression_dims: tuple = (16,)  # H output feature dims between layers
    leak: float = 0.55
    input_scale: float = 0.4
    density: float = 0.15
    ridge: float = 1e-3
    esp_target: float = 0.85  # ||W|| ||S|| target
    graph_norm: str = "rw"
    add_self_loop: bool = True
    washout: int = 0
    seed: int = 42
    use_bias: bool = True
    concatenate_layers: bool = True
    # for sequence samples (B,T,N): use last state / mean / all steps
    pool: str = "last"  # last | mean | mix
    # shared node-wise readout (recommended for spatiotemporal)
    nodewise_readout: bool = True
    # residual: predict residual over last observed frame
    residual: bool = True
    # append last input value to node features
    append_input: bool = True

class ICSDGRN:
    """
    Proposed model: Deep Graph Reservoir + interlayer sparse compression.

    Matrix-valued reservoir state X in R^{N x d} with left graph op S
    and right recurrent W, matching the ESP analysis in 2.pdf.
    """

    def __init__(
        self,
        n_nodes: int,
        n_in_feat: int,
        n_outputs: int,
        adj: np.ndarray,
        cfg: Optional[DGRNConfig] = None,
    ):
        self.cfg = cfg or DGRNConfig()
        self.n_nodes = n_nodes
        self.n_in_feat = n_in_feat
        self.n_outputs = n_outputs
        rng = np.random.default_rng(self.cfg.seed)

        s = normalize_adjacency(
            adj,
            add_self_loop=self.cfg.add_self_loop,
            mode=self.cfg.graph_norm,
        )
        self.S = s
        self.s_norm = spectral_norm(s)

        dims = list(self.cfg.layer_dims)
        self.dims = dims
        comps = list(self.cfg.compression_dims)
        if len(comps) < len(dims) - 1:
            comps = comps + [max(dims[-1] // 2, 4)] * (len(dims) - 1 - len(comps))
        self.comps = comps[: max(len(dims) - 1, 0)]

        self.wins: List[np.ndarray] = []
        self.ws: List[np.ndarray] = []
        self.phis: List[Optional[np.ndarray]] = []
        self.esp_products: List[float] = []

        prev_feat = n_in_feat
        for i, d in enumerate(dims):
            # Win: R^{prev_feat x d}  so  H @ Win  -> (N, d)
            win = rng.uniform(-self.cfg.input_scale, self.cfg.input_scale, size=(prev_feat, d))
            w = sparse_random_matrix(d, d, density=self.cfg.density, rng=rng)
            _, w, _, nw = scale_operator_for_esp(s, w, target=self.cfg.esp_target)
            self.wins.append(win.astype(np.float64))
            self.ws.append(w.astype(np.float64))
            self.esp_products.append(float(self.s_norm * nw))
            if i < len(dims) - 1:
                m = self.comps[i]
                # compress features: H(X)= X @ Phi.T , Phi (m, d)
                phi = gaussian_measurement(d, m, rng=rng)
                self.phis.append(phi)
                prev_feat = m
            else:
                self.phis.append(None)

        self.states: List[np.ndarray] = [np.zeros((n_nodes, d), dtype=np.float64) for d in dims]
        self.wout: Optional[np.ndarray] = None
        self._node_feat_dim = self._compute_node_feat_dim()
        self._horizon: Optional[int] = None

    def _compute_node_feat_dim(self) -> int:
        base = int(sum(self.dims) if self.cfg.concatenate_layers else self.dims[-1])
        if self.cfg.pool == "mix":
            base *= 2  # last + mean
        if self.cfg.append_input:
            base += 1
        return base

    @property
    def feature_dim(self) -> int:
        return self._node_feat_dim + (1 if self.cfg.use_bias else 0)

    def reset(self) -> None:
        self.states = [np.zeros((self.n_nodes, d), dtype=np.float64) for d in self.dims]

    def _compress(self, layer_idx: int, x: np.ndarray) -> np.ndarray:
        phi = self.phis[layer_idx]
        if phi is None:
            return x
        return x @ phi.T

    def step_states(self, u: np.ndarray) -> List[np.ndarray]:
        """Update reservoirs; return list of layer states (each N x d_l)."""
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            h = u.reshape(self.n_nodes, 1)
        else:
            h = u.reshape(self.n_nodes, -1)

        alpha = self.cfg.leak
        outs = []
        for i, d in enumerate(self.dims):
            sxw = self.S @ (self.states[i] @ self.ws[i])
            hin = h @ self.wins[i]
            pre = sxw + hin
            x = (1.0 - alpha) * self.states[i] + alpha * np.tanh(pre)
            self.states[i] = x
            outs.append(x)
            if i < len(self.dims) - 1:
                h = self._compress(i, x)
        return outs

    def step(self, u: np.ndarray) -> np.ndarray:
        """Flattened global feature (legacy / ESP diagnostics)."""
        outs = self.step_states(u)
        if self.cfg.concatenate_layers:
            return np.concatenate([o.reshape(-1) for o in outs])
        return outs[-1].reshape(-1)

    def _node_features_from_traj(
        self,
        traj: List[List[np.ndarray]],
        u_win: np.ndarray,
    ) -> np.ndarray:
        """
        traj: length T, each item list of layer states
        returns (N, F_node)
        """
        def _cat_layers(states: List[np.ndarray]) -> np.ndarray:
            if self.cfg.concatenate_layers:
                return np.concatenate(states, axis=1)
            return states[-1]

        last = _cat_layers(traj[-1])  # (N, dsum)
        if self.cfg.pool == "mean":
            stacked = np.stack([_cat_layers(s) for s in traj], axis=0).mean(axis=0)
            feat = stacked
        elif self.cfg.pool == "mix":
            mean = np.stack([_cat_layers(s) for s in traj], axis=0).mean(axis=0)
            feat = np.concatenate([last, mean], axis=1)
        else:
            feat = last

        if self.cfg.append_input:
            last_u = u_win[-1]
            if last_u.ndim == 2:
                last_u = last_u[:, 0]
            last_u = np.asarray(last_u, dtype=np.float64).reshape(self.n_nodes, 1)
            feat = np.concatenate([feat, last_u], axis=1)
        return feat.astype(np.float64)

    def encode_window_nodes(self, u_win: np.ndarray) -> np.ndarray:
        """u_win (T,N) or (T,N,C) -> node features (N, F)."""
        return self.encode_batch_nodes(u_win[None, ...])[0]

    def encode_batch_nodes(self, x: np.ndarray) -> np.ndarray:
        """
        Batched reservoir rollout.
        x: (B,T,N) or (B,T,N,C) -> (B,N,F)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            # (B,T,N) -> (B,T,N,1)
            x = x[..., None]
        B, T, N, C = x.shape
        assert N == self.n_nodes
        alpha = self.cfg.leak
        states = [np.zeros((B, N, d), dtype=np.float64) for d in self.dims]
        sum_cat = None
        last_cat = None

        for t in range(T):
            h = x[:, t]  # (B,N,C)
            outs = []
            for i, d in enumerate(self.dims):
                # S @ X @ W  for each batch: (N,N)@(B,N,d)->(B,N,d)
                xw = states[i] @ self.ws[i]  # (B,N,d)
                sxw = self.S @ xw  # broadcast (N,N)@(B,N,d)
                hin = h @ self.wins[i]  # (B,N,d)
                pre = sxw + hin
                new = (1.0 - alpha) * states[i] + alpha * np.tanh(pre)
                states[i] = new
                outs.append(new)
                if i < len(self.dims) - 1:
                    phi = self.phis[i]
                    h = new @ phi.T if phi is not None else new
            if self.cfg.concatenate_layers:
                cat = np.concatenate(outs, axis=-1)
            else:
                cat = outs[-1]
            if self.cfg.pool in ("mean", "mix"):
                sum_cat = cat if sum_cat is None else sum_cat + cat
            last_cat = cat

        if self.cfg.pool == "mean":
            feat = sum_cat / T
        elif self.cfg.pool == "mix":
            feat = np.concatenate([last_cat, sum_cat / T], axis=-1)
        else:
            feat = last_cat

        if self.cfg.append_input:
            last_u = x[:, -1, :, 0:1]  # (B,N,1)
            feat = np.concatenate([feat, last_u], axis=-1)
        return feat

    def collect_sequence_features(self, u_seq: np.ndarray, washout: Optional[int] = None) -> np.ndarray:
        washout = self.cfg.washout if washout is None else washout
        self.reset()
        feats = []
        for t in range(len(u_seq)):
            f = self.step(u_seq[t])
            if t >= washout:
                feats.append(f)
        return np.stack(feats, axis=1)

    def encode_window(self, u_win: np.ndarray) -> np.ndarray:
        return self.encode_window_nodes(u_win).reshape(-1)

    def encode_batch(self, x: np.ndarray) -> np.ndarray:
        feats = [self.encode_window(x[i]) for i in range(len(x))]
        return np.stack(feats, axis=1)

    def fit_windows(self, x: np.ndarray, y: np.ndarray) -> "ICSDGRN":
        """
        x: (B, T, N) or (B, T, N, C), y: (B, H, N)
        Uses shared node-wise ridge readout (parameter-efficient, ESP-faithful).
        """
        y2 = np.asarray(y, dtype=np.float64)
        if y2.ndim == 2:
            y2 = y2[:, None, :]
        B, H, N = y2.shape
        assert N == self.n_nodes
        self._horizon = H

        node_feat = self.encode_batch_nodes(x)  # (B,N,F)
        F = node_feat.shape[-1]
        xf = node_feat.reshape(B * N, F).T  # (F, B*N)

        # primary traffic channel for residual
        if x.ndim == 4:
            last = x[:, -1, :, 0]
        else:
            last = x[:, -1, :]
        if self.cfg.residual:
            target = y2 - last[:, None, :]
        else:
            target = y2
        yf = np.transpose(target, (1, 0, 2)).reshape(H, B * N)

        if self.cfg.use_bias:
            xf = add_bias(xf)
        self.wout = ridge_fit(xf, yf, ridge=self.cfg.ridge)
        self.n_outputs = H
        return self

    def predict_windows(self, x: np.ndarray, out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        node_feat = self.encode_batch_nodes(x)  # (B,N,F)
        B, N, F = node_feat.shape
        xf = node_feat.reshape(B * N, F).T
        if self.cfg.use_bias:
            xf = add_bias(xf)
        yhat = ridge_predict(self.wout, xf)  # (H, B*N)
        H = yhat.shape[0]
        yhat = yhat.reshape(H, B, N).transpose(1, 0, 2)  # (B,H,N)
        if self.cfg.residual:
            if x.ndim == 4:
                last = x[:, -1, :, 0]
            else:
                last = x[:, -1, :]
            yhat = yhat + last[:, None, :]
        if out_shape is not None:
            yhat = yhat.reshape((-1,) + out_shape)
        return yhat

    def fit_sequence(self, u_seq: np.ndarray, y_seq: np.ndarray) -> "ICSDGRN":
        wash = self.cfg.washout
        x = self.collect_sequence_features(u_seq, washout=wash)
        y = np.asarray(y_seq, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = y[wash:].T
        if self.cfg.use_bias:
            x = add_bias(x)
        self.wout = ridge_fit(x, y, ridge=self.cfg.ridge)
        return self

    def predict_sequence(self, u_seq: np.ndarray, washout: int = 0) -> np.ndarray:
        x = self.collect_sequence_features(u_seq, washout=washout)
        if self.cfg.use_bias:
            x = add_bias(x)
        return ridge_predict(self.wout, x).T

    def esp_report(self) -> dict:
        return {
            "||S||_2": self.s_norm,
            "layer_||W||*||S||": self.esp_products,
            "esp_ok": all(p < 1.0 for p in self.esp_products),
            "esp_target": self.cfg.esp_target,
            "node_feat_dim": self._node_feat_dim,
        }

# Alias matching paper naming
DeepGraphReservoir = ICSDGRN
DGRN = ICSDGRN
