# Copyright 2026 ZengWenquan
# https://github.com/chaosbull
# SPDX-License-Identifier: Apache-2.0

"""ICS-DGRN.

Stacked graph reservoirs. Between layers the hidden state is shortened
by a learned map. Recurrent weights are rescaled on every step so that
||W||_2 * ||S||_2 stays under esp_target.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _scale_w(w: torch.Tensor, s_norm: torch.Tensor, esp_target: float) -> torch.Tensor:
    # keep the product of the two spectral norms under the ESP bound
    sigma = torch.linalg.matrix_norm(w, ord=2).clamp_min(1e-8)
    scale = esp_target / (s_norm.clamp_min(1e-8) * sigma)
    return w * scale


class GraphReservoirLayer(nn.Module):
    def __init__(self, d_in: int, d_hid: int, esp_target: float = 0.95, leak_init: float = 0.6):
        super().__init__()
        self.esp_target = esp_target
        self.win = nn.Linear(d_in, d_hid, bias=True)
        self.w_raw = nn.Parameter(torch.empty(d_hid, d_hid))
        nn.init.orthogonal_(self.w_raw)
        self.leak_logit = nn.Parameter(torch.logit(torch.tensor(float(leak_init))))
        self.norm = nn.LayerNorm(d_hid)

    def forward(self, h, state, s, s_norm):
        # x <- (1-a) x + a tanh(S x W + Win h)
        alpha = torch.sigmoid(self.leak_logit)
        w = _scale_w(self.w_raw, s_norm, self.esp_target)
        xw = torch.matmul(state, w)
        sxw = torch.einsum("ij,bjd->bid", s, xw)
        pre = sxw + self.win(h)
        new = (1.0 - alpha) * state + alpha * torch.tanh(pre)
        return self.norm(new)


class ICSCompress(nn.Module):
    """Shrink the state before the next reservoir layer."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(d_out, d_in) / max(d_out, 1) ** 0.5)
        self.proj = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x):
        return self.proj(torch.matmul(x, self.phi.t()))


class TemporalMixer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.gate = nn.Conv1d(d, d, kernel_size=3, padding=1)

    def forward(self, traj):
        # (B, N, T, D) -> (B, N, D)
        b, n, t, d = traj.shape
        x = traj.reshape(b * n, t, d).transpose(1, 2)
        y = torch.tanh(self.conv(x)) * torch.sigmoid(self.gate(x))
        return y.mean(dim=-1).reshape(b, n, d)


class ICSDGRN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        hist: int,
        horizon: int,
        layer_dims: Tuple[int, ...] = (64, 48, 32),
        compression_dims: Tuple[int, ...] = (40, 24),
        esp_target: float = 0.95,
        in_feats: int = 3,
        dropout: float = 0.1,
        emb_dim: int = 16,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.hist = hist
        self.horizon = horizon
        self.layer_dims = list(layer_dims)
        self.esp_target = esp_target

        comps = list(compression_dims)
        while len(comps) < len(layer_dims) - 1:
            comps.append(max(layer_dims[len(comps) + 1] // 2, 8))
        self.comps = comps[: len(layer_dims) - 1]

        self.node_emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.in_proj = nn.Linear(in_feats + emb_dim, layer_dims[0])

        self.layers = nn.ModuleList()
        self.compress = nn.ModuleList()
        self.layers.append(GraphReservoirLayer(layer_dims[0], layer_dims[0], esp_target=esp_target))
        for i in range(len(layer_dims) - 1):
            self.compress.append(ICSCompress(layer_dims[i], self.comps[i]))
            self.layers.append(GraphReservoirLayer(self.comps[i], layer_dims[i + 1], esp_target=esp_target))

        self.mixers = nn.ModuleList([TemporalMixer(d) for d in self.layer_dims])

        feat_dim = sum(self.layer_dims) * 2 + 1 + emb_dim
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, horizon),
        )
        self.register_buffer("s_norm", torch.tensor(1.0), persistent=True)

    def set_s_norm(self, s: torch.Tensor) -> None:
        with torch.no_grad():
            self.s_norm = torch.linalg.matrix_norm(s, ord=2).detach()

    def _build_input(self, x, s):
        # raw series, one-hop, two-hop
        su = torch.einsum("ij,btj->bti", s, x)
        s2u = torch.einsum("ij,btj->bti", s, su)
        return torch.stack([x, su, s2u], dim=-1)

    def forward(self, x, s):
        if self.s_norm.item() <= 0 or torch.isnan(self.s_norm):
            self.set_s_norm(s)

        b, t, n = x.shape
        u_raw = self._build_input(x, s)
        emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)
        u = self.in_proj(torch.cat([u_raw, emb], dim=-1))

        states = [torch.zeros(b, n, d, device=x.device, dtype=x.dtype) for d in self.layer_dims]
        trajs: List[List[torch.Tensor]] = [[] for _ in self.layer_dims]

        for step in range(t):
            h = u[:, step]
            for i, layer in enumerate(self.layers):
                states[i] = layer(h, states[i], s, self.s_norm)
                trajs[i].append(states[i])
                if i < len(self.compress):
                    h = self.compress[i](states[i])

        feats = []
        for i in range(len(self.layer_dims)):
            traj = torch.stack(trajs[i], dim=2)
            feats.append(states[i])
            feats.append(self.mixers[i](traj))
        feats.append(x[:, -1].unsqueeze(-1))
        feats.append(self.node_emb.unsqueeze(0).expand(b, -1, -1))
        feat = self.drop(torch.cat(feats, dim=-1))
        y_res = self.readout(feat)
        # last observation as a residual, horizon is only 12 steps
        return y_res.transpose(1, 2) + x[:, -1:].contiguous()

    def esp_report(self, s: torch.Tensor) -> dict:
        self.set_s_norm(s)
        products = []
        for layer in self.layers:
            w = _scale_w(layer.w_raw, self.s_norm, layer.esp_target)
            nw = torch.linalg.matrix_norm(w, ord=2).item()
            products.append(float(nw * self.s_norm.item()))
        return {
            "||S||_2": float(self.s_norm.item()),
            "layer_||W||*||S||": products,
            "esp_ok": all(p < 1.0 + 1e-5 for p in products),
            "esp_target": self.esp_target,
        }


def build_ics_dgrn(n_nodes: int, hist: int, horizon: int) -> ICSDGRN:
    return ICSDGRN(
        n_nodes=n_nodes,
        hist=hist,
        horizon=horizon,
        layer_dims=(64, 48, 32),
        compression_dims=(40, 24),
        esp_target=0.95,
        in_feats=3,
        dropout=0.1,
        emb_dim=16,
    )
