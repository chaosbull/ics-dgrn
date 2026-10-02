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

"""Trainable ICS-DGRN v3 (Deep Learning) — fast ESP-constrained Deep Graph Reservoir."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _power_spectral_norm(w: torch.Tensor, n_iter: int = 8) -> torch.Tensor:
    """Approximate operator 2-norm for square or rectangular matrices."""
    if w.ndim != 2:
        raise ValueError("expected 2D matrix")
    # Start on the larger side for better conditioning
    if w.shape[0] >= w.shape[1]:
        v = w.new_empty(w.shape[1]).normal_()
        v = F.normalize(v, dim=0)
        for _ in range(n_iter):
            u = F.normalize(w @ v, dim=0)
            v = F.normalize(w.t() @ u, dim=0)
        return torch.linalg.vector_norm(w @ v).clamp_min(1e-8)
    v = w.new_empty(w.shape[0]).normal_()
    v = F.normalize(v, dim=0)
    for _ in range(n_iter):
        u = F.normalize(w.t() @ v, dim=0)
        v = F.normalize(w @ u, dim=0)
    return torch.linalg.vector_norm(w.t() @ v).clamp_min(1e-8)

def _make_sparse_w(d: int, density: float = 0.2, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    w = torch.rand(d, d, generator=g) * 2.0 - 1.0
    mask = torch.rand(d, d, generator=g) < density
    w = w * mask.float()
    return w

class GraphReservoirLayer(nn.Module):
    """ESP-scaled graph reservoir layer; W fixed by default, optionally trainable."""

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        esp_target: float = 0.9,
        leak_init: float = 0.55,
        density: float = 0.2,
        seed: int = 0,
        train_recurrent: bool = False,
    ):
        super().__init__()
        self.esp_target = float(esp_target)
        self.train_recurrent = bool(train_recurrent)
        self.win = nn.Linear(d_in, d_hid, bias=True)
        nn.init.uniform_(self.win.weight, -0.4, 0.4)
        w = _make_sparse_w(d_hid, density=density, seed=seed)
        self.register_buffer("w", w, persistent=True)
        if self.train_recurrent:
            self.w_scaled = nn.Parameter(w.clone())
        else:
            self.register_buffer("w_scaled", w.clone(), persistent=True)
        self._w_ready = False
        leak = min(max(float(leak_init), 1e-3), 1.0 - 1e-3)
        self.leak_logit = nn.Parameter(torch.logit(torch.tensor(leak)))

    def prepare_esp(self, s_norm: torch.Tensor) -> None:
        # Scale once from the fixed random skeleton `w`. For trainable W we keep
        # the ESP-compliant initialization and then allow gradient updates.
        if self._w_ready and self.train_recurrent:
            return
        with torch.no_grad():
            sigma = _power_spectral_norm(self.w, n_iter=10)
            scale = self.esp_target / (s_norm.clamp_min(1e-8) * sigma)
            self.w_scaled.data.copy_(self.w * scale)
            self._w_ready = True

    def forward(self, h: torch.Tensor, state: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.leak_logit)
        xw = torch.matmul(state, self.w_scaled)
        sxw = torch.matmul(s, xw)
        pre = sxw + self.win(h)
        return (1.0 - alpha) * state + alpha * torch.tanh(pre)

class ICSCompress(nn.Module):
    """Interlayer Gaussian compression H(X)=X Phi^T with optional sparsity mask."""

    def __init__(self, d_in: int, d_out: int, density: float = 1.0, seed: int = 0):
        super().__init__()
        self.density = float(density)
        g = torch.Generator()
        g.manual_seed(seed)
        phi = torch.randn(d_out, d_in, generator=g) * (1.0 / max(d_out, 1) ** 0.5)
        if self.density < 1.0 - 1e-12:
            mask = (torch.rand(d_out, d_in, generator=g) < self.density).float()
            # keep at least one nonzero per row/col when possible
            if mask.sum() < 1:
                mask[0, 0] = 1.0
            phi = phi * mask
            self.register_buffer("mask", mask, persistent=True)
        else:
            self.register_buffer("mask", torch.ones(d_out, d_in), persistent=True)
        self.phi = nn.Parameter(phi)

    def effective_phi(self) -> torch.Tensor:
        return self.phi * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.effective_phi().t())

    def spectral_norm(self, n_iter: int = 10) -> float:
        with torch.no_grad():
            return float(_power_spectral_norm(self.effective_phi(), n_iter=n_iter).item())

    def nonzero_ratio(self) -> float:
        with torch.no_grad():
            m = self.mask
            return float((m > 0).float().mean().item())

class ICSDGRNTrainable(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        hist: int,
        horizon: int,
        layer_dims: Tuple[int, ...] = (48, 32),
        compression_dims: Tuple[int, ...] = (24,),
        esp_target: float = 0.9,
        in_feats: int = 3,
        dropout: float = 0.05,
        emb_dim: int = 8,
        density: float = 0.2,
        phi_density: float = 1.0,
        train_recurrent: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.hist = hist
        self.horizon = horizon
        self.layer_dims = list(layer_dims)
        self.esp_target = esp_target
        self.phi_density = float(phi_density)
        self.train_recurrent = bool(train_recurrent)
        self.seed = int(seed)

        comps = list(compression_dims)
        while len(comps) < len(layer_dims) - 1:
            comps.append(max(layer_dims[len(comps) + 1] // 2, 8))
        self.comps = comps[: max(len(layer_dims) - 1, 0)]

        self.node_emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.05)
        self.time_emb = nn.Parameter(torch.randn(hist, emb_dim) * 0.05)
        self.in_proj = nn.Linear(in_feats + emb_dim * 2, layer_dims[0])

        self.layers = nn.ModuleList()
        self.compress = nn.ModuleList()
        self.layers.append(
            GraphReservoirLayer(
                layer_dims[0],
                layer_dims[0],
                esp_target=esp_target,
                leak_init=0.55,
                density=density,
                seed=seed,
                train_recurrent=train_recurrent,
            )
        )
        for i in range(len(layer_dims) - 1):
            self.compress.append(
                ICSCompress(
                    layer_dims[i],
                    self.comps[i],
                    density=phi_density,
                    seed=seed + 100 + i,
                )
            )
            self.layers.append(
                GraphReservoirLayer(
                    self.comps[i],
                    layer_dims[i + 1],
                    esp_target=esp_target,
                    leak_init=0.55,
                    density=density,
                    seed=seed + i + 1,
                    train_recurrent=train_recurrent,
                )
            )

        feat_dim = sum(self.layer_dims) * 2 + 1 + emb_dim
        self.drop = nn.Dropout(dropout)
        mix_r = 32
        self.graph_down = nn.Linear(feat_dim, mix_r, bias=False)
        self.graph_up = nn.Linear(mix_r, feat_dim, bias=False)
        self.readout = nn.Sequential(
            nn.Linear(feat_dim, 80),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(80, horizon),
        )
        self.register_buffer("s_norm", torch.tensor(1.0), persistent=True)

    def param_stats(self) -> Dict[str, int]:
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        total = int(sum(p.numel() for p in self.parameters()) + sum(b.numel() for b in self.buffers() if b.dtype.is_floating_point))
        # count stored float tensors more carefully
        stored = 0
        for p in self.parameters():
            stored += p.numel()
        for name, b in self.named_buffers():
            if b is not None and b.is_floating_point() and name not in ("s_norm",):
                stored += b.numel()
        return {
            "trainable_params": trainable,
            "total_params": stored,
            "named_parameters": trainable,
        }

    def set_s_norm(self, s: torch.Tensor) -> None:
        with torch.no_grad():
            n = s.shape[0]
            v = s.new_empty(n).normal_()
            v = F.normalize(v, dim=0)
            for _ in range(12):
                v = F.normalize(s @ v, dim=0)
                v = F.normalize(s.t() @ v, dim=0)
            self.s_norm = torch.linalg.vector_norm(s @ v).detach().clamp_min(1e-8)
        for layer in self.layers:
            layer.prepare_esp(self.s_norm)

    def _build_input(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        b, t, n = x.shape
        xt = x.reshape(b * t, n)
        su = torch.matmul(xt, s.t()).reshape(b, t, n)
        s2u = torch.matmul(su.reshape(b * t, n), s.t()).reshape(b, t, n)
        return torch.stack([x, su, s2u], dim=-1)

    def _encode_u(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        b, t, n = x.shape
        u_raw = self._build_input(x, s)
        emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)
        te = self.time_emb[:t].unsqueeze(0).unsqueeze(2).expand(b, t, n, -1)
        return self.in_proj(torch.cat([u_raw, emb, te], dim=-1))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if float(self.s_norm.item()) <= 0 or torch.isnan(self.s_norm) or not self.layers[0]._w_ready:
            self.set_s_norm(s)

        b, t, n = x.shape
        u = self._encode_u(x, s)

        states: List[torch.Tensor] = [
            torch.zeros(b, n, d, device=x.device, dtype=x.dtype) for d in self.layer_dims
        ]
        sum_states: List[torch.Tensor] = [
            torch.zeros(b, n, d, device=x.device, dtype=x.dtype) for d in self.layer_dims
        ]

        for step in range(t):
            h = u[:, step]
            for i, layer in enumerate(self.layers):
                states[i] = layer(h, states[i], s)
                sum_states[i] = sum_states[i] + states[i]
                if i < len(self.compress):
                    h = self.compress[i](states[i])

        feats = []
        inv_t = 1.0 / float(t)
        for i in range(len(self.layer_dims)):
            feats.append(states[i])
            feats.append(sum_states[i] * inv_t)
        feats.append(x[:, -1].unsqueeze(-1))
        feats.append(self.node_emb.unsqueeze(0).expand(b, -1, -1))
        feat = torch.cat(feats, dim=-1)
        feat = feat + self.graph_up(torch.matmul(s, self.graph_down(feat)))
        feat = self.drop(feat)
        y_res = self.readout(feat)
        return y_res.transpose(1, 2) + x[:, -1:].contiguous()

    @torch.no_grad()
    def state_difference_decay(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        seed_a: int = 0,
        seed_b: int = 1,
        init_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Drive two copies with identical input and different initial states.
        Returns per-layer Frobenius norms D_l(t) and normalized hat{D}_l(t).
        """
        if not self.layers[0]._w_ready:
            self.set_s_norm(s)

        was_training = self.training
        self.eval()
        b, t, n = x.shape
        assert b == 1, "state_difference_decay expects batch size 1"
        u = self._encode_u(x, s)

        g_a = torch.Generator()
        g_b = torch.Generator()
        g_a.manual_seed(seed_a)
        g_b.manual_seed(seed_b)

        states_a = [
            torch.randn(1, n, d, dtype=x.dtype, generator=g_a).to(x.device) * init_scale
            for d in self.layer_dims
        ]
        states_b = [
            torch.randn(1, n, d, dtype=x.dtype, generator=g_b).to(x.device) * init_scale
            for d in self.layer_dims
        ]
        d0 = torch.stack(
            [torch.linalg.vector_norm(sa - sb) for sa, sb in zip(states_a, states_b)]
        ).clamp_min(1e-12)

        diffs = [[] for _ in self.layer_dims]
        for step in range(t):
            h_a = u[:, step]
            h_b = u[:, step]
            for i, layer in enumerate(self.layers):
                states_a[i] = layer(h_a, states_a[i], s)
                states_b[i] = layer(h_b, states_b[i], s)
                diffs[i].append(torch.linalg.vector_norm(states_a[i] - states_b[i]))
                if i < len(self.compress):
                    h_a = self.compress[i](states_a[i])
                    h_b = self.compress[i](states_b[i])

        D = torch.stack([torch.stack(row) for row in diffs], dim=0)  # (L, T)
        D_hat = D / d0.unsqueeze(1)
        if was_training:
            self.train()
        return {"D": D, "D_hat": D_hat, "D0": d0}

    def esp_report(self, s: torch.Tensor) -> dict:
        self.set_s_norm(s)
        products = []
        # For tanh, Lipschitz constant L_f = 1 => kappa_l = ||S||_2 ||W_l||_2
        kappas = []
        for layer in self.layers:
            with torch.no_grad():
                nw = _power_spectral_norm(layer.w_scaled, n_iter=10).item()
            prod = float(nw * float(self.s_norm.item()))
            products.append(prod)
            kappas.append(prod)  # L_f = 1
        phi_norms = [c.spectral_norm() for c in self.compress]
        return {
            "||S||_2": float(self.s_norm.item()),
            "layer_||W||*||S||": products,
            "kappa_l": kappas,
            "max_kappa": float(max(kappas) if kappas else 0.0),
            "esp_ok": all(p < 1.0 + 1e-5 for p in products),
            "esp_target": self.esp_target,
            "phi_spectral_norms": phi_norms,
            "phi_density": self.phi_density,
            "compression_dims": list(self.comps),
            "layer_dims": list(self.layer_dims),
            "train_recurrent": self.train_recurrent,
            "mode": "trainable_dl_v3_trainW" if self.train_recurrent else "trainable_dl_v3_fixedW",
        }

def build_ics_dgrn_trainable(
    n_nodes: int,
    hist: int,
    horizon: int,
    *,
    layer_dims: Tuple[int, ...] = (56, 40),
    compression_dims: Optional[Tuple[int, ...]] = None,
    compression_ratio: Optional[float] = None,
    esp_target: float = 0.9,
    density: float = 0.2,
    phi_density: float = 1.0,
    train_recurrent: bool = False,
    seed: int = 42,
) -> ICSDGRNTrainable:
    """Factory with optional compression-ratio / sparsity / trainable-W overrides."""
    if compression_dims is None:
        if compression_ratio is not None:
            d0 = int(layer_dims[0])
            cdim = max(int(round(d0 * float(compression_ratio))), 1)
            compression_dims = (cdim,)
        else:
            compression_dims = (28,)
    return ICSDGRNTrainable(
        n_nodes=n_nodes,
        hist=hist,
        horizon=horizon,
        layer_dims=layer_dims,
        compression_dims=compression_dims,
        esp_target=esp_target,
        in_feats=3,
        dropout=0.05,
        emb_dim=8,
        density=density,
        phi_density=phi_density,
        train_recurrent=train_recurrent,
        seed=seed,
    )
