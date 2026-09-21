# Copyright 2026 ZengWenquan
# https://github.com/chaosbull
# SPDX-License-Identifier: Apache-2.0

"""STGCN, T-GCN, Graph WaveNet and AGCRN, plus the shared training loop.

These are compact reimplementations. Widths are the ones used in the
comparison (mostly 32), not the official checkpoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def normalized_adj(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    a = adj.astype(np.float64)
    a = a + np.eye(a.shape[0])
    d = a.sum(axis=1)
    d_inv_sqrt = np.power(np.maximum(d, 1e-12), -0.5)
    s = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    return torch.tensor(s, dtype=torch.float32, device=device)


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.theta = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x, s):
        return self.theta(torch.matmul(s, x))


class TemporalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 3):
        super().__init__()
        # GLU: split the conv output in half
        self.conv = nn.Conv2d(c_in, c_out * 2, kernel_size=(1, kernel), padding=(0, kernel // 2))

    def forward(self, x):
        y = self.conv(x)
        p, q = y.chunk(2, dim=1)
        return p * torch.sigmoid(q)


class STConvBlock(nn.Module):
    def __init__(self, c_in: int, c_hid: int, c_out: int):
        super().__init__()
        self.t1 = TemporalConv(c_in, c_hid)
        self.g = GraphConv(c_hid, c_hid)
        self.t2 = TemporalConv(c_hid, c_out)
        self.bn = nn.BatchNorm2d(c_out)
        self.align = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, s):
        h = self.t1(x)
        b, c, n, t = h.shape
        h2 = h.permute(0, 3, 2, 1).reshape(b * t, n, c)
        h2 = F.relu(self.g(h2, s))
        h2 = h2.reshape(b, t, n, c).permute(0, 3, 2, 1)
        out = self.t2(h2)
        out = self.bn(out + self.align(x))
        return F.relu(out)


class STGCN(nn.Module):
    def __init__(self, n_nodes: int, hist: int, horizon: int, channels: int = 32):
        super().__init__()
        self.block1 = STConvBlock(1, channels, channels)
        self.block2 = STConvBlock(channels, channels, channels)
        self.head = nn.Conv2d(channels, horizon, kernel_size=(1, hist))

    def forward(self, x, s):
        # (B, T, N) -> (B, C, N, T)
        h = x.unsqueeze(1).permute(0, 1, 3, 2)
        h = self.block1(h, s)
        h = self.block2(h, s)
        return self.head(h).squeeze(-1)


class TGCNCell(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.gc_u = GraphConv(1 + hid, hid)
        self.gc_r = GraphConv(1 + hid, hid)
        self.gc_c = GraphConv(1 + hid, hid)

    def forward(self, x, h, s):
        xu = torch.cat([x, h], dim=-1)
        u = torch.sigmoid(self.gc_u(xu, s))
        r = torch.sigmoid(self.gc_r(xu, s))
        c = torch.tanh(self.gc_c(torch.cat([x, r * h], dim=-1), s))
        return u * h + (1 - u) * c


class TGCN(nn.Module):
    def __init__(self, n_nodes: int, hist: int, horizon: int, hid: int = 32):
        super().__init__()
        self.hid = hid
        self.cell = TGCNCell(hid)
        self.out = nn.Linear(hid, horizon)

    def forward(self, x, s):
        b, t, n = x.shape
        h = torch.zeros(b, n, self.hid, device=x.device)
        for i in range(t):
            h = self.cell(x[:, i, :].unsqueeze(-1), h, s)
        return self.out(h).permute(0, 2, 1)


class DilatedInception(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 2), dilation=(1, dilation))

    def forward(self, x):
        return self.conv(x)


class GWNet(nn.Module):
    def __init__(self, n_nodes: int, hist: int, horizon: int, channels: int = 32, layers: int = 4):
        super().__init__()
        self.start = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.layers = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        for i in range(layers):
            self.layers.append(DilatedInception(channels, channels, dilation=2**i))
            self.gconvs.append(GraphConv(channels, channels))
        self.end = nn.Conv2d(channels, horizon, kernel_size=(1, 1))

    def forward(self, x, s):
        h = x.unsqueeze(1).permute(0, 1, 3, 2)
        h = self.start(h)
        for dil, gc in zip(self.layers, self.gconvs):
            pad = dil.conv.dilation[1]
            h_pad = F.pad(h, (pad, 0))
            z = torch.tanh(dil(h_pad))
            b, c, n, t = z.shape
            z2 = z.permute(0, 3, 2, 1).reshape(b * t, n, c)
            z2 = F.relu(gc(z2, s)).reshape(b, t, n, c).permute(0, 3, 2, 1)
            if z2.size(-1) != h.size(-1):
                z2 = z2[..., -h.size(-1) :]
            h = h + z2
        return self.end(h[..., -1:]).squeeze(-1)


class AGCRN(nn.Module):
    def __init__(self, n_nodes: int, hist: int, horizon: int, hid: int = 32, emb: int = 8):
        super().__init__()
        self.hid = hid
        self.node_emb = nn.Parameter(torch.randn(n_nodes, emb) * 0.1)
        self.w_in = nn.Linear(1 + emb, hid)
        self.w_hh = nn.Linear(hid, hid)
        self.g = GraphConv(hid, hid)
        self.out = nn.Linear(hid, horizon)

    def forward(self, x, s):
        b, t, n = x.shape
        emb = self.node_emb.unsqueeze(0).expand(b, -1, -1)
        h = torch.zeros(b, n, self.hid, device=x.device)
        for i in range(t):
            xt = x[:, i, :].unsqueeze(-1)
            h = torch.tanh(self.w_in(torch.cat([xt, emb], dim=-1)) + self.w_hh(h))
            h = F.relu(self.g(h, s)) + h
        return self.out(h).permute(0, 2, 1)


MAX_EPOCHS = 1000


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    epochs: int = MAX_EPOCHS
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = MAX_EPOCHS
    device: str = "auto"
    seed: int = 42
    loss: str = "smooth_l1"


def _loss_fn(pred, target, kind: str):
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    return F.l1_loss(pred, target)


def train_model(
    model: nn.Module,
    adj: np.ndarray,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Optional[TrainConfig] = None,
) -> Dict:
    cfg = cfg or TrainConfig()
    cfg.epochs = min(max(int(cfg.epochs), 1), MAX_EPOCHS)
    torch.manual_seed(cfg.seed)
    device = torch.device(default_device() if cfg.device == "auto" else cfg.device)
    if device.type == "cuda":
        print(f"    device={device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"    device={device}")

    model = model.to(device)
    s = normalized_adj(adj, device)
    if hasattr(model, "set_s_norm"):
        model.set_s_norm(s)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=1e-5)

    def _loader(x, y, shuffle):
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(xt, yt), batch_size=cfg.batch_size, shuffle=shuffle)

    tr_loader = _loader(x_tr, y_tr, True)
    va_loader = _loader(x_va, y_va, False)

    best_state = None
    best_va = float("inf")
    bad = 0
    t0 = time.perf_counter()
    history = []

    for ep in range(cfg.epochs):
        model.train()
        tr_loss = tr_mae = 0.0
        nobs = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, s)
            loss = _loss_fn(pred, yb, cfg.loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if hasattr(model, "set_s_norm"):
                model.set_s_norm(s)
            bs = len(xb)
            tr_loss += loss.item() * bs
            tr_mae += F.l1_loss(pred.detach(), yb).item() * bs
            nobs += bs
        tr_loss /= max(nobs, 1)
        tr_mae /= max(nobs, 1)

        model.eval()
        va_loss = va_mae = 0.0
        nobs = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, s)
                loss = _loss_fn(pred, yb, cfg.loss)
                bs = len(xb)
                va_loss += loss.item() * bs
                va_mae += F.l1_loss(pred, yb).item() * bs
                nobs += bs
        va_loss /= max(nobs, 1)
        va_mae /= max(nobs, 1)
        sched.step()

        history.append(
            {
                "epoch": ep + 1,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "train_mae": tr_mae,
                "val_mae": va_mae,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        if va_mae + 1e-6 < best_va:
            best_va = va_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

        # 1000 epoch 时别每轮都刷，大约每 20 轮报一次
        log_every = 20 if cfg.epochs > 100 else max(1, cfg.epochs // 10)
        if (ep + 1) % log_every == 0 or ep == 0 or (ep + 1) == cfg.epochs:
            print(
                f"    ep {ep + 1}/{cfg.epochs}  "
                f"train {tr_loss:.4f}  val {va_loss:.4f}  val_mae {va_mae:.4f}"
            )

    train_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "adj_t": s,
        "train_sec": train_sec,
        "epochs_ran": len(history),
        "best_val_mae": best_va,
        "params": count_params(model),
        "history": history,
        "device": str(device),
        "loss_name": cfg.loss,
    }


@torch.no_grad()
def predict(pack: Dict, x: np.ndarray, batch_size: int = 64) -> Tuple[np.ndarray, float]:
    model: nn.Module = pack["model"]
    s = pack["adj_t"]
    device = next(model.parameters()).device
    model.eval()
    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        preds.append(model(xb, s).cpu().numpy())
    return np.concatenate(preds, axis=0), time.perf_counter() - t0


def build_model(name: str, n_nodes: int, hist: int, horizon: int) -> nn.Module:
    key = name.upper().replace("-", "").replace("_", "")
    if key == "STGCN":
        return STGCN(n_nodes, hist, horizon, channels=32)
    if key == "TGCN":
        return TGCN(n_nodes, hist, horizon, hid=32)
    if key == "GWNET":
        return GWNet(n_nodes, hist, horizon, channels=32, layers=4)
    if key == "AGCRN":
        return AGCRN(n_nodes, hist, horizon, hid=32, emb=8)
    if key in ("ICSDGRN", "DGRN"):
        from model.ics_dgrn import build_ics_dgrn

        return build_ics_dgrn(n_nodes, hist, horizon)
    raise ValueError(f"unknown model: {name}")
