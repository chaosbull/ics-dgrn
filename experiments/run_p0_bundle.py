"""
Priority-1 support experiments for ICS-DGRN.

Protocol (same for every trainable model):
  hist = 12, horizon = 12, SmoothL1, Adam + cosine, early stopping on val MAE.
  Hyperparameters of ICS-DGRN are chosen on PEMS08 validation only.
  Recurrent W and interlayer Phi stay fixed unless an ablation says otherwise.

Outputs:
  result/p0_support/   csv, npz summaries, figures
  tex/figure/p0/       paper-ready png/pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.trainable_st import build_model, build_normalized_adj, count_params, predict_st_model
from data.pems_loader import load_pems
from model.dgrn import DGRNConfig, ICSDGRN
from model.dgrn_trainable import build_ics_dgrn_trainable
from utils.metrics import evaluate_all, mae, rmse

RESULT = os.path.join(ROOT, "result", "p0_support")
FIG = os.path.join(ROOT, "tex", "figure", "p0")

DATASETS = ("PEMS08", "PEMS04", "PEMS03")
SEEDS = (7, 21, 42)
MAIN_MODELS = ("ICS-DGRN", "AGCRN", "DCRNN", "STGCN", "GWNet", "TGCN", "DeepESN")

COLORS = {
    "ICS-DGRN": "#0F6E56",
    "AGCRN": "#C2410C",
    "DCRNN": "#6D28D9",
    "STGCN": "#1D4ED8",
    "GWNet": "#BE123C",
    "TGCN": "#0F766E",
    "DeepESN": "#64748B",
    "GESN": "#94A3B8",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _save(fig, path_no_ext: str) -> None:
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".png", bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def degree_preserving_rewire(adj: np.ndarray, seed: int) -> np.ndarray:
    """Undirected rewiring that keeps the edge count (configuration-style pairing)."""
    a = (np.asarray(adj) > 0).astype(np.int32)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0)
    n = a.shape[0]
    deg = a.sum(axis=1).astype(int)
    stubs = np.repeat(np.arange(n), deg)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        rng.shuffle(stubs)
        if len(stubs) % 2 == 1:
            stubs = stubs[:-1]
        b = np.zeros((n, n), dtype=np.float64)
        pairs = stubs.reshape(-1, 2)
        for i, j in pairs:
            if i == j:
                continue
            b[i, j] = 1.0
            b[j, i] = 1.0
        if b.sum() > 0:
            return b
    iu = np.triu_indices(n, 1)
    m = int(a[iu].sum())
    pick = rng.choice(len(iu[0]), size=min(m, len(iu[0])), replace=False)
    b = np.zeros((n, n), dtype=np.float64)
    b[iu[0][pick], iu[1][pick]] = 1.0
    return b + b.T


def load_data(name: str, seed: int, max_train: int, max_val: int, max_test: int) -> dict:
    return load_pems(name, max_train=max_train, max_val=max_val, max_test=max_test, seed=seed)


def train_fast(
    model: nn.Module,
    adj: np.ndarray,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    *,
    epochs: int,
    seed: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    s = build_normalized_adj(adj, device)
    if hasattr(model, "set_s_norm"):
        model.set_s_norm(s)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)

    def _t(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    xt, yt = _t(x_tr), _t(y_tr)
    xv, yv = _t(x_va), _t(y_va)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    best_state = None
    best_va = float("inf")
    bad = 0
    history = []
    n = xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tr_loss = tr_mae = 0.0
        seen = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = xt[idx], yt[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb, s)
            loss = torch.nn.functional.smooth_l1_loss(pred, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = xb.shape[0]
            tr_loss += float(loss.detach()) * bs
            tr_mae += float(torch.nn.functional.l1_loss(pred.detach(), yb)) * bs
            seen += bs
        model.eval()
        with torch.no_grad():
            va_mae = 0.0
            seen_v = 0
            for i in range(0, xv.shape[0], batch_size):
                pred = model(xv[i : i + batch_size], s)
                bs = pred.shape[0]
                va_mae += float(torch.nn.functional.l1_loss(pred, yv[i : i + batch_size])) * bs
                seen_v += bs
            va_mae /= max(seen_v, 1)
        sched.step()
        history.append({"epoch": ep + 1, "val_mae": va_mae, "train_mae": tr_mae / max(seen, 1)})
        if va_mae + 1e-6 < best_va:
            best_va = va_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if (ep + 1) % 8 == 0 or ep == 0:
            print(f"      ep {ep+1}/{epochs} val_mae={va_mae:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
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
        "peak_mb": peak,
        "device": str(device),
    }


def _sync_predict(pack, x, batch_size=128):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    pred, sec = predict_st_model(pack, x, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return pred, sec


def horizon_metrics(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    hmae = [mae(yt[:, h], yp[:, h]) for h in range(yt.shape[1])]
    hrmse = [rmse(yt[:, h], yp[:, h]) for h in range(yt.shape[1])]
    g = (hmae[-1] - hmae[0]) / max(hmae[0], 1e-8)
    out = {f"MAE_h{i+1}": hmae[i] for i in range(len(hmae))}
    out.update({f"RMSE_h{i+1}": hrmse[i] for i in range(len(hrmse))})
    out["G_MAE"] = g
    # pre-specified groups
    out["MAE_short"] = float(np.mean(hmae[0:4]))
    out["MAE_mid"] = float(np.mean(hmae[4:8]))
    out["MAE_long"] = float(np.mean(hmae[8:12]))
    return out


def score_row(yt, yp, extra: dict) -> dict:
    acc = evaluate_all(yt, yp)
    row = {**acc, **horizon_metrics(yt, yp), **extra}
    return row


def fit_ridge_reservoir(data: dict, *, graph: bool, seed: int, layer_dims: Tuple[int, ...], comp: Tuple[int, ...]) -> dict:
    adj = data["adj"] if graph else np.eye(data["n_nodes"], dtype=np.float64)
    cfg = DGRNConfig(
        layer_dims=layer_dims,
        compression_dims=comp,
        leak=0.55,
        input_scale=0.4,
        density=0.15,
        ridge=1e-3,
        esp_target=0.85,
        seed=seed,
        pool="mix",
        residual=True,
        append_input=True,
        nodewise_readout=True,
    )
    t0 = time.perf_counter()
    model = ICSDGRN(data["n_nodes"], 1, data["horizon"], adj, cfg)
    init_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    # fit on train+ not val; val used only as a reported number
    model.fit_windows(data["x_train"], data["y_train"])
    fit_sec = time.perf_counter() - t1
    t2 = time.perf_counter()
    yp = model.predict_windows(data["x_test"])
    infer = time.perf_counter() - t2
    yt = data["denorm"](data["y_test"])
    ypd = data["denorm"](yp)
    # parameter accounting: fixed reservoir + ridge readout
    state_dim = int(np.sum(layer_dims))
    feat = state_dim * (2 if cfg.pool == "mix" else 1) + 1
    readout = (feat + 1) * data["horizon"]
    fixed = 0
    for w in model.ws:
        fixed += int(w.size)
    for w in model.wins:
        fixed += int(w.size)
    for p in model.phis:
        if p is not None:
            fixed += int(np.asarray(p).size)
    row = score_row(
        yt,
        ypd,
        {
            "TrainTimeSec": init_sec + fit_sec,
            "InferTimeSec": infer,
            "TrainableParams": float(readout),
            "TotalParams": float(fixed + readout),
            "FixedParams": float(fixed),
            "Epochs": 1.0,
            "best_val_mae": np.nan,
            "PeakGPUMemMB": 0.0,
            "seed": seed,
            "model": "GESN" if graph else "DeepESN",
        },
    )
    err = np.mean(np.abs(yt - ypd), axis=(1, 2))
    return {"row": row, "err": err, "hmae": [row[f"MAE_h{i+1}"] for i in range(12)]}


def build_variant(name: str, data: dict, seed: int, ics_cfg: dict, variant: str = "full"):
    n, h, hist = data["n_nodes"], data["horizon"], data["hist_len"]
    if name != "ICS-DGRN":
        return build_model(name, n, hist, h), data["adj"]
    dims = tuple(ics_cfg["layer_dims"])
    comp = tuple(ics_cfg["compression_dims"])
    train_phi = bool(ics_cfg.get("train_phi", False))
    train_w = False
    adj = data["adj"]
    if variant == "no_graph":
        adj = np.eye(n, dtype=np.float64)
    elif variant == "random_graph":
        adj = degree_preserving_rewire(data["adj"], seed=seed + 17)
    elif variant == "L1":
        dims = (int(sum(dims)),)
        comp = tuple()
    elif variant == "L1_matched":
        dims = (int(sum(dims)),)
        comp = tuple()
    elif variant == "no_compress":
        if len(dims) >= 2:
            comp = tuple(int(dims[i]) for i in range(len(dims) - 1))
    elif variant == "trainable_w":
        train_w = True
    elif variant == "deep3":
        dims = (48, 36, 24)
        comp = (24, 16)
    model = build_ics_dgrn_trainable(
        n,
        hist,
        h,
        layer_dims=dims,
        compression_dims=comp if comp else None,
        esp_target=float(ics_cfg.get("esp_target", 0.9)),
        phi_density=float(ics_cfg.get("phi_density", 1.0)),
        train_recurrent=train_w,
        train_phi=train_phi,
        seed=seed,
    )
    if variant == "L1":
        # single layer: factory may still expect compression; layer_dims length 1 is fine
        pass
    return model, adj


def run_one(name, data, seed, epochs, batch_size, lr, patience, ics_cfg, variant="full") -> dict:
    model, adj = build_variant(name, data, seed, ics_cfg, variant=variant)
    pack = train_fast(
        model,
        adj,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    yp, infer = _sync_predict(pack, data["x_test"])
    yt = data["denorm"](data["y_test"])
    ypd = data["denorm"](yp)
    pstats = {}
    if hasattr(pack["model"], "param_stats"):
        pstats = pack["model"].param_stats()
    total = float(pstats.get("total_params", pack["params"]))
    trainable = float(pstats.get("trainable_params", pack["params"]))
    row = score_row(
        yt,
        ypd,
        {
            "TrainTimeSec": pack["train_sec"],
            "InferTimeSec": infer,
            "TrainableParams": trainable,
            "TotalParams": total,
            "FixedParams": max(total - trainable, 0.0),
            "Epochs": float(pack["epochs_ran"]),
            "best_val_mae": pack["best_val_mae"],
            "PeakGPUMemMB": pack["peak_mb"],
            "seed": seed,
            "model": name if variant == "full" else f"ICS-DGRN/{variant}",
            "variant": variant,
        },
    )
    if hasattr(pack["model"], "esp_report"):
        try:
            esp = pack["model"].esp_report(pack["adj_t"])
            row["esp_ok"] = bool(esp.get("esp_ok", False))
            row["max_kappa"] = float(esp.get("max_kappa", np.nan))
        except Exception:
            row["esp_ok"] = False
            row["max_kappa"] = np.nan
    err = np.mean(np.abs(yt - ypd), axis=(1, 2))
    return {"row": row, "err": err, "pack": pack, "yt": yt, "yp": ypd}


def append_csv(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def already(path: str, **kw) -> bool:
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        if k not in df.columns:
            return False
        col = df[k]
        if pd.api.types.is_numeric_dtype(col):
            m &= np.isclose(col.to_numpy(dtype=float), float(v), equal_nan=False)
        else:
            m &= col.astype(str) == str(v)
    return bool(m.any())


def sweep(args) -> dict:
    path = os.path.join(RESULT, "ics_config.json")
    if os.path.exists(path) and not args.force_sweep:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    data = load_data("PEMS08", 42, args.max_train, args.max_val, args.max_test)
    cands = [
        {"name": "fixed_56", "layer_dims": [56, 40], "compression_dims": [28], "train_phi": False, "esp_target": 0.9},
        {"name": "phi_56", "layer_dims": [56, 40], "compression_dims": [28], "train_phi": True, "esp_target": 0.9},
        {"name": "fixed_72", "layer_dims": [72, 48], "compression_dims": [36], "train_phi": False, "esp_target": 0.9},
        {"name": "fixed_48x3", "layer_dims": [48, 36, 24], "compression_dims": [24, 16], "train_phi": False, "esp_target": 0.9},
    ]
    rows = []
    best = None
    for c in cands:
        print(f"  sweep {c['name']}", flush=True)
        model = build_ics_dgrn_trainable(
            data["n_nodes"],
            data["hist_len"],
            data["horizon"],
            layer_dims=tuple(c["layer_dims"]),
            compression_dims=tuple(c["compression_dims"]),
            esp_target=c["esp_target"],
            train_phi=c["train_phi"],
            train_recurrent=False,
            seed=42,
        )
        pack = train_fast(
            model,
            data["adj"],
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            epochs=args.sweep_epochs,
            seed=42,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=max(6, args.sweep_epochs // 3),
        )
        rec = {**c, "val_mae": pack["best_val_mae"], "epochs": pack["epochs_ran"], "sec": pack["train_sec"], "params": pack["params"]}
        rows.append(rec)
        print(f"    val_mae={pack['best_val_mae']:.4f} params={pack['params']}", flush=True)
        if best is None or rec["val_mae"] < best["val_mae"]:
            best = rec
    os.makedirs(RESULT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT, "sweep_val.csv"), index=False)
    # Paper model prefers fixed Phi. Keep a trainable-phi candidate only when it
    # improves validation MAE by more than 1.5% relative.
    fixed = [r for r in rows if not r["train_phi"]]
    chosen = min(fixed, key=lambda r: r["val_mae"])
    phi = [r for r in rows if r["train_phi"]]
    if phi and phi[0]["val_mae"] < chosen["val_mae"] * 0.985:
        chosen = phi[0]
    cfg = {
        "name": chosen["name"],
        "layer_dims": chosen["layer_dims"],
        "compression_dims": chosen["compression_dims"],
        "train_phi": chosen["train_phi"],
        "esp_target": chosen["esp_target"],
        "val_mae": chosen["val_mae"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("chosen", cfg, flush=True)
    return cfg


def stage_main(args, ics_cfg):
    csv_path = os.path.join(RESULT, "main_runs.csv")
    for ds in args.datasets:
        for seed in args.seeds:
            data = load_data(ds, seed, args.max_train, args.max_val, args.max_test)
            for name in args.models:
                if already(csv_path, dataset=ds, seed=seed, model=name):
                    print(f"skip {ds} {name} seed {seed}", flush=True)
                    continue
                print(f"MAIN {ds} {name} seed {seed}", flush=True)
                if name == "DeepESN":
                    dims = tuple(ics_cfg["layer_dims"])
                    comp = tuple(ics_cfg["compression_dims"])
                    out = fit_ridge_reservoir(data, graph=False, seed=seed, layer_dims=dims, comp=comp)
                else:
                    out = run_one(name, data, seed, args.epochs, args.batch_size, args.lr, args.patience, ics_cfg)
                row = out["row"]
                row["dataset"] = ds
                row["model"] = name
                append_csv(csv_path, row)
                np.save(os.path.join(RESULT, f"err_{ds}_{name}_s{seed}.npy"), out["err"])
                print(
                    f"  MAE={row['MAE']:.3f} RMSE={row['RMSE']:.3f} R2={row['R2']:.4f} "
                    f"time={row['TrainTimeSec']:.1f}s",
                    flush=True,
                )


def stage_ablation(args, ics_cfg):
    csv_path = os.path.join(RESULT, "ablation_runs.csv")
    variants = ["full", "no_graph", "random_graph", "L1", "no_compress", "trainable_w"]
    ds = "PEMS08"
    for seed in args.seeds:
        data = load_data(ds, seed, args.max_train, args.max_val, args.max_test)
        for variant in variants:
            tag = "ICS-DGRN" if variant == "full" else f"ICS-DGRN/{variant}"
            if already(csv_path, dataset=ds, seed=seed, model=tag):
                print(f"skip ablation {tag} {seed}", flush=True)
                continue
            print(f"ABL {variant} seed {seed}", flush=True)
            out = run_one("ICS-DGRN", data, seed, args.epochs, args.batch_size, args.lr, args.patience, ics_cfg, variant=variant)
            row = out["row"]
            row["dataset"] = ds
            row["model"] = tag
            row["variant"] = variant
            append_csv(csv_path, row)
            print(f"  MAE={row['MAE']:.3f} trainable={row['TrainableParams']:.0f} t={row['TrainTimeSec']:.1f}s", flush=True)


def stage_compression(args, ics_cfg):
    csv_path = os.path.join(RESULT, "compression_runs.csv")
    ratios = (0.25, 0.5, 0.75, 1.0)
    ds = "PEMS08"
    base_dim = int(ics_cfg["layer_dims"][0])
    for seed in args.seeds[:2]:
        data = load_data(ds, seed, args.max_train, args.max_val, args.max_test)
        for r in ratios:
            tag = f"r={r}"
            if already(csv_path, dataset=ds, seed=seed, model=tag):
                continue
            print(f"COMP r={r} seed {seed}", flush=True)
            cfg = dict(ics_cfg)
            cfg["compression_dims"] = [max(int(round(base_dim * r)), 4)]
            # keep 2-layer widths
            out = run_one("ICS-DGRN", data, seed, max(16, args.epochs // 2), args.batch_size, args.lr, args.patience, cfg, variant="full")
            row = out["row"]
            row["dataset"] = ds
            row["model"] = tag
            row["ratio"] = r
            # storage proxy: phi entries * 4 bytes if dense, times density
            d0 = int(cfg["layer_dims"][0])
            c = int(cfg["compression_dims"][0])
            row["phi_entries"] = d0 * c
            row["phi_bytes"] = d0 * c * 4
            append_csv(csv_path, row)


def stage_esp(args, ics_cfg):
    """State-difference curves at several spectral targets. One trained model per tau."""
    out_dir = os.path.join(RESULT, "esp")
    os.makedirs(out_dir, exist_ok=True)
    ds = "PEMS08"
    data = load_data(ds, 42, args.max_train, args.max_val, args.max_test)
    taus = (0.5, 0.7, 0.9, 1.05, 1.3)
    scales = (0.25, 1.0)
    summary = []
    curves = {}
    for tau in taus:
        print(f"ESP tau={tau}", flush=True)
        cfg = dict(ics_cfg)
        cfg["esp_target"] = float(tau)
        # short fit so Win is in a realistic regime; ESP is about W,S
        out = run_one(
            "ICS-DGRN",
            data,
            42,
            epochs=max(8, args.sweep_epochs // 2),
            batch_size=args.batch_size,
            lr=args.lr,
            patience=4,
            ics_cfg=cfg,
        )
        model = out["pack"]["model"]
        s = out["pack"]["adj_t"]
        esp = model.esp_report(s)
        x = torch.tensor(data["x_test"][:1], dtype=torch.float32, device=s.device)
        for scale in scales:
            diffs = []
            for a, b in ((0, 1), (2, 3), (4, 5)):
                dec = model.state_difference_decay(x, s, seed_a=a, seed_b=b, init_scale=scale)
                d = dec["D"].detach().cpu().numpy()
                dhat = dec["D_hat"].detach().cpu().numpy()
                diffs.append(dhat)
            mean_hat = np.mean(np.stack(diffs, 0), axis=0)  # (L,T)
            curves[f"tau{tau}_s{scale}"] = mean_hat
            np.save(os.path.join(out_dir, f"Dhat_tau{tau}_scale{scale}.npy"), mean_hat)
        summary.append(
            {
                "tau": tau,
                "max_kappa": esp.get("max_kappa"),
                "esp_ok": esp.get("esp_ok"),
                "MAE": out["row"]["MAE"],
                "Dhat_last": float(curves[f"tau{tau}_s1.0"][-1, -1]),
                "Dhat_mid": float(curves[f"tau{tau}_s1.0"][-1, mean_hat.shape[1] // 2]),
            }
        )
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "esp_summary.csv"), index=False)
    with open(os.path.join(out_dir, "curves.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in curves.items()}, f)


def _mean_std(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    g = df.groupby(["dataset", "model"], as_index=False)
    rows = []
    for (ds, model), sub in df.groupby(["dataset", "model"]):
        rec = {"dataset": ds, "model": model, "n": len(sub)}
        for c in cols:
            if c not in sub.columns:
                continue
            v = sub[c].astype(float).to_numpy()
            rec[f"{c}_mean"] = float(np.mean(v))
            rec[f"{c}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            if len(v) > 1:
                tcrit = 2.776 if len(v) == 3 else (12.706 if len(v) == 2 else 1.96)
                # rough Student-t 95% for n=2,3; n>3 use 1.96
                if len(v) == 2:
                    tcrit = 12.706
                elif len(v) == 3:
                    tcrit = 4.303
                elif len(v) == 4:
                    tcrit = 3.182
                elif len(v) == 5:
                    tcrit = 2.776
                else:
                    tcrit = 1.96
                rec[f"{c}_ci"] = float(tcrit * rec[f"{c}_std"] / np.sqrt(len(v)))
            else:
                rec[f"{c}_ci"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def block_bootstrap_delta(err_a: np.ndarray, err_b: np.ndarray, block: int = 12, n_boot: int = 400, seed: int = 0) -> dict:
    """Paired block bootstrap of mean(err_b - err_a). Positive => a is better."""
    n = min(len(err_a), len(err_b))
    a = np.asarray(err_a[:n], dtype=np.float64)
    b = np.asarray(err_b[:n], dtype=np.float64)
    diff = b - a
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(0, n, block)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(starts), size=n_blocks, replace=True)
        chunks = [diff[starts[i] : starts[i] + block] for i in pick]
        sample = np.concatenate(chunks)[:n]
        boots.append(float(sample.mean()))
    boots = np.sort(np.asarray(boots))
    return {
        "delta_mae": float(diff.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def stage_figures():
    _style()
    os.makedirs(FIG, exist_ok=True)
    main_path = os.path.join(RESULT, "main_runs.csv")
    if not os.path.exists(main_path):
        print("no main_runs.csv")
        return
    raw = pd.read_csv(main_path)
    metric_cols = ["MAE", "RMSE", "R2", "TrainTimeSec", "InferTimeSec", "TrainableParams", "TotalParams", "PeakGPUMemMB", "G_MAE", "MAE_short", "MAE_mid", "MAE_long"]
    metric_cols += [f"MAE_h{i}" for i in range(1, 13)]
    summary = _mean_std(raw, [c for c in metric_cols if c in raw.columns])
    summary.to_csv(os.path.join(RESULT, "main_summary.csv"), index=False)

    # relative improvement vs best non-ICS baseline per dataset
    rel_rows = []
    for ds, sub in summary.groupby("dataset"):
        base = sub[sub["model"] != "ICS-DGRN"]
        ours = sub[sub["model"] == "ICS-DGRN"]
        if ours.empty or base.empty:
            continue
        best_name = base.loc[base["MAE_mean"].idxmin(), "model"]
        best = float(base["MAE_mean"].min())
        our = float(ours["MAE_mean"].iloc[0])
        rel_rows.append(
            {
                "dataset": ds,
                "best_baseline": best_name,
                "baseline_MAE": best,
                "ICS_MAE": our,
                "abs_improve": best - our,
                "rel_improve_pct": (best - our) / best * 100.0,
            }
        )
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(os.path.join(RESULT, "improvement_vs_best.csv"), index=False)

    # block bootstrap vs AGCRN if error files exist
    boot_rows = []
    for ds in summary["dataset"].unique():
        for seed in sorted(raw.loc[raw.dataset == ds, "seed"].unique()):
            fa = os.path.join(RESULT, f"err_{ds}_ICS-DGRN_s{int(seed)}.npy")
            fb = os.path.join(RESULT, f"err_{ds}_AGCRN_s{int(seed)}.npy")
            if os.path.exists(fa) and os.path.exists(fb):
                st = block_bootstrap_delta(np.load(fa), np.load(fb), seed=int(seed))
                st.update({"dataset": ds, "seed": int(seed), "baseline": "AGCRN"})
                boot_rows.append(st)
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(os.path.join(RESULT, "block_bootstrap_vs_agcrn.csv"), index=False)

    _fig_main_bars(summary)
    _fig_horizon(raw)
    _fig_efficiency(summary)
    _fig_ablation()
    _fig_compression()
    _fig_esp()
    _write_tex_tables(summary, rel)
    _write_replace_note(summary, rel)
    print("figures written to", FIG, flush=True)


def _fig_main_bars(summary: pd.DataFrame) -> None:
    models = [m for m in MAIN_MODELS if m in set(summary["model"])]
    datasets = [d for d in DATASETS if d in set(summary["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    x = np.arange(len(models))
    for ax, ds in zip(axes, datasets):
        sub = summary[summary.dataset == ds].set_index("model")
        means = [sub.loc[m, "MAE_mean"] if m in sub.index else np.nan for m in models]
        stds = [sub.loc[m, "MAE_std"] if m in sub.index else 0 for m in models]
        colors = [COLORS.get(m, "#333") for m in models]
        bars = ax.bar(x, means, yerr=stds, color=colors, width=0.72, capsize=3, error_kw={"lw": 0.8, "ecolor": "#222"})
        # highlight ours
        if "ICS-DGRN" in models:
            bars[models.index("ICS-DGRN")].set_edgecolor("#08382C")
            bars[models.index("ICS-DGRN")].set_linewidth(1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(ds)
        ax.set_ylabel("MAE" if ds == datasets[0] else "")
        finite = [v for v in means if np.isfinite(v)]
        if finite:
            lo, hi = min(finite), max(finite)
            pad = max(0.4, 0.25 * (hi - lo))
            ax.set_ylim(max(0, lo - pad), hi + pad * 1.35)
        for i, (m, s) in enumerate(zip(means, stds)):
            if np.isfinite(m):
                ax.text(i, m + (s if np.isfinite(s) else 0) + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")
    fig.suptitle("Test MAE (mean ± std across seeds)", y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_main_mae"))

    # RMSE companion
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = summary[summary.dataset == ds].set_index("model")
        means = [sub.loc[m, "RMSE_mean"] if m in sub.index else np.nan for m in models]
        stds = [sub.loc[m, "RMSE_std"] if m in sub.index else 0 for m in models]
        ax.bar(x, means, yerr=stds, color=[COLORS.get(m, "#333") for m in models], width=0.72, capsize=3, error_kw={"lw": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(ds)
        ax.set_ylabel("RMSE" if ds == datasets[0] else "")
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_main_rmse"))


def _fig_horizon(raw: pd.DataFrame) -> None:
    # average seeds
    focus = [m for m in ("ICS-DGRN", "AGCRN", "DCRNN", "STGCN", "GWNet") if m in set(raw["model"])]
    datasets = [d for d in DATASETS if d in set(raw["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.3 * len(datasets), 3.7), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    hs = np.arange(1, 13)
    for ax, ds in zip(axes, datasets):
        for m in focus:
            sub = raw[(raw.dataset == ds) & (raw.model == m)]
            if sub.empty:
                continue
            cols = [f"MAE_h{h}" for h in hs]
            mat = sub[cols].astype(float).to_numpy()
            mu = mat.mean(0)
            sd = mat.std(0, ddof=1) if len(mat) > 1 else np.zeros_like(mu)
            ax.plot(hs, mu, color=COLORS.get(m, "#333"), lw=2.4 if m == "ICS-DGRN" else 1.5, label=m, marker="o", ms=3.5)
            ax.fill_between(hs, mu - sd, mu + sd, color=COLORS.get(m, "#333"), alpha=0.12)
        ax.set_xlabel("Forecast horizon (5 min)")
        ax.set_title(ds)
        ax.set_xticks([1, 3, 6, 9, 12])
        ax.set_ylabel("MAE" if ds == datasets[0] else "")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_horizon"))


def _fig_efficiency(summary: pd.DataFrame) -> None:
    sub = summary[summary.dataset == "PEMS08"].copy()
    if sub.empty or "TrainableParams_mean" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for _, r in sub.iterrows():
        ax.scatter(
            r["TrainableParams_mean"] / 1000.0,
            r["MAE_mean"],
            s=40 + 0.15 * float(r.get("TrainTimeSec_mean", 40)),
            color=COLORS.get(r["model"], "#333"),
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.annotate(r["model"], (r["TrainableParams_mean"] / 1000.0, r["MAE_mean"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Trainable parameters (×10³)")
    ax.set_ylabel("PEMS08 test MAE")
    ax.set_title("Accuracy–parameter trade-off")
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_efficiency_scatter"))

    # grouped efficiency bars
    models = list(sub["model"])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    specs = [
        ("TrainableParams_mean", "Trainable params"),
        ("TrainTimeSec_mean", "Train time (s)"),
        ("PeakGPUMemMB_mean", "Peak GPU mem (MB)"),
    ]
    x = np.arange(len(models))
    for ax, (col, title) in zip(axes, specs):
        if col not in sub.columns:
            continue
        vals = sub[col].to_numpy()
        ax.bar(x, vals, color=[COLORS.get(m, "#333") for m in models])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_efficiency_bars"))


def _fig_ablation():
    path = os.path.join(RESULT, "ablation_runs.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    order = ["full", "no_graph", "random_graph", "L1", "no_compress", "trainable_w"]
    labels = {
        "full": "Full ICS-DGRN",
        "no_graph": "No graph propagation",
        "random_graph": "Rewired graph",
        "L1": "Single layer",
        "no_compress": "No compression",
        "trainable_w": "Trainable reservoir",
    }
    rows = []
    for v in order:
        sub = df[df.variant == v] if "variant" in df.columns else df[df.model.str.endswith(v)]
        if sub.empty:
            continue
        rows.append((labels.get(v, v), float(sub.MAE.mean()), float(sub.MAE.std(ddof=1)) if len(sub) > 1 else 0.0))
    if not rows:
        return
    names, means, stds = zip(*rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(names))[::-1]
    colors = ["#0F6E56" if n.startswith("Full") else "#94A3B8" for n in names]
    ax.barh(y, means, xerr=stds, color=colors, capsize=3, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("PEMS08 test MAE")
    full = means[0]
    for yi, m in zip(y, means):
        delta = (m - full) / full * 100
        ax.text(m, yi, f"  {m:.2f}  ({delta:+.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, max(means) * 1.28)
    ax.set_title("Module ablation")
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_ablation"))
    pd.DataFrame({"variant": names, "MAE": means, "std": stds}).to_csv(os.path.join(RESULT, "ablation_summary.csv"), index=False)


def _fig_compression():
    path = os.path.join(RESULT, "compression_runs.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    g = df.groupby("ratio", as_index=False).agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), phi_bytes=("phi_bytes", "mean"), sec=("TrainTimeSec", "mean"))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(g["phi_bytes"] / 1024.0, g["MAE"], color="#0F6E56", marker="o", lw=2)
    for _, r in g.iterrows():
        ax.annotate(f"r={r['ratio']:.2f}", (r["phi_bytes"] / 1024.0, r["MAE"]), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("Inter-layer map storage (KB)")
    ax.set_ylabel("PEMS08 MAE")
    ax.set_title("Compression–accuracy trade-off")
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_compression"))
    g.to_csv(os.path.join(RESULT, "compression_summary.csv"), index=False)


def _fig_esp():
    summary_path = os.path.join(RESULT, "esp", "esp_summary.csv")
    if not os.path.exists(summary_path):
        return
    summary = pd.read_csv(summary_path)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    # panel A: decay curves at scale 1
    ax = axes[0]
    cmap = {0.5: "#93C5FD", 0.7: "#2563EB", 0.9: "#0F6E56", 1.05: "#D97706", 1.3: "#B91C1C"}
    for _, r in summary.iterrows():
        tau = float(r["tau"])
        f = os.path.join(RESULT, "esp", f"Dhat_tau{tau}_scale1.0.npy")
        if not os.path.exists(f):
            # filenames use the python float repr of the loop variable
            matches = [p for p in os.listdir(os.path.join(RESULT, "esp")) if p.startswith(f"Dhat_tau{tau}") and p.endswith("scale1.0.npy")]
            if not matches:
                continue
            f = os.path.join(RESULT, "esp", matches[0])
        dhat = np.load(f)
        t = np.arange(1, dhat.shape[1] + 1)
        ok = "satisfied" if bool(r["esp_ok"]) else "not satisfied"
        ax.plot(t, dhat[-1], color=cmap.get(tau, "#333"), lw=2, label=fr"$\tau$={tau:g} ({ok})")
    ax.set_yscale("log")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Normalized state gap $\tilde d_t^{(L)}$")
    ax.set_title("Last-layer state gap")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    # per-layer for tau closest to 0.9 and a violating tau
    for tau, ls in ((0.9, "-"), (1.3, "--")):
        matches = [p for p in os.listdir(os.path.join(RESULT, "esp")) if p.startswith(f"Dhat_tau{tau}") and "scale1.0" in p]
        if not matches:
            continue
        dhat = np.load(os.path.join(RESULT, "esp", matches[0]))
        t = np.arange(1, dhat.shape[1] + 1)
        for li in range(dhat.shape[0]):
            ax.plot(t, dhat[li], ls=ls, lw=1.8, label=fr"$\tau$={tau:g}, layer {li+1}")
    ax.set_yscale("log")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$\tilde d_t^{(l)}$")
    ax.set_title("Per-layer gap")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    _save(fig, os.path.join(FIG, "F_esp"))


def _write_tex_tables(summary: pd.DataFrame, rel: pd.DataFrame) -> None:
    # wide MAE/RMSE/R2 table
    models = [m for m in MAIN_MODELS if m in set(summary["model"])]
    datasets = [d for d in DATASETS if d in set(summary["dataset"])]
    lines = [
        "% Auto-generated by experiments/run_p0_bundle.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{多随机种子测试性能（均值$\\pm$标准差）。ICS-DGRN 的循环权重与层间投影在初始化后保持固定。}",
        "\\label{tab:main_results}",
        "\\small",
        "\\begin{tabular}{l" + "ccc" * len(datasets) + "}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Model}} & " + " & ".join([f"\\multicolumn{{3}}{{c}}{{\\textbf{{{ds}}}}}" for ds in datasets]) + " \\\\",
    ]
    # cmidrules
    col = 2
    rules = []
    for _ in datasets:
        rules.append(f"\\cmidrule(lr){{{col}-{col+2}}}")
        col += 3
    lines.append(" ".join(rules))
    head = []
    for _ in datasets:
        head += ["MAE", "RMSE", "$R^2$"]
    lines.append("& " + " & ".join(head) + " \\\\")
    lines.append("\\midrule")
    for m in models:
        cells = [f"\\textbf{{{m}}}" if m == "ICS-DGRN" else m]
        for ds in datasets:
            sub = summary[(summary.dataset == ds) & (summary.model == m)]
            if sub.empty:
                cells += ["--", "--", "--"]
                continue
            r = sub.iloc[0]
            # bold if best MAE in dataset
            best = summary[summary.dataset == ds]["MAE_mean"].min()
            mae_s = f"{r['MAE_mean']:.2f}$\\pm${r['MAE_std']:.2f}"
            rmse_s = f"{r['RMSE_mean']:.2f}$\\pm${r['RMSE_std']:.2f}"
            r2_s = f"{r['R2_mean']:.3f}"
            if abs(r["MAE_mean"] - best) < 1e-6:
                mae_s = f"\\textbf{{{mae_s}}}"
            cells += [mae_s, rmse_s, r2_s]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    if rel is not None and len(rel):
        lines.append("% improvement vs best baseline")
        for _, r in rel.iterrows():
            lines.append(
                f"% {r['dataset']}: vs {r['best_baseline']}  abs {r['abs_improve']:.3f}  rel {r['rel_improve_pct']:.2f}%"
            )
    text = "\n".join(lines)
    with open(os.path.join(FIG, "tables_snippet.tex"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(RESULT, "tables_snippet.tex"), "w", encoding="utf-8") as f:
        f.write(text)


def _write_replace_note(summary: pd.DataFrame, rel: pd.DataFrame) -> None:
    path = os.path.join(RESULT, "替换说明.md")
    lines = [
        "# 新图表替换对照",
        "",
        "新结果目录：`feifa2/result/p0_support/`，论文图：`feifa2/tex/figure/p0/`。",
        "",
        "协议：历史 12 步预测未来 12 步；ICS-DGRN 结构只在 PEMS08 验证集上选定；循环权重固定。层间投影是否可训练以 `ics_config.json` 为准。",
        "",
        "## 正文替换",
        "",
        "| 新文件 | 替换现在的内容 | 作用 |",
        "| --- | --- | --- |",
        "| `tex/figure/p0/tables_snippet.tex` 与 `F_main_mae.pdf` | 表 `tab:main_results`，以及图 `fig:multi_seed` 整段 | 用多种子均值±标准差替换单次运行主表，三种子柱状图不再单列 |",
        "| `F_horizon.pdf` | 图 `fig:multi_horizon` | 三个数据集上 1–12 步 MAE 曲线，替代只在 PEMS08 上的增长条形图 |",
        "| `F_ablation.pdf` | 表 `tab:ridge_ablation` | 用同一套深度学习读出做图传播、随机重连、单层、去压缩、可训练循环权重消融 |",
        "| `F_compression.pdf` | 图 `fig:compression_ablation` | 压缩率对存储和 MAE 的折中，替代仅看误差几乎不动的双折线 |",
        "| `F_efficiency_scatter.pdf` 与 `F_efficiency_bars.pdf` | 表 `tab:fixed_vs_trainable`、表 `tab:overhead`、图 `fig:compute_overhead` | 把固定/可训练与各基线的参数、时间和显存收成一张效率图 |",
        "| `F_esp.pdf` | 图 `fig:state_decay` 与图 `fig:esp_forecast_joint` | 只保留状态差曲线。预测 MAE 不再和 ESP 画在同一张图里 |",
        "",
        "定性预测图 `fig:scenarios` 可以留在正文，但不要再用它承担性能结论。",
        "",
        "## 建议移入补充材料、不必再占正文",
        "",
        "- 旧的逐步长数值表、全部种子原始行：`result/p0_support/main_runs.csv`",
        "- 与 AGCRN 的时间块 Bootstrap：`block_bootstrap_vs_agcrn.csv`",
        "- 验证集选型记录：`sweep_val.csv`、`ics_config.json`",
        "- 压缩扫描明细：`compression_runs.csv`",
        "",
        "## 这次相对旧结果改了什么",
        "",
        "- 主结果改为多种子，并增加 DCRNN 与 DeepESN。",
        "- 图传播消融改到与主模型相同的训练读出，不再用另一套 Ridge 大表。",
        "- 增加保持边数的随机重连图，用来区分真实拓扑和普通混合。",
        "- ESP 图只报告归一化状态差，并分开画满足与不满足充分条件的谱尺度。",
        "",
    ]
    if rel is not None and len(rel):
        lines.append("## 相对最强基线的 MAE 变化（正数表示 ICS-DGRN 更低）")
        lines.append("")
        for _, r in rel.iterrows():
            lines.append(
                f"- {r['dataset']}: 对比 {r['best_baseline']}，绝对差 {r['abs_improve']:.3f}，相对 {r['rel_improve_pct']:.2f}%"
            )
        lines.append("")
    lines.append("写作时只声称表格里确实更优的数据集和指标。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["sweep", "main", "ablation", "compression", "esp", "figures", "all"])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=["ICS-DGRN", "AGCRN", "DCRNN", "STGCN", "DeepESN"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--sweep-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=1600)
    parser.add_argument("--max-val", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--force-sweep", action="store_true")
    args = parser.parse_args()
    os.makedirs(RESULT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    print("device", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    ics_cfg = {
        "layer_dims": [56, 40],
        "compression_dims": [28],
        "train_phi": False,
        "esp_target": 0.9,
        "name": "default",
    }
    if args.stage in ("sweep", "all", "main", "ablation", "compression", "esp"):
        if args.stage in ("sweep", "all") or not os.path.exists(os.path.join(RESULT, "ics_config.json")):
            ics_cfg = sweep(args)
        else:
            with open(os.path.join(RESULT, "ics_config.json"), encoding="utf-8") as f:
                ics_cfg = json.load(f)
    if args.stage in ("main", "all"):
        stage_main(args, ics_cfg)
    if args.stage in ("ablation", "all"):
        stage_ablation(args, ics_cfg)
    if args.stage in ("compression", "all"):
        stage_compression(args, ics_cfg)
    if args.stage in ("esp", "all"):
        stage_esp(args, ics_cfg)
    if args.stage in ("figures", "all"):
        stage_figures()


if __name__ == "__main__":
    main()
