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

"""Mechanism-validation experiments for ICS-DGRN (compression, ESP decay, multi-seed)."""

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.trainable_st import (
    TrainConfig,
    build_model,
    build_normalized_adj,
    count_params,
    predict_st_model,
    train_st_model,
)
from data.pems_loader import load_pems
from model.dgrn_trainable import build_ics_dgrn_trainable
from utils.metrics import evaluate_all, mae
from utils import viz

RESULT_ROOT = os.path.join(ROOT, "result", "pems_newexp")

DEFAULT_LAYER_DIMS = (56, 40)
DEFAULT_COMP = (28,)

def _cfg(epochs: int, seed: int, batch_size: int = 48) -> TrainConfig:
    return TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=1.2e-3,
        patience=max(epochs, 20),
        seed=seed,
        device="auto",
        loss="smooth_l1",
        weight_decay=5e-5,
    )

def _peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    return 0.0

def _reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def train_ics(
    data: dict,
    *,
    epochs: int,
    seed: int,
    compression_ratio: Optional[float] = None,
    compression_dims: Optional[Tuple[int, ...]] = None,
    phi_density: float = 1.0,
    esp_target: float = 0.9,
    train_recurrent: bool = False,
    track_mem: bool = False,
):
    """Train one ICS-DGRN variant; return pack, preds, row, esp, hmae, yt."""
    N, H, hist = data["n_nodes"], data["horizon"], data["hist_len"]
    model = build_ics_dgrn_trainable(
        N,
        hist,
        H,
        layer_dims=DEFAULT_LAYER_DIMS,
        compression_dims=compression_dims,
        compression_ratio=compression_ratio,
        esp_target=esp_target,
        phi_density=phi_density,
        train_recurrent=train_recurrent,
        seed=seed,
    )
    pstats = model.param_stats()
    if track_mem:
        _reset_peak_mem()
    t_bw0 = time.perf_counter()
    pack = train_st_model(model, data["adj"], data["x_train"], data["y_train"], data["x_val"], data["y_val"], _cfg(epochs, seed))
    # rough backward+train wall already in pack; optional peak mem
    peak = _peak_mem_mb() if track_mem else 0.0
    yp, infer_sec = predict_st_model(pack, data["x_test"])
    ypd = data["denorm"](yp)
    yt = data["denorm"](data["y_test"])
    acc = evaluate_all(yt, ypd)
    esp = {}
    try:
        esp = pack["model"].esp_report(pack["adj_t"])
    except Exception as e:
        esp = {"error": str(e)}
    hmae = [mae(yt[:, h], ypd[:, h]) for h in range(yt.shape[1])]
    row = {
        **acc,
        "TrainTimeSec": pack["train_sec"],
        "InferTimeSec": infer_sec,
        "TrainableParams": float(pack["params"]),
        "TotalParams": float(pstats["total_params"]),
        "Epochs": float(pack["epochs_ran"]),
        "best_val_mae": pack["best_val_mae"],
        "PeakGPUMemMB": peak,
        "MAE_h1": hmae[0],
        "MAE_h12": hmae[-1],
        "DeltaMAE": hmae[-1] - hmae[0],
        "G_horizon": (hmae[-1] - hmae[0]) / max(hmae[0], 1e-8),
        "esp_ok": bool(esp.get("esp_ok", False)),
        "max_kappa": float(esp.get("max_kappa", np.nan)),
        "esp_target": float(esp_target),
        "phi_density": float(phi_density),
        "train_recurrent": bool(train_recurrent),
        "seed": int(seed),
    }
    if compression_ratio is not None:
        row["compression_ratio"] = float(compression_ratio)
    if compression_dims is not None:
        row["compression_dim"] = int(compression_dims[0])
    elif compression_ratio is not None:
        row["compression_dim"] = int(round(DEFAULT_LAYER_DIMS[0] * compression_ratio))
    else:
        row["compression_dim"] = int(DEFAULT_COMP[0])
        row["compression_ratio"] = row["compression_dim"] / float(DEFAULT_LAYER_DIMS[0])
    # phi spectral norms
    if "phi_spectral_norms" in esp:
        for i, v in enumerate(esp["phi_spectral_norms"]):
            row[f"phi_spec_norm_l{i}"] = float(v)
    return pack, ypd, row, esp, hmae, yt

def load_data(ds: str, seed: int, max_train: int, max_val: int, max_test: int) -> dict:
    return load_pems(ds, max_train=max_train, max_val=max_val, max_test=max_test, seed=seed)

# ---------------------------------------------------------------------------
# 1. ESP state-difference decay
# ---------------------------------------------------------------------------
def run_esp_decay(args) -> None:
    out = os.path.join(RESULT_ROOT, "esp_state_decay")
    os.makedirs(out, exist_ok=True)
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = build_normalized_adj(data["adj"], device)
    # use one normalized test window as driving input
    x = torch.tensor(data["x_test"][:1], dtype=torch.float32, device=device)

    taus = [0.5, 0.7, 0.9, 1.0, 1.1]
    summary_rows = []
    viz.set_style()

    for tau in taus:
        model = build_ics_dgrn_trainable(
            data["n_nodes"],
            data["hist_len"],
            data["horizon"],
            esp_target=tau,
            seed=args.seed,
        ).to(device)
        model.set_s_norm(s)
        pack = model.state_difference_decay(x, s, seed_a=0, seed_b=1)
        D_hat = pack["D_hat"].cpu().numpy()  # (L, T)
        D = pack["D"].cpu().numpy()
        T = D_hat.shape[1]
        t_axis = np.arange(1, T + 1)

        # decay rate: mean log ratio over last half vs first 3 steps
        rates = []
        for li in range(D_hat.shape[0]):
            early = float(np.mean(D_hat[li, :3]))
            late = float(np.mean(D_hat[li, T // 2 :]))
            rates.append(late / max(early, 1e-12))
            np.savetxt(
                os.path.join(out, f"{ds}_tau{tau:.1f}_layer{li+1}_Dhat.csv"),
                np.column_stack([t_axis, D_hat[li], D[li]]),
                delimiter=",",
                header="t,D_hat,D",
                comments="",
            )

        esp = model.esp_report(s)
        label = "Strong" if max(rates) < 0.15 and esp.get("max_kappa", 99) < 1 else (
            "Moderate" if max(rates) < 0.5 else "Weak/Unstable"
        )
        summary_rows.append(
            {
                "tau": tau,
                "max_kappa": esp.get("max_kappa"),
                "esp_ok": esp.get("esp_ok"),
                "layer1_late_over_early": rates[0],
                "layer2_late_over_early": rates[1] if len(rates) > 1 else np.nan,
                "final_Dhat_mean": float(np.mean(D_hat[:, -1])),
                "StateDecay": label,
            }
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for li in range(D_hat.shape[0]):
            ax.semilogy(t_axis, np.clip(D_hat[li], 1e-8, None), label=f"Layer {li+1}", lw=1.8)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\hat{D}_l(t)$")
        ax.set_title(f"{ds} State Difference Decay | τ={tau} | max κ={esp.get('max_kappa', float('nan')):.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{ds}_tau{tau:.1f}_state_decay.png"), bbox_inches="tight")
        plt.close(fig)

    # overlay final-layer across tau
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for tau in taus:
        path = os.path.join(out, f"{ds}_tau{tau:.1f}_layer{len(DEFAULT_LAYER_DIMS)}_Dhat.csv")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        ax.semilogy(arr[:, 0], np.clip(arr[:, 1], 1e-8, None), label=f"τ={tau}", lw=1.6)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\hat{D}_L(t)$ (final layer)")
    ax.set_title(f"{ds} Final-layer State Difference Decay vs Spectral Scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{ds}_state_decay_vs_tau.png"), bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(out, f"{ds}_esp_decay_summary.csv"), index=False)
    with open(os.path.join(out, f"{ds}_esp_decay_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print("[esp_decay] saved", out)

# ---------------------------------------------------------------------------
# helper: sensitivity sweep
# ---------------------------------------------------------------------------
def _save_sensitivity(out: str, ds: str, tag: str, rows: List[dict], x_key: str, title: str) -> None:
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{ds}_{tag}_metrics.csv"), index=False)
    viz.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    xs = df[x_key].values
    for ax, metric, ylab in zip(axes, ["MAE", "RMSE", "R2"], ["MAE", "RMSE", r"$R^2$"]):
        ax.plot(xs, df[metric].values, marker="o", lw=1.8, color="#1f4e79")
        ax.set_xlabel(x_key)
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{ds}_{tag}_curves.png"), bbox_inches="tight")
    plt.close(fig)

    # params vs performance
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(df[x_key], df["TrainableParams"], marker="s", color="#c45911", label="TrainableParams")
    ax.set_xlabel(x_key)
    ax.set_ylabel("Trainable parameters")
    ax2 = ax.twinx()
    ax2.plot(df[x_key], df["MAE"], marker="o", color="#1f4e79", label="MAE")
    ax2.set_ylabel("MAE")
    ax.set_title(f"{title} — capacity vs accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{ds}_{tag}_params_vs_mae.png"), bbox_inches="tight")
    plt.close(fig)

def run_compression(args) -> None:
    out = os.path.join(RESULT_ROOT, "compression_ratio")
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
    rows = []
    for r in ratios:
        print("=" * 60, f"compression r={r}")
        pack, ypd, row, esp, hmae, yt = train_ics(
            data, epochs=args.epochs, seed=args.seed, compression_ratio=r
        )
        row["dataset"] = ds
        rows.append(row)
        print(f"  r={r} MAE={row['MAE']:.4f} params={row['TrainableParams']:.0f} cdim={row['compression_dim']}")
    os.makedirs(out, exist_ok=True)
    _save_sensitivity(
        out,
        ds,
        "compression",
        rows,
        "compression_ratio",
        f"{ds} Interlayer Compression Ratio Sensitivity",
    )
    with open(os.path.join(out, f"{ds}_compression_esp.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=float)
    print("[compression] saved", out)

def run_density(args) -> None:
    out = os.path.join(RESULT_ROOT, "sparse_density")
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    densities = [1.0, 0.5, 0.2, 0.1, 0.05]
    rows = []
    for rho in densities:
        print("=" * 60, f"phi_density ρ={rho}")
        pack, ypd, row, esp, hmae, yt = train_ics(
            data, epochs=args.epochs, seed=args.seed, phi_density=rho
        )
        row["dataset"] = ds
        row["rho"] = rho
        rows.append(row)
        print(f"  ρ={rho} MAE={row['MAE']:.4f} phi_norm={row.get('phi_spec_norm_l0', float('nan')):.4f}")
    _save_sensitivity(
        out,
        ds,
        "density",
        rows,
        "rho",
        f"{ds} Compression Sparse-Density Sensitivity",
    )
    print("[density] saved", out)

def run_esp_forecast(args) -> None:
    out = os.path.join(RESULT_ROOT, "esp_forecast_joint")
    os.makedirs(out, exist_ok=True)
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = build_normalized_adj(data["adj"], device)
    x = torch.tensor(data["x_test"][:1], dtype=torch.float32, device=device)
    taus = [0.5, 0.7, 0.9, 1.0, 1.1]
    rows = []
    for tau in taus:
        print("=" * 60, f"esp_forecast τ={tau}")
        pack, ypd, row, esp, hmae, yt = train_ics(
            data, epochs=args.epochs, seed=args.seed, esp_target=tau
        )
        # state decay on trained model
        model = pack["model"]
        decay = model.state_difference_decay(x.to(next(model.parameters()).device), pack["adj_t"], seed_a=0, seed_b=1)
        D_hat = decay["D_hat"].cpu().numpy()
        late = float(np.mean(D_hat[:, D_hat.shape[1] // 2 :]))
        early = float(np.mean(D_hat[:, :3]))
        ratio = late / max(early, 1e-12)
        if row["max_kappa"] < 1 and ratio < 0.15:
            decay_label = "Strong"
        elif ratio < 0.5:
            decay_label = "Moderate"
        else:
            decay_label = "Weak/Unstable"
        row.update(
            {
                "dataset": ds,
                "tau": tau,
                "StateDecay": decay_label,
                "decay_late_over_early": ratio,
                "horizon_mae": hmae,
            }
        )
        rows.append(row)
        print(f"  τ={tau} κ={row['max_kappa']:.3f} decay={decay_label} MAE={row['MAE']:.4f}")

        np.savetxt(
            os.path.join(out, f"{ds}_tau{tau:.1f}_horizon_mae.csv"),
            np.column_stack([np.arange(1, len(hmae) + 1), hmae]),
            delimiter=",",
            header="h,MAE",
            comments="",
        )

    df = pd.DataFrame(rows)
    cols = ["tau", "max_kappa", "StateDecay", "MAE", "RMSE", "R2", "MAE_h12", "G_horizon", "esp_ok"]
    df[cols].to_csv(os.path.join(out, f"{ds}_esp_forecast_table.csv"), index=False)
    df.to_csv(os.path.join(out, f"{ds}_esp_forecast_full.csv"), index=False)

    viz.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(df["tau"], df["max_kappa"], "o-", color="#c45911", label=r"$\max_l \kappa_l$")
    axes[0].axhline(1.0, color="gray", ls="--", lw=1)
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$\max_l \kappa_l$")
    axes[0].set_title("Theoretical contraction quantity")
    axes[1].plot(df["tau"], df["MAE"], "o-", color="#1f4e79", label="MAE")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Forecasting MAE vs spectral scaling")
    fig.suptitle(f"{ds} ESP Condition ↔ Forecasting", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{ds}_esp_forecast_joint.png"), bbox_inches="tight")
    plt.close(fig)
    print("[esp_forecast] saved", out)

def run_fixed_vs_trainable(args) -> None:
    out = os.path.join(RESULT_ROOT, "fixed_vs_trainable")
    os.makedirs(out, exist_ok=True)
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    rows = []
    for train_w, name in [(False, "FixedReservoir"), (True, "TrainableReservoir")]:
        print("=" * 60, name)
        pack, ypd, row, esp, hmae, yt = train_ics(
            data,
            epochs=args.epochs,
            seed=args.seed,
            train_recurrent=train_w,
            track_mem=True,
        )
        row["variant"] = name
        row["dataset"] = ds
        rows.append(row)
        print(
            f"  {name}: MAE={row['MAE']:.4f} trainable={row['TrainableParams']:.0f} "
            f"total={row['TotalParams']:.0f} mem={row['PeakGPUMemMB']:.1f}MB time={row['TrainTimeSec']:.1f}s"
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{ds}_fixed_vs_trainable.csv"), index=False)

    viz.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    labels = df["variant"].tolist()
    axes[0].bar(labels, df["MAE"], color=["#1f4e79", "#c45911"])
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Test MAE")
    axes[1].bar(labels, df["TrainableParams"], color=["#1f4e79", "#c45911"])
    axes[1].set_ylabel("# trainable params")
    axes[1].set_title("Trainable parameters")
    axes[2].bar(labels, df["TrainTimeSec"], color=["#1f4e79", "#c45911"])
    axes[2].set_ylabel("seconds")
    axes[2].set_title("Training time")
    fig.suptitle(f"{ds} Fixed vs Trainable Reservoir", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{ds}_fixed_vs_trainable.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fixed_vs_trainable] saved", out)

def run_multi_seed(args) -> None:
    out = os.path.join(RESULT_ROOT, "multi_seed")
    os.makedirs(out, exist_ok=True)
    seeds = list(args.seeds)
    # Prefer all three PEMS sets for stability reporting when caller left default.
    datasets = list(args.datasets)
    if getattr(args, "multiseed_datasets", None):
        datasets = list(args.multiseed_datasets)
    all_rows = []
    for ds in datasets:
        for seed in seeds:
            print("=" * 60, ds, "ICS-DGRN", f"seed={seed}")
            data = load_data(ds, seed, args.max_train, args.max_val, args.max_test)
            pack, ypd, row, esp, hmae, yt = train_ics(data, epochs=args.epochs, seed=seed)
            row["dataset"] = ds
            row["model"] = "ICS-DGRN"
            all_rows.append(row)
            print(f"  => MAE={row['MAE']:.4f}")

            # AGCRN multi-seed mainly needed on PEMS08 (near-tie in main table)
            if args.with_agcrn and ds == "PEMS08":
                print("=" * 60, ds, "AGCRN", f"seed={seed}")
                model = build_model("AGCRN", data["n_nodes"], data["hist_len"], data["horizon"])
                cfg = TrainConfig(
                    epochs=args.epochs,
                    batch_size=32,
                    lr=1e-3,
                    patience=max(args.epochs, 20),
                    seed=seed,
                    device="auto",
                    loss="smooth_l1",
                    weight_decay=1e-4,
                )
                pack2 = train_st_model(
                    model, data["adj"], data["x_train"], data["y_train"], data["x_val"], data["y_val"], cfg
                )
                yp2, infer2 = predict_st_model(pack2, data["x_test"])
                yt = data["denorm"](data["y_test"])
                acc = evaluate_all(yt, data["denorm"](yp2))
                all_rows.append(
                    {
                        **acc,
                        "TrainTimeSec": pack2["train_sec"],
                        "InferTimeSec": infer2,
                        "TrainableParams": float(pack2["params"]),
                        "Epochs": float(pack2["epochs_ran"]),
                        "dataset": ds,
                        "model": "AGCRN",
                        "seed": seed,
                    }
                )
                print(f"  => AGCRN MAE={acc['MAE']:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out, "all_seed_runs.csv"), index=False)

    # aggregate mean±std
    agg_rows = []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        agg_rows.append(
            {
                "dataset": ds,
                "model": model,
                "n_seeds": len(g),
                "MAE_mean": g["MAE"].mean(),
                "MAE_std": g["MAE"].std(ddof=1) if len(g) > 1 else 0.0,
                "RMSE_mean": g["RMSE"].mean(),
                "RMSE_std": g["RMSE"].std(ddof=1) if len(g) > 1 else 0.0,
                "R2_mean": g["R2"].mean(),
                "R2_std": g["R2"].std(ddof=1) if len(g) > 1 else 0.0,
            }
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(os.path.join(out, "multi_seed_summary.csv"), index=False)

    # paper-style strings
    lines = ["# Multi-seed summary (mean ± std)", ""]
    for _, r in agg.iterrows():
        lines.append(
            f"- **{r['dataset']} / {r['model']}** (n={int(r['n_seeds'])}): "
            f"MAE={r['MAE_mean']:.3f}±{r['MAE_std']:.3f}, "
            f"RMSE={r['RMSE_mean']:.3f}±{r['RMSE_std']:.3f}, "
            f"R²={r['R2_mean']:.4f}±{r['R2_std']:.4f}"
        )
    with open(os.path.join(out, "REPORT_snippet.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    viz.set_style()
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        models = list(sub["model"].unique())
        means = [sub[sub["model"] == m]["MAE"].mean() for m in models]
        stds = [
            sub[sub["model"] == m]["MAE"].std(ddof=1) if sub[sub["model"] == m].shape[0] > 1 else 0
            for m in models
        ]
        ax.bar(models, means, yerr=stds, capsize=4, color=["#1f4e79", "#70ad47"][: len(models)])
        ax.set_ylabel("MAE")
        ax.set_title(f"{ds} Multi-seed MAE (mean±std)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{ds}_multiseed_mae.png"), bbox_inches="tight")
        plt.close(fig)
    print("[multi_seed] saved", out)

def run_multi_horizon(args) -> None:
    """Train ICS-DGRN + baselines once; report G_horizon / ΔMAE."""
    out = os.path.join(RESULT_ROOT, "multi_horizon")
    os.makedirs(out, exist_ok=True)
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    yt = data["denorm"](data["y_test"])
    models = ["ICS-DGRN", "STGCN", "TGCN", "GWNet", "AGCRN"]
    rows = []
    hmae_map = {}
    for name in models:
        print("=" * 60, "multi_horizon", name)
        if name == "ICS-DGRN":
            pack, ypd, row, esp, hmae, _ = train_ics(data, epochs=args.epochs, seed=args.seed)
        else:
            model = build_model(name, data["n_nodes"], data["hist_len"], data["horizon"])
            cfg = TrainConfig(
                epochs=args.epochs,
                batch_size=32,
                lr=1e-3,
                patience=max(args.epochs, 20),
                seed=args.seed,
                device="auto",
                loss="smooth_l1",
                weight_decay=1e-4,
            )
            pack = train_st_model(model, data["adj"], data["x_train"], data["y_train"], data["x_val"], data["y_val"], cfg)
            yp, infer = predict_st_model(pack, data["x_test"])
            ypd = data["denorm"](yp)
            acc = evaluate_all(yt, ypd)
            hmae = [mae(yt[:, h], ypd[:, h]) for h in range(yt.shape[1])]
            row = {
                **acc,
                "TrainTimeSec": pack["train_sec"],
                "InferTimeSec": infer,
                "TrainableParams": float(pack["params"]),
                "MAE_h1": hmae[0],
                "MAE_h12": hmae[-1],
                "DeltaMAE": hmae[-1] - hmae[0],
                "G_horizon": (hmae[-1] - hmae[0]) / max(hmae[0], 1e-8),
            }
        row["model"] = name
        row["dataset"] = ds
        rows.append(row)
        hmae_map[name] = hmae
        print(f"  {name}: MAE={row['MAE']:.4f} G_horizon={row['G_horizon']:.4f} ΔMAE={row['DeltaMAE']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{ds}_horizon_growth.csv"), index=False)
    # detailed per-horizon
    detail = pd.DataFrame({m: hmae_map[m] for m in hmae_map})
    detail.index = np.arange(1, len(next(iter(hmae_map.values()))) + 1)
    detail.index.name = "horizon"
    detail.to_csv(os.path.join(out, f"{ds}_horizon_mae_detail.csv"))

    viz.plot_horizon_curves(hmae_map, os.path.join(out, f"{ds}_horizon_mae.png"), title=f"{ds} MAE vs horizon")
    viz.set_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(df["model"], df["G_horizon"], color="#1f4e79")
    ax.set_ylabel(r"$G_{\mathrm{horizon}}=(MAE_{12}-MAE_1)/MAE_1$")
    ax.set_title(f"{ds} Horizon Error Growth")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{ds}_G_horizon_bar.png"), bbox_inches="tight")
    plt.close(fig)
    print("[multi_horizon] saved", out)

def run_compute(args) -> None:
    """Parameterization / memory overhead vs baselines (single dataset)."""
    out = os.path.join(RESULT_ROOT, "compute_overhead")
    os.makedirs(out, exist_ok=True)
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    rows = []
    for name in ["ICS-DGRN", "STGCN", "TGCN", "GWNet", "AGCRN"]:
        print("=" * 60, "compute", name)
        _reset_peak_mem()
        if name == "ICS-DGRN":
            pack, ypd, row, esp, hmae, yt = train_ics(
                data, epochs=args.epochs, seed=args.seed, track_mem=True
            )
            row["model"] = name
        else:
            model = build_model(name, data["n_nodes"], data["hist_len"], data["horizon"])
            total = int(sum(p.numel() for p in model.parameters()))
            cfg = TrainConfig(
                epochs=args.epochs,
                batch_size=32,
                lr=1e-3,
                patience=max(args.epochs, 20),
                seed=args.seed,
                device="auto",
                loss="smooth_l1",
                weight_decay=1e-4,
            )
            pack = train_st_model(model, data["adj"], data["x_train"], data["y_train"], data["x_val"], data["y_val"], cfg)
            yp, infer = predict_st_model(pack, data["x_test"])
            yt = data["denorm"](data["y_test"])
            acc = evaluate_all(yt, data["denorm"](yp))
            row = {
                **acc,
                "model": name,
                "TrainTimeSec": pack["train_sec"],
                "InferTimeSec": infer,
                "TrainableParams": float(pack["params"]),
                "TotalParams": float(total),
                "PeakGPUMemMB": _peak_mem_mb(),
                "Epochs": float(pack["epochs_ran"]),
            }
        row["dataset"] = ds
        rows.append(row)
        print(f"  {name}: trainable={row['TrainableParams']:.0f} mem={row.get('PeakGPUMemMB', 0):.1f}MB")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{ds}_compute_overhead.csv"), index=False)
    viz.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].bar(df["model"], df["TrainableParams"], color="#1f4e79")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Trainable params")
    axes[1].bar(df["model"], df["TrainTimeSec"], color="#c45911")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("Train time (s)")
    axes[2].bar(df["model"], df["PeakGPUMemMB"], color="#548235")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_title("Peak GPU memory (MB)")
    fig.suptitle(f"{ds} Computational Overhead", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{ds}_compute_overhead.png"), bbox_inches="tight")
    plt.close(fig)
    print("[compute] saved", out)

def run_qualitative(args) -> None:
    """Multi-scenario qualitative curves: ICS-DGRN vs AGCRN."""
    out = os.path.join(RESULT_ROOT, "qualitative")
    os.makedirs(out, exist_ok=True)
    ds = args.datasets[0]
    data = load_data(ds, args.seed, args.max_train, args.max_val, args.max_test)
    yt_raw = data["y_test"]  # normalized
    # train both
    print("=" * 60, "qualitative ICS-DGRN")
    pack, ypd_ics, row, esp, hmae, yt = train_ics(data, epochs=args.epochs, seed=args.seed)
    print("=" * 60, "qualitative AGCRN")
    model = build_model("AGCRN", data["n_nodes"], data["hist_len"], data["horizon"])
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=32,
        lr=1e-3,
        patience=max(args.epochs, 20),
        seed=args.seed,
        device="auto",
        loss="smooth_l1",
        weight_decay=1e-4,
    )
    pack2 = train_st_model(model, data["adj"], data["x_train"], data["y_train"], data["x_val"], data["y_val"], cfg)
    yp2, _ = predict_st_model(pack2, data["x_test"])
    ypd_ag = data["denorm"](yp2)

    # pick scenarios on node 0, horizon 1 series
    series_true = yt[:, 0, 0]
    # sliding windows of length 80
    win = 80
    diffs = np.abs(np.diff(series_true, prepend=series_true[0]))
    # Scenario A: low variation (normal)
    smooth = np.convolve(diffs, np.ones(win) / win, mode="valid")
    a_idx = int(np.argmin(smooth))
    # Scenario B: rapid rise
    rise = np.convolve(np.maximum(np.diff(series_true, prepend=series_true[0]), 0), np.ones(win) / win, mode="valid")
    b_idx = int(np.argmax(rise))
    # Scenario C: rapid fall
    fall = np.convolve(np.maximum(-np.diff(series_true, prepend=series_true[0]), 0), np.ones(win) / win, mode="valid")
    c_idx = int(np.argmax(fall))

    scenarios = {
        "A_normal": a_idx,
        "B_rapid_rise": b_idx,
        "C_rapid_fall": c_idx,
    }
    meta = {}
    viz.set_style()
    for name, start in scenarios.items():
        end = min(start + win, len(series_true))
        t = np.arange(start, end)
        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        ax.plot(t, series_true[start:end], label="Ground Truth", color="#1f4e79", lw=1.8)
        ax.plot(t, ypd_ics[start:end, 0, 0], label="ICS-DGRN", color="#c45911", lw=1.5)
        ax.plot(t, ypd_ag[start:end, 0, 0], label="AGCRN", color="#548235", lw=1.3, ls="--")
        ax.set_title(f"{ds} Scenario {name.replace('_', ' ')} | node0 | h=1")
        ax.set_xlabel("test time index")
        ax.set_ylabel("flow")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{ds}_scenario_{name}.png"), bbox_inches="tight")
        plt.close(fig)
        meta[name] = {"start": int(start), "end": int(end)}
    with open(os.path.join(out, f"{ds}_scenarios.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[qualitative] saved", out)

def write_master_report(args) -> None:
    """Write a compact metrics summary (does not overwrite the figure guide REPORT.md)."""
    lines = [
        "# ICS-DGRN mechanism experiments — metrics summary",
        "",
        f"- Protocol: hist=12, horizon=12, epochs={args.epochs}, SmoothL1, Adam+Cosine",
        f"- Subsample train/val/test = {args.max_train}/{args.max_val}/{args.max_test}",
        f"- Default ICS-DGRN: layer_dims={list(DEFAULT_LAYER_DIMS)}, compression_dims={list(DEFAULT_COMP)}, esp_target=0.9",
        "",
        "See `REPORT.md` in this folder for per-figure explanations.",
        "",
    ]

    def _append_csv(title: str, path: str, cols=None) -> None:
        if not os.path.exists(path):
            return
        df = pd.read_csv(path)
        if cols:
            cols = [c for c in cols if c in df.columns]
            df = df[cols]
        lines.extend(["## " + title, "", df.round(4).to_string(index=False), ""])

    _append_csv(
        "Compression ratio",
        os.path.join(RESULT_ROOT, "compression_ratio", "PEMS08_compression_metrics.csv"),
        ["compression_ratio", "compression_dim", "MAE", "RMSE", "R2", "TrainableParams", "TrainTimeSec"],
    )
    _append_csv(
        "Sparse density",
        os.path.join(RESULT_ROOT, "sparse_density", "PEMS08_density_metrics.csv"),
        ["rho", "MAE", "RMSE", "R2", "TrainableParams", "TrainTimeSec"],
    )
    _append_csv(
        "ESP state decay",
        os.path.join(RESULT_ROOT, "esp_state_decay", "PEMS08_esp_decay_summary.csv"),
    )
    _append_csv(
        "ESP–forecast joint",
        os.path.join(RESULT_ROOT, "esp_forecast_joint", "PEMS08_esp_forecast_table.csv"),
    )
    _append_csv(
        "Fixed vs trainable",
        os.path.join(RESULT_ROOT, "fixed_vs_trainable", "PEMS08_fixed_vs_trainable.csv"),
        ["variant", "MAE", "RMSE", "R2", "TrainableParams", "TotalParams", "TrainTimeSec", "PeakGPUMemMB"],
    )
    _append_csv(
        "Multi-seed",
        os.path.join(RESULT_ROOT, "multi_seed", "multi_seed_summary.csv"),
    )
    _append_csv(
        "Multi-horizon growth",
        os.path.join(RESULT_ROOT, "multi_horizon", "PEMS08_horizon_growth.csv"),
        ["model", "MAE", "MAE_h1", "MAE_h12", "DeltaMAE", "G_horizon"],
    )
    _append_csv(
        "Compute overhead",
        os.path.join(RESULT_ROOT, "compute_overhead", "PEMS08_compute_overhead.csv"),
        ["model", "MAE", "TrainableParams", "TotalParams", "TrainTimeSec", "PeakGPUMemMB"],
    )

    os.makedirs(RESULT_ROOT, exist_ok=True)
    out = os.path.join(RESULT_ROOT, "METRICS_SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote", out)


EXPERIMENT_FUNCS = {
    "esp_decay": run_esp_decay,
    "compression": run_compression,
    "density": run_density,
    "esp_forecast": run_esp_forecast,
    "fixed_vs_trainable": run_fixed_vs_trainable,
    "multi_seed": run_multi_seed,
    "multi_horizon": run_multi_horizon,
    "compute": run_compute,
    "qualitative": run_qualitative,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps",
        nargs="+",
        default=["all"],
        help="subset or 'all'. names: " + ", ".join(EXPERIMENT_FUNCS),
    )
    parser.add_argument("--datasets", nargs="+", default=["PEMS08"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--max-val", type=int, default=150)
    parser.add_argument("--max-test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--multiseed-datasets",
        nargs="+",
        default=["PEMS08", "PEMS04", "PEMS03"],
        help="datasets used only by multi_seed",
    )
    parser.add_argument("--with-agcrn", action="store_true", help="also multi-seed AGCRN on PEMS08")
    args = parser.parse_args()

    os.makedirs(RESULT_ROOT, exist_ok=True)
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    print("Results ->", RESULT_ROOT)

    if "all" in args.exps:
        ordered = [
            "esp_decay",
            "compression",
            "multi_seed",
            "density",
            "fixed_vs_trainable",
            "esp_forecast",
            "multi_horizon",
            "compute",
            "qualitative",
        ]
        args.with_agcrn = True
    else:
        ordered = args.exps

    for name in ordered:
        if name not in EXPERIMENT_FUNCS:
            raise ValueError(f"Unknown experiment {name}")
        print("\n" + "#" * 72)
        print("# Running", name)
        print("#" * 72)
        EXPERIMENT_FUNCS[name](args)
        write_master_report(args)

    write_master_report(args)
    print("Done. See", RESULT_ROOT)

if __name__ == "__main__":
    main()
