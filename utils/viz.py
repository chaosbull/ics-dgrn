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

"""Visualization helpers (paper-style figures)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.unicode_minus"] = False

def plot_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "Prediction vs Ground Truth",
    max_points: int = 500,
    labels: Optional[Sequence[str]] = None,
) -> None:
    set_style()
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    n = min(len(yt), max_points)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(yt[:n], label=labels[0] if labels else "True", lw=1.6, color="#1f4e79")
    ax.plot(yp[:n], label=labels[1] if labels else "Pred", lw=1.4, color="#c45911", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "Scatter",
    max_points: int = 3000,
) -> None:
    set_style()
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if len(yt) > max_points:
        idx = np.random.default_rng(0).choice(len(yt), max_points, replace=False)
        yt, yp = yt[idx], yp[idx]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(yt, yp, s=8, alpha=0.35, c="#2e75b6")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "--", color="#c00000", lw=1.2)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_error_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "Error distribution",
) -> None:
    set_style()
    err = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(err, bins=50, color="#548235", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_metric_bars(
    results: Dict[str, Dict[str, float]],
    metric: str,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    set_style()
    names = list(results.keys())
    vals = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names) + 2), 4))
    colors = sns.color_palette("muted", n_colors=len(names))
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} comparison")
    ax.tick_params(axis="x", rotation=25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_radar(
    results: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
    out_path: str,
    title: str = "Multi-metric radar",
) -> None:
    set_style()
    # Invert error metrics so larger is better on radar
    inv = {"MSE", "MAE", "RMSE", "MAPE"}
    models = list(results.keys())
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for name in models:
        vals = []
        for m in metrics:
            v = results[name][m]
            vals.append(1.0 / (v + 1e-8) if m in inv else v)
        # normalize per metric across models later — local normalize
        arr = np.array(vals, dtype=float)
        vals = (arr / (arr.max() + 1e-12)).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=name, lw=1.5)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(metrics))
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_heatmap_metrics(
    df: pd.DataFrame,
    out_path: str,
    title: str = "Metrics heatmap",
) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(1.2 * df.shape[1] + 3, 0.45 * df.shape[0] + 2))
    sns.heatmap(df, annot=True, fmt=".4g", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_adjacency(
    adj: np.ndarray,
    out_path: str,
    title: str = "Graph adjacency",
    max_n: int = 80,
) -> None:
    set_style()
    a = adj
    if a.shape[0] > max_n:
        a = a[:max_n, :max_n]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(a, cmap="mako", cbar=True, ax=ax, xticklabels=False, yticklabels=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_esp_decay(
    curves: Dict[str, np.ndarray],
    out_path: str,
    title: str = "ESP state-difference decay",
) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in curves.items():
        ax.semilogy(c + 1e-15, label=name, lw=1.6)
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$\|\Delta X(t)\|_F$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_horizon_curves(
    horizon_metrics: Dict[str, List[float]],
    out_path: str,
    metric_name: str = "MAE",
    title: Optional[str] = None,
) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in horizon_metrics.items():
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=name, lw=1.5)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} vs horizon")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_table(df: pd.DataFrame, out_csv: str, out_xlsx: Optional[str] = None) -> None:
    _ensure_dir(os.path.dirname(out_csv) or ".")
    df.to_csv(out_csv, index=True, float_format="%.6g")
    if out_xlsx:
        try:
            df.to_excel(out_xlsx)
        except Exception:
            pass

def plot_ablation(
    names: Sequence[str],
    values: Sequence[float],
    out_path: str,
    ylabel: str = "MAE",
    title: str = "Ablation study",
) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color=sns.color_palette("Set2", n_colors=len(names)), edgecolor="black", lw=0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
