# Copyright 2026 ZengWenquan
# https://github.com/chaosbull
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.unicode_minus"] = False


def plot_prediction_curve(y_true, y_pred, out_path, title, max_points: int = 400):
    set_style()
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    n = min(len(yt), max_points)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(yt[:n], label="true", lw=1.4, color="#1f4e79")
    ax.plot(yp[:n], label="pred", lw=1.2, color="#c45911")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(y_true, y_pred, out_path, title, max_points: int = 3000):
    set_style()
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if len(yt) > max_points:
        idx = np.random.default_rng(0).choice(len(yt), max_points, replace=False)
        yt, yp = yt[idx], yp[idx]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.scatter(yt, yp, s=8, alpha=0.35, c="#2e75b6", linewidths=0)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#c00000", lw=1)
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_hist(y_true, y_pred, out_path, title):
    set_style()
    err = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(err, bins=40, color="#548235", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("pred - true")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(results: Dict[str, Dict[str, float]], metric: str, out_path: str, title: str):
    set_style()
    names = list(results.keys())
    vals = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.7 * len(names) + 1.5), 3.6))
    colors = sns.color_palette("muted", n_colors=len(names))
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_path: str, title: str):
    set_style()
    fig, ax = plt.subplots(figsize=(1.15 * df.shape[1] + 2.5, 0.42 * df.shape[0] + 1.6))
    sns.heatmap(df, annot=True, fmt=".3g", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_curves(horizon_metrics: Dict[str, List[float]], out_path: str, title: str):
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, vals in horizon_metrics.items():
        ax.plot(range(1, len(vals) + 1), vals, marker="o", ms=3.5, label=name, lw=1.3)
    ax.set_xlabel("horizon")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, out_csv: str, out_xlsx: Optional[str] = None) -> None:
    folder = os.path.dirname(out_csv)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df.to_csv(out_csv, index=True, float_format="%.6g")
    if not out_xlsx:
        return
    try:
        df.to_excel(out_xlsx)
    except Exception as exc:
        print(f"skip xlsx ({exc})")
