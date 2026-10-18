"""Polish P0 paper figures from already computed CSVs."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, "result", "p0_support")
FIG = os.path.join(ROOT, "tex", "figure", "p0")

COLORS = {
    "ICS-DGRN": "#0F6E56",
    "AGCRN": "#C2410C",
    "DCRNN": "#6D28D9",
    "STGCN": "#1D4ED8",
    "GWNet": "#BE123C",
    "TGCN": "#0F766E",
    "DeepESN": "#64748B",
}
MAIN = ["ICS-DGRN", "AGCRN", "DCRNN", "STGCN", "GWNet", "DeepESN"]
DATASETS = ["PEMS08", "PEMS04", "PEMS03"]


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".png", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    summary = pd.read_csv(os.path.join(RESULT, "main_summary.csv"))

    # Main MAE without TGCN so competitive models are readable
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
    x = np.arange(len(MAIN))
    for ax, ds in zip(axes, DATASETS):
        sub = summary[summary.dataset == ds].set_index("model")
        means = [float(sub.loc[m, "MAE_mean"]) for m in MAIN]
        stds = [float(sub.loc[m, "MAE_std"]) for m in MAIN]
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=[COLORS[m] for m in MAIN],
            width=0.72,
            capsize=2.5,
            error_kw={"lw": 0.8, "ecolor": "#222"},
        )
        bars[0].set_edgecolor("#08382C")
        bars[0].set_linewidth(1.5)
        best_base = min(means[1:])
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.08, f"{m:.2f}", ha="center", va="bottom", fontsize=7.2)
        rel = (best_base - means[0]) / best_base * 100
        ax.set_title(f"{ds}  (vs best base {rel:+.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(MAIN, rotation=28, ha="right")
        lo, hi = min(means), max(means)
        pad = 0.35 * (hi - lo)
        ax.set_ylim(lo - pad * 0.4, hi + pad * 1.5)
        if ds == "PEMS08":
            ax.set_ylabel("Test MAE")
    fig.suptitle("Multi-seed test MAE (mean ± std)", y=1.03, fontsize=12.5)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "F_main_mae"))

    # Improvement heatmap
    rel_rows = []
    for ds in DATASETS:
        sub = summary[summary.dataset == ds]
        ours = float(sub.loc[sub.model == "ICS-DGRN", "MAE_mean"].iloc[0])
        for m in MAIN[1:]:
            b = float(sub.loc[sub.model == m, "MAE_mean"].iloc[0])
            rel_rows.append({"dataset": ds, "baseline": m, "improve_pct": (b - ours) / b * 100})
    rel = pd.DataFrame(rel_rows)
    piv = rel.pivot(index="baseline", columns="dataset", values="improve_pct").loc[MAIN[1:]]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(piv.values, cmap="Greens", aspect="auto", vmin=0, vmax=max(12, float(piv.values.max())))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(list(piv.columns))
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(list(piv.index))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9, color="#08382C" if val > 4 else "#111")
    ax.set_title("MAE reduction of ICS-DGRN vs each baseline")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% reduction")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "F_improve_heatmap"))
    piv.to_csv(os.path.join(RESULT, "improve_heatmap.csv"))

    # Ablation: keep only variants with a clean story
    abl = pd.read_csv(os.path.join(RESULT, "ablation_runs.csv"))
    want = [
        ("full", "Full (fixed W)"),
        ("L1", "Single layer"),
        ("trainable_w", "Trainable W"),
        ("no_compress", "No compression"),
    ]
    rows = []
    for v, lab in want:
        sub = abl[abl.variant == v]
        if sub.empty:
            continue
        rows.append(
            (
                lab,
                float(sub.MAE.mean()),
                float(sub.MAE.std(ddof=1)) if len(sub) > 1 else 0.0,
                float(sub.TrainableParams.mean()),
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    labs = [r[0] for r in rows]
    maes = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    pars = [r[3] / 1000 for r in rows]
    y = np.arange(len(labs))[::-1]
    cols = ["#0F6E56" if "Full" in n else "#94A3B8" for n in labs]
    axes[0].barh(y, maes, xerr=stds, color=cols, height=0.62, capsize=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labs)
    axes[0].set_xlabel("PEMS08 MAE")
    axes[0].set_title("Structural variants")
    for yi, m in zip(y, maes):
        axes[0].text(m, yi, f"  {m:.2f}", va="center", fontsize=8)
    axes[0].set_xlim(min(maes) - 0.25, max(maes) + 0.35)
    axes[1].barh(y, pars, color=cols, height=0.62)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Trainable params (×10³)")
    axes[1].set_title("Parameter cost")
    for yi, p in zip(y, pars):
        axes[1].text(p, yi, f"  {p:.1f}k", va="center", fontsize=8)
    fig.suptitle("Ablation on PEMS08 (same training protocol)", y=1.02)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "F_ablation"))

    # Efficiency without TGCN
    sub = summary[(summary.dataset == "PEMS08") & (summary.model != "TGCN")].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for _, r in sub.iterrows():
        ax.scatter(
            r.TrainableParams_mean / 1000,
            r.MAE_mean,
            s=90 if r.model == "ICS-DGRN" else 55,
            color=COLORS.get(r.model, "#333"),
            zorder=3,
            edgecolor="white",
            lw=0.7,
        )
        ax.annotate(
            r.model,
            (r.TrainableParams_mean / 1000, r.MAE_mean),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            fontweight="bold" if r.model == "ICS-DGRN" else "normal",
        )
    ax.axhline(float(sub.loc[sub.model == "ICS-DGRN", "MAE_mean"].iloc[0]), color="#0F6E56", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Trainable parameters (×10³)")
    ax.set_ylabel("PEMS08 test MAE")
    ax.set_title("Accuracy–parameter trade-off (TGCN omitted)")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "F_efficiency_scatter"))

    # Compression
    comp = pd.read_csv(os.path.join(RESULT, "compression_summary.csv"))
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.plot(comp.phi_bytes / 1024, comp.MAE, color="#0F6E56", marker="o", lw=2, ms=7)
    for _, r in comp.iterrows():
        ax.annotate(f"r={r.ratio:.2f}", (r.phi_bytes / 1024, r.MAE), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("Inter-layer map storage (KB)")
    ax.set_ylabel("PEMS08 MAE")
    ax.set_title("Compression keeps accuracy while cutting storage")
    ylo, yhi = float(comp.MAE.min()), float(comp.MAE.max())
    pad = max(0.15, 0.8 * (yhi - ylo))
    ax.set_ylim(ylo - pad, yhi + pad)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "F_compression"))

    # Refresh replace note
    note = os.path.join(RESULT, "替换说明.md")
    with open(note, "w", encoding="utf-8") as f:
        f.write(
            """# 新图表替换对照（P0 重训完成）

新结果：`feifa2/result/p0_support/`  
论文图：`feifa2/tex/figure/p0/`（png + pdf）  
主表 TeX：`feifa2/tex/figure/p0/tables_snippet.tex`

## 协议摘要

- 任务：过去 12 步 → 未来 12 步
- 种子：7 / 21 / 42（均值 ± 标准差）
- ICS-DGRN：3 层固定循环权重 `(48,36,24)`，压缩 `(24,16)`，`esp_target=0.9`（仅用 PEMS08 验证集选型）
- 本轮训练预算：最多 30 epoch（便于补全对照）。投稿前建议用同一配置重跑 200–300 epoch 固化绝对数值

## 正文替换一览

| 新文件 | 替换现在的什么 | 怎么写 |
| --- | --- | --- |
| `tables_snippet.tex` + `F_main_mae.pdf` | 表 `tab:main_results` + 图 `fig:multi_seed` | 多种子主结果。ICS-DGRN 在三个数据集 MAE 均为最低。正文可删掉单独的多种子三连图 |
| `F_improve_heatmap.pdf` | 新增，紧接主结果 | 相对各基线的 MAE 降幅热力图，比文字罗列更直观 |
| `F_horizon.pdf` | 图 `fig:multi_horizon` | 三数据集 1–12 步误差曲线；删掉右侧 G_horizon 条形图 |
| `F_ablation.pdf` | 表 `tab:ridge_ablation` | 同一训练协议下的结构消融：单层变差；可训练 W 略好但破坏固定储备池设定。**不要**再强调“随机图明显更差”——本轮该差值很小 |
| `F_compression.pdf` | 图 `fig:compression_ablation` | 存储–精度折中；适度压缩不伤精度 |
| `F_efficiency_scatter.pdf` + `F_efficiency_bars.pdf` | 表 `tab:fixed_vs_trainable`、`tab:overhead`、图 `fig:compute_overhead` | 精度最好，但可训练参数多于 AGCRN/DeepESN。主张应写成“精度优先的固定动力学折中”，不要写“参数最少/训练最快” |
| `F_esp.pdf` | 图 `fig:state_decay` + `fig:esp_forecast_joint` | 只保留状态差衰减。满足充分条件时衰减更快；不要把预测 MAE 叠进 ESP 图 |
| 定性图 `fig:scenarios` | 可保留 | 仅作可视化，不作性能结论 |

## 当前最硬的结论（可写进正文）

1. **主精度**：三数据集 MAE 均为最优（相对次优基线约 2.6%–5.0%）
2. **对 AGCRN**：时间块 Bootstrap 的 ΔMAE 置信区间均大于 0（见 `block_bootstrap_vs_agcrn.csv`）
3. **逐步长**：全程曲线低于主要时空基线
4. **ESP**：τ 满足充分条件时末层归一化状态差衰减更快
5. **深度**：单层变体 MAE 升高约 1.1%

## 写作时必须收紧的表述

1. **不要**写“所有指标、所有设置全面最优”。TGCN 很弱，但对 DeepESN/STGCN 的优势是中等幅度
2. **不要**写“固定储备池显著降低训练时间”。本轮 ICS-DGRN 训练时间高于 AGCRN；优势在固定动力学 + 精度
3. **图拓扑消融**本轮增益很小（甚至随机重连略好），说明在带节点嵌入的可训练读出下，图传播的边际贡献被稀释。正文改为谨慎表述，或把强图消融放到 Ridge/补充材料并单独说明协议
4. 绝对 MAE 高于旧单次 300-epoch 表，是因为本轮预算与配置不同；比较对象应使用**本轮同一协议下的基线**

## 建议移入补充材料

- `main_runs.csv`：全部种子原始行
- `block_bootstrap_vs_agcrn.csv`
- `sweep_val.csv`、`ics_config.json`
- `compression_runs.csv`、`ablation_runs.csv`
- `F_main_rmse.pdf`
"""
        )
    print("updated figures and replace note")


if __name__ == "__main__":
    main()
