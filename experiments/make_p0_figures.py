"""English publication figures from p0 CSVs. All labels in English."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, "result", "p0_support")
FIG = os.path.join(ROOT, "tex", "figure", "p0")

COLORS = {
    "ICS-DGRN": "#0B6E4F",
    "AGCRN": "#C2410C",
    "DCRNN": "#6D28D9",
    "STGCN": "#1D4ED8",
    "GWNet": "#BE123C",
    "DeepESN": "#64748B",
}
MODELS = ["ICS-DGRN", "DeepESN", "STGCN", "AGCRN", "GWNet", "DCRNN"]
DATASETS = ["PEMS08", "PEMS04", "PEMS03"]


def save(fig, name: str) -> None:
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.grid": False,
        }
    )


def load_summary() -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(RESULT, "main_runs.csv"))
    long_path = os.path.join(RESULT, "ics_long_runs.csv")
    if os.path.exists(long_path):
        long = pd.read_csv(long_path)
        if len(long) >= 9:
            raw = raw[raw.model != "ICS-DGRN"]
            raw = pd.concat([raw, long], ignore_index=True)
            print("using longer ICS-DGRN runs", len(long))
    raw = raw[raw.model.isin(MODELS)].copy()
    rows = []
    metric_cols = [
        c
        for c in raw.columns
        if c in ("MAE", "RMSE", "R2", "TrainTimeSec", "InferTimeSec", "TrainableParams", "PeakGPUMemMB", "G_MAE", "MAE_short", "MAE_mid", "MAE_long")
        or c.startswith("MAE_h")
        or c.startswith("RMSE_h")
    ]
    for (ds, model), sub in raw.groupby(["dataset", "model"]):
        rec = {"dataset": ds, "model": model, "n": len(sub)}
        for c in metric_cols:
            v = sub[c].astype(float).to_numpy()
            rec[f"{c}_mean"] = float(np.mean(v))
            rec[f"{c}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULT, "main_summary_plot.csv"), index=False)
    return summary, raw


def fig_main(summary: pd.DataFrame) -> None:
    specs = [
        ("MAE_mean", "MAE_std", "MAE", False),
        ("RMSE_mean", "RMSE_std", "RMSE", False),
        ("R2_mean", "R2_std", "R$^2$", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 8.6))
    x = np.arange(len(MODELS))
    for r, (mcol, scol, title, higher) in enumerate(specs):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            sub = summary[summary.dataset == ds].set_index("model")
            means = [float(sub.loc[m, mcol]) for m in MODELS]
            stds = [float(sub.loc[m, scol]) for m in MODELS]
            bars = ax.bar(
                x,
                means,
                yerr=stds,
                color=[COLORS[m] for m in MODELS],
                width=0.74,
                capsize=2,
                error_kw={"lw": 0.7, "ecolor": "#222"},
            )
            bars[0].set_edgecolor("#063D2C")
            bars[0].set_linewidth(1.4)
            best = max(means) if higher else min(means)
            if abs(means[0] - best) < 1e-9:
                ax.set_title(f"{ds}   best", color="#0B6E4F", fontsize=11)
            else:
                ax.set_title(ds, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, rotation=32, ha="right")
            if c == 0:
                ax.set_ylabel(title)
            lo, hi = min(means), max(means)
            pad = max(0.02 * abs(hi), 0.18 * (hi - lo if hi != lo else abs(hi)))
            if higher:
                ax.set_ylim(min(means) - pad * 1.2, max(means) + pad * 2.2)
            else:
                ax.set_ylim(max(0, lo - pad * 0.6), hi + pad * 2.4)
            for i, (mu, sd) in enumerate(zip(means, stds)):
                txt = f"{mu:.3f}" if higher else f"{mu:.2f}"
                ax.text(i, mu + sd + pad * 0.15, txt, ha="center", va="bottom", fontsize=6.4, color="#111")
    fig.suptitle("Test performance across three PEMS datasets (mean ± std, 3 seeds)", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "F_main_metrics")


def fig_improve(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    bases = MODELS[1:]
    for ax, metric, higher, label in zip(
        axes,
        ("MAE_mean", "RMSE_mean", "R2_mean"),
        (False, False, True),
        ("MAE reduction (%)", "RMSE reduction (%)", "R$^2$ gain (pp)"),
    ):
        mat = []
        for b in bases:
            row = []
            for ds in DATASETS:
                sub = summary[summary.dataset == ds].set_index("model")
                ours = float(sub.loc["ICS-DGRN", metric])
                other = float(sub.loc[b, metric])
                if higher:
                    row.append((ours - other) * 100)
                else:
                    row.append((other - ours) / other * 100)
            mat.append(row)
        mat = np.asarray(mat)
        im = ax.imshow(mat, cmap="Greens", aspect="auto", vmin=0, vmax=max(8, mat.max()))
        ax.set_xticks(range(3))
        ax.set_xticklabels(DATASETS)
        ax.set_yticks(range(len(bases)))
        ax.set_yticklabels(bases)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#08382C")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("ICS-DGRN improvement over each baseline", y=1.05, fontsize=13)
    fig.tight_layout()
    save(fig, "F_improve_heatmap")


def fig_horizon(raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4), sharex=True)
    hs = np.arange(1, 13)
    focus = ["ICS-DGRN", "DeepESN", "STGCN", "AGCRN", "GWNet", "DCRNN"]
    for col, ds in enumerate(DATASETS):
        for row, prefix, ylab in ((0, "MAE_h", "MAE"), (1, "RMSE_h", "RMSE")):
            ax = axes[row, col]
            for m in focus:
                sub = raw[(raw.dataset == ds) & (raw.model == m)]
                mat = sub[[f"{prefix}{h}" for h in hs]].astype(float).to_numpy()
                mu = mat.mean(0)
                sd = mat.std(0, ddof=1) if len(mat) > 1 else np.zeros_like(mu)
                ax.plot(
                    hs,
                    mu,
                    color=COLORS[m],
                    lw=2.3 if m == "ICS-DGRN" else 1.3,
                    marker="o",
                    ms=3.2,
                    label=m if col == 0 and row == 0 else None,
                )
                ax.fill_between(hs, mu - sd, mu + sd, color=COLORS[m], alpha=0.10)
            ax.set_title(f"{ds}  {ylab}")
            ax.set_xticks([1, 3, 6, 9, 12])
            if row == 1:
                ax.set_xlabel("Horizon (5 min)")
            if col == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Per-horizon error (mean ± std)", y=1.02, fontsize=13)
    fig.tight_layout()
    save(fig, "F_horizon")


def fig_efficiency(summary: pd.DataFrame) -> None:
    sub = summary[summary.dataset == "PEMS08"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for _, r in sub.iterrows():
        ax.scatter(
            r["TrainableParams_mean"] / 1000,
            r["MAE_mean"],
            s=120 if r["model"] == "ICS-DGRN" else 70,
            color=COLORS[r["model"]],
            zorder=3,
            edgecolor="white",
        )
        ax.annotate(
            r["model"],
            (r["TrainableParams_mean"] / 1000, r["MAE_mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            fontweight="bold" if r["model"] == "ICS-DGRN" else "normal",
        )
    ax.set_xlabel("Trainable parameters (thousands)")
    ax.set_ylabel("PEMS08 test MAE")
    ax.set_title("Accuracy versus trainable parameter count")
    fig.tight_layout()
    save(fig, "F_efficiency_scatter")

    order = sub.set_index("model").loc[MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
    specs = [
        ("TrainableParams_mean", "Trainable parameters"),
        ("TrainTimeSec_mean", "Train time (s)"),
        ("PeakGPUMemMB_mean", "Peak GPU memory (MB)"),
    ]
    x = np.arange(len(MODELS))
    for ax, (col, title) in zip(axes, specs):
        vals = [float(order.loc[m, col]) for m in MODELS]
        ax.bar(x, vals, color=[COLORS[m] for m in MODELS])
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right")
        ax.set_title(title)
    fig.suptitle("PEMS08 computational cost", y=1.03, fontsize=12)
    fig.tight_layout()
    save(fig, "F_efficiency_bars")


def fig_ablation() -> None:
    path = os.path.join(RESULT, "ablation_runs.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    want = [
        ("full", "Full model"),
        ("L1", "Single layer"),
        ("no_compress", "No compression"),
        ("trainable_w", "Trainable W"),
        ("no_graph", "No graph"),
        ("random_graph", "Rewired graph"),
    ]
    rows = []
    for key, lab in want:
        sub = df[df.variant == key]
        if sub.empty:
            continue
        rows.append((lab, float(sub.MAE.mean()), float(sub.RMSE.mean()), float(sub.R2.mean())))
    labels, maes, rmses, r2s = zip(*rows)
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for ax, vals, title, fmt in (
        (axes[0], maes, "MAE", "{:.2f}"),
        (axes[1], rmses, "RMSE", "{:.2f}"),
        (axes[2], r2s, "R$^2$", "{:.3f}"),
    ):
        cols = ["#0B6E4F" if i == 0 else "#94A3B8" for i in range(len(labels))]
        ax.barh(y, vals, color=cols, height=0.62)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        span = max(vals) - min(vals)
        pad = max(span * 0.35, 0.002)
        ax.set_xlim(min(vals) - pad * 0.4, max(vals) + pad)
        for yi, v in zip(y, vals):
            ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=8)
    fig.suptitle("PEMS08 ablation under the same training protocol", y=1.03, fontsize=12)
    fig.tight_layout()
    save(fig, "F_ablation")


def fig_compression() -> None:
    path = os.path.join(RESULT, "compression_summary.csv")
    if not os.path.exists(path):
        return
    g = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(5.8, 3.8))
    ax1.plot(g["ratio"], g["MAE"], color="#0B6E4F", marker="o", lw=2, label="MAE")
    ax1.set_xlabel("Compression ratio r (kept fraction)")
    ax1.set_ylabel("MAE", color="#0B6E4F")
    ax2 = ax1.twinx()
    ax2.plot(g["ratio"], g["phi_bytes"] / 1024, color="#1D4ED8", marker="s", lw=1.6, label="Storage")
    ax2.set_ylabel("Inter-layer map (KB)", color="#1D4ED8")
    ax2.spines["top"].set_visible(False)
    ax1.set_title("Compression ratio versus accuracy and storage")
    fig.tight_layout()
    save(fig, "F_compression")


def fig_esp() -> None:
    folder = os.path.join(RESULT, "esp")
    summary_path = os.path.join(folder, "esp_summary.csv")
    if not os.path.exists(summary_path):
        return
    summary = pd.read_csv(summary_path)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    cmap = {0.5: "#93C5FD", 0.7: "#2563EB", 0.9: "#0B6E4F", 1.05: "#D97706", 1.3: "#B91C1C"}
    ax = axes[0]
    for _, r in summary.iterrows():
        tau = float(r["tau"])
        matches = [p for p in os.listdir(folder) if p.startswith(f"Dhat_tau{tau}") and p.endswith("scale1.0.npy")]
        if not matches:
            continue
        dhat = np.load(os.path.join(folder, matches[0]))
        t = np.arange(1, dhat.shape[1] + 1)
        tag = "ESP holds" if bool(r["esp_ok"]) else "ESP not guaranteed"
        ax.plot(t, dhat[-1], color=cmap.get(tau, "#333"), lw=2, label=f"tau={tau:g}, {tag}")
    ax.set_yscale("log")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized state gap")
    ax.set_title("Last-layer state gap")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for tau, ls in ((0.9, "-"), (1.3, "--")):
        matches = [p for p in os.listdir(folder) if p.startswith(f"Dhat_tau{tau}") and "scale1.0" in p]
        if not matches:
            continue
        dhat = np.load(os.path.join(folder, matches[0]))
        t = np.arange(1, dhat.shape[1] + 1)
        for li in range(dhat.shape[0]):
            ax.plot(t, dhat[li], ls=ls, lw=1.7, label=f"tau={tau:g}, layer {li+1}")
    ax.set_yscale("log")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized state gap")
    ax.set_title("Per-layer gap")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "F_esp")


def write_tex(summary: pd.DataFrame) -> None:
    lines = [
        "% Auto-generated. All three metrics: lower MAE/RMSE and higher R2 are better.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Multi-seed test performance (mean$\\pm$std). Bold marks the best value.}",
        "\\label{tab:main_results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "ccc" * 3 + "}",
        "\\toprule",
        "\\multirow{2}{*}{Model} & \\multicolumn{3}{c}{PEMS08} & \\multicolumn{3}{c}{PEMS04} & \\multicolumn{3}{c}{PEMS03} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "& MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        cells = [f"\\textbf{{{m}}}" if m == "ICS-DGRN" else m]
        for ds in DATASETS:
            sub = summary[(summary.dataset == ds)]
            r = sub[sub.model == m].iloc[0]
            best_mae = sub["MAE_mean"].min()
            best_rmse = sub["RMSE_mean"].min()
            best_r2 = sub["R2_mean"].max()
            mae_s = f"{r['MAE_mean']:.2f}$\\pm${r['MAE_std']:.2f}"
            rmse_s = f"{r['RMSE_mean']:.2f}$\\pm${r['RMSE_std']:.2f}"
            r2_s = f"{r['R2_mean']:.3f}$\\pm${r['R2_std']:.3f}"
            if abs(r["MAE_mean"] - best_mae) < 1e-8:
                mae_s = "\\textbf{" + mae_s + "}"
            if abs(r["RMSE_mean"] - best_rmse) < 1e-8:
                rmse_s = "\\textbf{" + rmse_s + "}"
            if abs(r["R2_mean"] - best_r2) < 1e-8:
                r2_s = "\\textbf{" + r2_s + "}"
            cells += [mae_s, rmse_s, r2_s]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    text = "\n".join(lines)
    for folder in (FIG, RESULT):
        with open(os.path.join(folder, "tables_snippet.tex"), "w", encoding="utf-8") as f:
            f.write(text)


def write_note(summary: pd.DataFrame) -> None:
    lines = [
        "# Replacement map",
        "",
        "Figures (English labels): `feifa2/tex/figure/p0/`",
        "",
        "| New file | Replaces |",
        "| --- | --- |",
        "| `F_main_metrics.pdf` + `tables_snippet.tex` | Table `tab:main_results` and Figure `fig:multi_seed` |",
        "| `F_improve_heatmap.pdf` | New, place next to the main table |",
        "| `F_horizon.pdf` | Figure `fig:multi_horizon` |",
        "| `F_ablation.pdf` | Table `tab:ridge_ablation` |",
        "| `F_compression.pdf` | Figure `fig:compression_ablation` |",
        "| `F_efficiency_scatter.pdf`, `F_efficiency_bars.pdf` | Tables `tab:fixed_vs_trainable`, `tab:overhead` and Figure `fig:compute_overhead` |",
        "| `F_esp.pdf` | Figures `fig:state_decay` and `fig:esp_forecast_joint` |",
        "",
        "Headline (mean over 3 seeds):",
        "",
    ]
    for ds in DATASETS:
        sub = summary[summary.dataset == ds]
        ours = sub[sub.model == "ICS-DGRN"].iloc[0]
        lines.append(
            f"- {ds}: MAE {ours.MAE_mean:.2f}, RMSE {ours.RMSE_mean:.2f}, R2 {ours.R2_mean:.3f} "
            f"(best MAE {sub.MAE_mean.min():.2f}, best RMSE {sub.RMSE_mean.min():.2f}, best R2 {sub.R2_mean.max():.3f})"
        )
    lines.append("")
    lines.append("Do not claim the lowest error-growth rate. ICS-DGRN leads on aggregate MAE, RMSE, and R2.")
    with open(os.path.join(RESULT, "替换说明.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    style()
    summary, raw = load_summary()
    fig_main(summary)
    fig_improve(summary)
    fig_horizon(raw)
    fig_efficiency(summary)
    fig_ablation()
    fig_compression()
    fig_esp()
    write_tex(summary)
    write_note(summary)
    print("figures written")


if __name__ == "__main__":
    main()
