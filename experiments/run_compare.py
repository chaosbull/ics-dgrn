# Copyright 2026 ZengWenquan
# https://github.com/chaosbull
# SPDX-License-Identifier: Apache-2.0

"""Compare ICS-DGRN with STGCN, T-GCN, Graph WaveNet and AGCRN on PEMS.

Epochs are capped at 1000. The csv files already in results/ are from
an earlier 20-epoch run (train 800 / val 150 / test 250, seed 42).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.st_models import MAX_EPOCHS, TrainConfig, build_model, predict, train_model
from data.pems import load_pems
from utils import viz
from utils.metrics import evaluate_all, mae

RESULT_ROOT = os.path.join(ROOT, "results")

# lower is better, except R2 and the ESP flag
SCORE_DIMS = [
    ("MAE", "low", "acc"),
    ("RMSE", "low", "acc"),
    ("MAPE", "low", "acc"),
    ("R2", "high", "acc"),
    ("InferTimeSec", "low", "eff"),
    ("TrainableParams", "low", "eff"),
    ("ModelSizeMB", "low", "eff"),
    ("ParamEff", "low", "eff"),
    ("MAE_TrainCost", "low", "eff"),
    ("ESP_Guarantee", "high", "theory"),
]

MODELS = ["ICS-DGRN", "STGCN", "TGCN", "GWNet", "AGCRN"]


def plot_loss_curves(histories, out_dir, ds, loss_name):
    viz.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, hist in histories.items():
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=f"{name} train", lw=1.2)
        axes[0].plot(ep, [h["val_loss"] for h in hist], label=f"{name} val", lw=1.2, ls="--")
        axes[1].plot(ep, [h["train_mae"] for h in hist], label=f"{name} train", lw=1.2)
        axes[1].plot(ep, [h["val_mae"] for h in hist], label=f"{name} val", lw=1.2, ls="--")
    axes[0].set_title(f"{ds} loss ({loss_name})")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title(f"{ds} MAE, normalized")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{ds}_loss_mae_curves.png"), bbox_inches="tight")
    plt.close(fig)

    for name, hist in histories.items():
        if not hist:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color="#1f4e79", label="train")
        axes[0].plot(ep, [h["val_loss"] for h in hist], color="#c45911", label="val")
        axes[0].set_title(f"{ds} {name} loss")
        axes[0].legend()
        axes[1].plot(ep, [h["train_mae"] for h in hist], color="#1f4e79", label="train")
        axes[1].plot(ep, [h["val_mae"] for h in hist], color="#c45911", label="val")
        axes[1].set_title(f"{ds} {name} MAE")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{ds}_{name}_curves.png"), bbox_inches="tight")
        plt.close(fig)


def scorecard(results, proposed="ICS-DGRN") -> pd.DataFrame:
    # top-2, or within 5% of the best, counts as "excellent" on accuracy.
    # efficiency just takes the top 3. This is only for the bar chart.
    rows = []
    for dim, direction, kind in SCORE_DIMS:
        vals = {m: float(results[m][dim]) for m in results}
        items = sorted(vals.items(), key=lambda kv: kv[1], reverse=(direction == "high"))
        rank = {m: i + 1 for i, (m, _) in enumerate(items)}
        best = items[0][1]
        if kind == "acc":
            if direction == "low":
                excellent = [m for m, v in vals.items() if rank[m] <= 2 or v <= best * 1.05]
            else:
                excellent = [m for m, v in vals.items() if rank[m] <= 2 or v >= best * 0.95]
        elif kind == "theory":
            excellent = [m for m, v in vals.items() if abs(v - best) < 1e-12]
        else:
            excellent = [m for m, r in rank.items() if r <= 3]
        rows.append(
            {
                "Dimension": dim,
                "Kind": kind,
                "BestValue": best,
                "ICS-DGRN": vals.get(proposed, np.nan),
                "ICS-DGRN_Rank": rank.get(proposed, -1),
                "ICS-DGRN_Excellent": proposed in excellent,
                "ExcellentModels": ", ".join(excellent),
            }
        )
    return pd.DataFrame(rows)


def run_one(name, data, epochs, seed, loss):
    model = build_model(name, data["n_nodes"], data["hist_len"], data["horizon"])
    lr = 8e-4 if name == "ICS-DGRN" else 1e-3
    epochs = min(max(int(epochs), 1), MAX_EPOCHS)
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=32,
        lr=lr,
        patience=epochs,
        seed=seed,
        device="auto",
        loss=loss,
        weight_decay=1e-4,
    )
    print(f"  {name}, up to {epochs} epochs, lr={lr}")
    pack = train_model(
        model,
        data["adj"],
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        cfg,
    )
    yp, infer_sec = predict(pack, data["x_test"])
    for h in pack["history"]:
        h["loss"] = pack.get("loss_name", loss)
    return pack, yp, infer_sec


def run_dataset(name, epochs, seed, max_train, max_val, max_test):
    out_dir = os.path.join(RESULT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    data = load_pems(name, max_train=max_train, max_val=max_val, max_test=max_test, seed=seed)
    denorm = data["denorm"]
    yt = denorm(data["y_test"])

    results = {}
    preds = {}
    histories = {}
    loss = "smooth_l1"

    for m in MODELS:
        print("=" * 60, name, m)
        pack, yp, infer_sec = run_one(m, data, epochs, seed, loss)
        ypd = denorm(yp)
        acc = evaluate_all(yt, ypd)
        params = float(pack["params"])
        row = {
            **acc,
            "TrainTimeSec": pack["train_sec"],
            "InferTimeSec": infer_sec,
            "TrainableParams": params,
            "StoredParams": params,
            "ModelSizeMB": params * 4 / (1024**2),
            "Epochs": float(pack["epochs_ran"]),
            "best_val_mae": pack["best_val_mae"],
            "MAE_TrainCost": float(acc["MAE"] * pack["train_sec"]),
            "ParamEff": float(acc["MAE"] * (params ** 0.5)),
            "ESP_Guarantee": 0.0,
        }
        if m == "ICS-DGRN":
            try:
                esp = pack["model"].esp_report(pack["adj_t"])
                row["ESP_Guarantee"] = 1.0 if esp.get("esp_ok") else 0.0
                with open(os.path.join(out_dir, f"{name}_esp.json"), "w", encoding="utf-8") as f:
                    json.dump(esp, f, indent=2)
            except Exception as exc:
                print("esp report failed:", exc)
        results[m] = row
        preds[m] = ypd
        histories[m] = pack["history"]
        pd.DataFrame(pack["history"]).to_csv(os.path.join(out_dir, f"{name}_{m}_history.csv"), index=False)
        torch.save(
            {
                "model": m,
                "dataset": name,
                "state_dict": {k: v.detach().cpu() for k, v in pack["model"].state_dict().items()},
                "epochs_ran": pack["epochs_ran"],
                "best_val_mae": pack["best_val_mae"],
            },
            os.path.join(out_dir, f"{name}_{m}.pt"),
        )
        print(
            f"  {m}: MAE={acc['MAE']:.4f}  RMSE={acc['RMSE']:.4f}  "
            f"epochs={pack['epochs_ran']}  train={pack['train_sec']:.1f}s"
        )

    plot_loss_curves(histories, out_dir, name, loss)

    for m, yp in preds.items():
        viz.plot_prediction_curve(
            yt[:, 0, 0],
            yp[:, 0, 0],
            os.path.join(out_dir, f"{name}_{m}_node0_h1.png"),
            title=f"{name} {m} node0 h1",
        )
        viz.plot_scatter(yt, yp, os.path.join(out_dir, f"{name}_{m}_scatter.png"), title=f"{name} {m}")
        viz.plot_error_hist(yt, yp, os.path.join(out_dir, f"{name}_{m}_errhist.png"), title=f"{name} {m}")

    for metric in ["MAE", "RMSE", "MAPE", "R2", "TrainTimeSec", "InferTimeSec", "TrainableParams"]:
        viz.plot_metric_bars(
            results,
            metric,
            os.path.join(out_dir, f"{name}_{metric}_bar.png"),
            title=f"{name} {metric}",
        )

    hmae = {m: [mae(yt[:, h], preds[m][:, h]) for h in range(yt.shape[1])] for m in preds}
    viz.plot_horizon_curves(hmae, os.path.join(out_dir, f"{name}_horizon_mae.png"), title=f"{name} MAE by horizon")

    sc = scorecard(results)
    sc.to_csv(os.path.join(out_dir, f"{name}_scorecard.csv"), index=False)
    wr = float(sc["ICS-DGRN_Excellent"].mean())
    viz.set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = ["#2e7d32" if w else "#c62828" for w in sc["ICS-DGRN_Excellent"]]
    ax.barh(sc["Dimension"], [1] * len(sc), color=colors)
    n_win = int(sc["ICS-DGRN_Excellent"].sum())
    ax.set_title(f"{name}  ICS-DGRN in the leading group: {n_win}/{len(sc)}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_winrate.png"), bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(results).T.apply(pd.to_numeric, errors="coerce")
    viz.save_table(df, os.path.join(out_dir, f"{name}_metrics.csv"), os.path.join(out_dir, f"{name}_metrics.xlsx"))
    heat_cols = [c for c, _, _ in SCORE_DIMS if c in df.columns]
    if heat_cols:
        viz.plot_heatmap(df[heat_cols], os.path.join(out_dir, f"{name}_heatmap.png"), title=f"{name}")

    if "STGCN" in results and "ICS-DGRN" in results:
        delta = results["STGCN"]["MAE"] - results["ICS-DGRN"]["MAE"]
        print(f"[{name}] MAE  ICS-DGRN {results['ICS-DGRN']['MAE']:.4f}  STGCN {results['STGCN']['MAE']:.4f}  gap {delta:+.4f}")

    with open(os.path.join(out_dir, f"{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "excellence_rate": wr,
                "mae": {m: results[m]["MAE"] for m in results},
                "epochs_ran": {m: results[m]["Epochs"] for m in results},
            },
            f,
            indent=2,
        )
    return df


def write_report(datasets, epochs):
    lines = [
        f"# PEMS comparison, {epochs} epoch",
        "",
        "ICS-DGRN, STGCN, TGCN, GWNet, AGCRN.",
        "Smooth L1, Adam, cosine decay. CUDA if it is there.",
        "MAE / RMSE / R2 are on the original flow scale.",
        "",
    ]
    for ds in datasets:
        path = os.path.join(RESULT_ROOT, ds, f"{ds}_metrics.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, index_col=0)
        lines.append(f"## {ds}")
        lines.append("")
        show = [c for c in ["MAE", "RMSE", "R2", "Epochs", "TrainTimeSec"] if c in d.columns]
        lines.append(d[show].round(4).to_string())
        lines.append("")
        if "ICS-DGRN" in d.index and "STGCN" in d.index:
            gap = d.loc["STGCN", "MAE"] - d.loc["ICS-DGRN", "MAE"]
            lines.append(f"STGCN MAE minus ICS-DGRN MAE = {gap:+.4f}")
            lines.append("")
    with open(os.path.join(RESULT_ROOT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="ICS-DGRN vs STGCN / TGCN / GWNet / AGCRN")
    parser.add_argument("--datasets", nargs="+", default=["PEMS08", "PEMS04", "PEMS03"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="1 to 1000")
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--max-val", type=int, default=150)
    parser.add_argument("--max-test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs > MAX_EPOCHS:
        print(f"epochs {args.epochs} is over the cap, using {MAX_EPOCHS}")
        args.epochs = MAX_EPOCHS
    if args.epochs < 1:
        raise SystemExit("epochs must be at least 1")

    os.makedirs(RESULT_ROOT, exist_ok=True)
    frames = []
    for ds in args.datasets:
        df = run_dataset(
            ds,
            epochs=args.epochs,
            seed=args.seed,
            max_train=args.max_train,
            max_val=args.max_val,
            max_test=args.max_test,
        )
        df = df.copy()
        df["dataset"] = ds
        frames.append(df)

    big = pd.concat(frames)
    viz.save_table(
        big,
        os.path.join(RESULT_ROOT, "all_metrics.csv"),
        os.path.join(RESULT_ROOT, "all_metrics.xlsx"),
    )
    pivot = big.reset_index().pivot_table(index="index", columns="dataset", values="MAE")
    viz.plot_heatmap(
        pivot,
        os.path.join(RESULT_ROOT, "summary_MAE_heatmap.png"),
        title=f"MAE, {args.epochs} epoch",
    )
    write_report(args.datasets, args.epochs)
    print("saved", RESULT_ROOT)


if __name__ == "__main__":
    main()
