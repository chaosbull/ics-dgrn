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

"""200-epoch full comparison: ICS-DGRN(v2) vs STGCN/TGCN/GWNet/AGCRN."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.pems_loader import load_pems
from baselines.trainable_st import TrainConfig, build_model, predict_st_model, train_st_model
from utils.metrics import evaluate_all, mae
from utils import viz

RESULT_ROOT = os.path.join(ROOT, "result", "pems_st200")

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

def plot_loss_curves(histories: Dict[str, list], out_dir: str, ds: str):
    viz.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, hist in histories.items():
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=f"{name}-train", lw=1.3)
        axes[0].plot(ep, [h["val_loss"] for h in hist], label=f"{name}-val", lw=1.3, ls="--")
        axes[1].plot(ep, [h["train_mae"] for h in hist], label=f"{name}-train", lw=1.3)
        axes[1].plot(ep, [h["val_mae"] for h in hist], label=f"{name}-val", lw=1.3, ls="--")
    axes[0].set_title(f"{ds} Loss ({histories and list(histories.values())[0][0].get('loss','')})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title(f"{ds} MAE (normalized space)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{ds}_loss_mae_curves.png"), bbox_inches="tight")
    plt.close(fig)

    # per-model separate figures
    for name, hist in histories.items():
        if not hist:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color="#1f4e79", label="train_loss")
        axes[0].plot(ep, [h["val_loss"] for h in hist], color="#c45911", label="val_loss")
        axes[0].set_title(f"{ds} | {name} | Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(ep, [h["train_mae"] for h in hist], color="#1f4e79", label="train_mae")
        axes[1].plot(ep, [h["val_mae"] for h in hist], color="#c45911", label="val_mae")
        axes[1].set_title(f"{ds} | {name} | MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{ds}_{name}_curves.png"), bbox_inches="tight")
        plt.close(fig)

def scorecard(results: Dict[str, Dict], proposed: str = "ICS-DGRN") -> pd.DataFrame:
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

def run_one_model(name, data, epochs, seed, loss):
    N, H, hist = data["n_nodes"], data["horizon"], data["hist_len"]
    model = build_model(name, N, hist, H)
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=48 if name == "ICS-DGRN" else 32,
        lr=1.2e-3 if name == "ICS-DGRN" else 1e-3,
        patience=max(epochs, 20),
        seed=seed,
        device="auto",
        loss=loss,
        weight_decay=5e-5 if name == "ICS-DGRN" else 1e-4,
    )
    print(f"  Training {name} for up to {epochs} epochs ...")
    pack = train_st_model(model, data["adj"], data["x_train"], data["y_train"], data["x_val"], data["y_val"], cfg)
    yp, infer_sec = predict_st_model(pack, data["x_test"])
    # annotate history with loss name
    for h in pack["history"]:
        h["loss"] = pack.get("loss_name", loss)
    return pack, yp, infer_sec

def run_dataset(name: str, epochs: int = 200, seed: int = 42, max_train: int = 800, max_val: int = 150, max_test: int = 250):
    out_dir = os.path.join(RESULT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    data = load_pems(name, max_train=max_train, max_val=max_val, max_test=max_test, seed=seed)
    denorm = data["denorm"]
    y_te = data["y_test"]
    yt = denorm(y_te)

    models = ["ICS-DGRN", "STGCN", "TGCN", "GWNet", "AGCRN"]
    results = {}
    preds = {}
    histories = {}

    for m in models:
        print("=" * 60, name, m)
        loss = "smooth_l1"
        pack, yp, infer_sec = run_one_model(m, data, epochs, seed, loss)
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
            "LossFn": pack.get("loss_name", loss),
        }
        if m == "ICS-DGRN":
            try:
                esp = pack["model"].esp_report(pack["adj_t"])
                row["ESP_Guarantee"] = 1.0 if esp.get("esp_ok") else 0.0
                with open(os.path.join(out_dir, f"{name}_esp.json"), "w", encoding="utf-8") as f:
                    json.dump(esp, f, indent=2)
            except Exception as e:
                print("esp report failed", e)
        results[m] = row
        preds[m] = ypd
        histories[m] = pack["history"]
        # save history csv
        pd.DataFrame(pack["history"]).to_csv(os.path.join(out_dir, f"{name}_{m}_history.csv"), index=False)
        print(
            f"  => {m}: MAE={acc['MAE']:.4f} RMSE={acc['RMSE']:.4f} "
            f"epochs={pack['epochs_ran']} train={pack['train_sec']:.1f}s"
        )

    plot_loss_curves(histories, out_dir, name)

    # prediction figures
    for m, yp in preds.items():
        viz.plot_prediction_curve(
            yt[:, 0, 0], yp[:, 0, 0], os.path.join(out_dir, f"{name}_{m}_node0_h1.png"), title=f"{name}|{m}|node0|h1"
        )
        viz.plot_scatter(yt, yp, os.path.join(out_dir, f"{name}_{m}_scatter.png"), title=f"{name}|{m}")
        viz.plot_error_hist(yt, yp, os.path.join(out_dir, f"{name}_{m}_errhist.png"), title=f"{name}|{m}")

    for metric in ["MAE", "RMSE", "MAPE", "R2", "TrainTimeSec", "InferTimeSec", "TrainableParams"]:
        viz.plot_metric_bars(results, metric, os.path.join(out_dir, f"{name}_{metric}_bar.png"), title=f"{name} {metric}")

    hmae = {m: [mae(yt[:, h], preds[m][:, h]) for h in range(yt.shape[1])] for m in preds}
    viz.plot_horizon_curves(hmae, os.path.join(out_dir, f"{name}_horizon_mae.png"), title=f"{name} MAE vs horizon")

    sc = scorecard(results)
    sc.to_csv(os.path.join(out_dir, f"{name}_scorecard.csv"), index=False)
    wr = float(sc["ICS-DGRN_Excellent"].mean())
    viz.set_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = ["#2e7d32" if w else "#c62828" for w in sc["ICS-DGRN_Excellent"]]
    ax.barh(sc["Dimension"], [1] * len(sc), color=colors)
    ax.set_title(f"{name} ICS-DGRN excellence={wr*100:.0f}% ({int(sc['ICS-DGRN_Excellent'].sum())}/{len(sc)})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_winrate.png"), bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(results).T
    if "LossFn" in df.columns:
        df = df.drop(columns=["LossFn"])
    df = df.apply(pd.to_numeric, errors="coerce")
    viz.save_table(df, os.path.join(out_dir, f"{name}_metrics.csv"), os.path.join(out_dir, f"{name}_metrics.xlsx"))
    heat_cols = [c for c, _, _ in SCORE_DIMS if c in df.columns]
    if heat_cols:
        viz.plot_heatmap_metrics(
            df[heat_cols],
            os.path.join(out_dir, f"{name}_heatmap.png"),
            title=f"{name} metrics",
        )

    # MAE comparison highlight vs STGCN
    if "STGCN" in results and "ICS-DGRN" in results:
        delta = results["STGCN"]["MAE"] - results["ICS-DGRN"]["MAE"]
        print(f"[{name}] MAE: ICS-DGRN={results['ICS-DGRN']['MAE']:.4f} STGCN={results['STGCN']['MAE']:.4f} delta={delta:+.4f}")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["PEMS08", "PEMS04", "PEMS03"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--max-val", type=int, default=150)
    parser.add_argument("--max-test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    viz.save_table(big, os.path.join(RESULT_ROOT, "all_metrics.csv"), os.path.join(RESULT_ROOT, "all_metrics.xlsx"))

    # MAE comparison table figure
    pivot = big.reset_index().pivot_table(index="index", columns="dataset", values="MAE")
    viz.plot_heatmap_metrics(
        pivot, os.path.join(RESULT_ROOT, "summary_MAE_heatmap.png"), title=f"MAE @ {args.epochs} epochs"
    )

    lines = [
        f"# {args.epochs}-epoch DL comparison",
        "",
        "Models: ICS-DGRN(v2), STGCN, TGCN, GWNet, AGCRN",
        f"Epochs (max): {args.epochs}, loss: SmoothL1, optimizer: Adam + CosineAnnealing, device: CUDA auto",
        "",
    ]
    for ds in args.datasets:
        p = os.path.join(RESULT_ROOT, ds, f"{ds}_metrics.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, index_col=0)
            lines.append(f"## {ds}")
            lines.append("")
            lines.append(d[["MAE", "RMSE", "R2", "Epochs", "TrainTimeSec"]].round(4).to_string())
            lines.append("")
            if "ICS-DGRN" in d.index and "STGCN" in d.index:
                lines.append(
                    f"- MAE gap (STGCN - ICS-DGRN) = {d.loc['STGCN','MAE'] - d.loc['ICS-DGRN','MAE']:+.4f}"
                )
            lines.append("")
    with open(os.path.join(RESULT_ROOT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved", RESULT_ROOT)

if __name__ == "__main__":
    main()
