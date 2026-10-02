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

"""Build a paper-style summary report from result tables/figures."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import viz

RESULT = os.path.join(ROOT, "result")

def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(["model"] + cols) + " |", "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def _collect_pems() -> pd.DataFrame:
    frames = []
    pems_root = os.path.join(RESULT, "pems")
    for name in ["PEMS03", "PEMS04", "PEMS08"]:
        path = os.path.join(pems_root, name, f"{name}_metrics.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df = df.copy()
            df["dataset"] = name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames)
    viz.save_table(
        big,
        os.path.join(pems_root, "all_pems_metrics.csv"),
        os.path.join(pems_root, "all_pems_metrics.xlsx"),
    )
    pivot = big.reset_index().pivot_table(index="index", columns="dataset", values="MAE")
    viz.plot_heatmap_metrics(pivot, os.path.join(pems_root, "summary_mae_heatmap.png"), title="PEMS MAE summary")
    return big

def main():
    lines = ["# ICS-DGRN Experiment Report", ""]
    lines.append("## 1. Method")
    lines.append(
        "- **ICS-DGRN**: Deep Graph Reservoir (ESP analysis in `2.pdf`) + interlayer Gaussian compression (ICS-DESN)."
    )
    lines.append("- Training: fixed reservoir + ridge readout (RC / classical ML level, same as ICS-DESN).")
    lines.append("- ESP condition enforced per layer: `||W_l||_2 * ||S||_2 < 1`.")
    lines.append("")

    df = _collect_pems()
    if len(df):
        lines.append("## 2. Spatiotemporal (PEMS) results")
        lines.append("")
        lines.append(
            "Baselines: HA, LastRepeat, Ridge, ESN, DeepESN, ICS-DESN, GraphESN, plus ablations DGRN-NoICS / DGRN-NoGraph."
        )
        lines.append("")
        for ds in sorted(df["dataset"].unique()):
            sub = df[df["dataset"] == ds].sort_values("MAE")
            lines.append(f"### {ds} (sorted by MAE)")
            lines.append("")
            lines.append(_df_to_md(sub[["MAE", "RMSE", "MSE", "R2", "TimeSec"]].round(4)))
            lines.append("")
            if "ICS-DESN" in sub.index and "ICS-DGRN" in sub.index:
                ics = float(sub.loc["ICS-DESN", "MAE"])
                prop = float(sub.loc["ICS-DGRN", "MAE"])
                gain = (ics - prop) / ics * 100
                lines.append(
                    f"- ICS-DGRN MAE={prop:.4f}, ICS-DESN MAE={ics:.4f}, relative MAE reduction={gain:.2f}%."
                )
            lines.append(f"- Best on this split: **{sub.index[0]}**.")
            lines.append("")

    ts_csv = os.path.join(RESULT, "timeseries", "all_timeseries_metrics.csv")
    if os.path.exists(ts_csv):
        tdf = pd.read_csv(ts_csv, index_col=0)
        lines.append("## 3. Time-series (ICS-DESN paper datasets)")
        lines.append("")
        for ds in sorted(tdf["dataset"].unique()):
            sub = tdf[tdf["dataset"] == ds].sort_values("MSE")
            lines.append(f"### {ds}")
            lines.append("")
            lines.append(_df_to_md(sub[["MSE", "MAE", "RMSE", "R2"]].round(6)))
            lines.append("")

    lines.append("## 4. Figure / table locations")
    lines.append("")
    lines.append("- `result/pems/<DATASET>/` : bars, radar, heatmap, horizon curves, ESP decay, adjacency, predictions")
    lines.append("- `result/timeseries/<DATASET>/` : ICS-DESN-style curves and metric tables")
    lines.append("- `result/REPORT.md` : this file")
    lines.append("")

    out = os.path.join(RESULT, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote", out)

if __name__ == "__main__":
    main()
