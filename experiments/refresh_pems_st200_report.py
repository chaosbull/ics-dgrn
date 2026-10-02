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

"""Rebuild pems_st200 REPORT.md + all_metrics from per-dataset metrics CSVs."""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import viz

RESULT_ROOT = os.path.join(ROOT, "result", "pems_st200")
DATASETS = ["PEMS08", "PEMS04", "PEMS03"]

def main():
    frames = []
    epochs_seen = set()
    lines = [
        "# 300-epoch DL comparison",
        "",
        "Models: ICS-DGRN(v3), STGCN, TGCN, GWNet, AGCRN",
        "Epochs (max): 300, loss: SmoothL1, optimizer: Adam + CosineAnnealing, device: CUDA auto",
        "",
    ]
    for ds in DATASETS:
        p = os.path.join(RESULT_ROOT, ds, f"{ds}_metrics.csv")
        if not os.path.exists(p):
            lines.append(f"## {ds}")
            lines.append("")
            lines.append("_missing metrics_")
            lines.append("")
            continue
        d = pd.read_csv(p, index_col=0)
        if "Epochs" in d.columns:
            epochs_seen.update(int(v) for v in d["Epochs"].dropna().tolist())
        df = d.copy()
        df["dataset"] = ds
        frames.append(df)
        lines.append(f"## {ds}")
        lines.append("")
        cols = [c for c in ["MAE", "RMSE", "R2", "Epochs", "TrainTimeSec"] if c in d.columns]
        lines.append(d[cols].round(4).to_string())
        lines.append("")
        if "ICS-DGRN" in d.index and "STGCN" in d.index:
            lines.append(
                f"- MAE gap (STGCN - ICS-DGRN) = {d.loc['STGCN','MAE'] - d.loc['ICS-DGRN','MAE']:+.4f}"
            )
        if "ICS-DGRN" in d.index and "AGCRN" in d.index:
            lines.append(
                f"- MAE gap (AGCRN - ICS-DGRN) = {d.loc['AGCRN','MAE'] - d.loc['ICS-DGRN','MAE']:+.4f}"
            )
        lines.append("")

    if epochs_seen:
        lines[0] = f"# {max(epochs_seen)}-epoch DL comparison"
        lines[3] = (
            f"Epochs (max): {max(epochs_seen)}, loss: SmoothL1, "
            "optimizer: Adam + CosineAnnealing, device: CUDA auto"
        )

    report_path = os.path.join(RESULT_ROOT, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if frames:
        big = pd.concat(frames)
        viz.save_table(
            big,
            os.path.join(RESULT_ROOT, "all_metrics.csv"),
            os.path.join(RESULT_ROOT, "all_metrics.xlsx"),
        )
        if "MAE" in big.columns:
            pivot = big.reset_index().pivot_table(index="index", columns="dataset", values="MAE")
            viz.plot_heatmap_metrics(
                pivot,
                os.path.join(RESULT_ROOT, "summary_MAE_heatmap.png"),
                title=f"MAE @ {max(epochs_seen) if epochs_seen else '?'} epochs",
            )
    print("Refreshed", report_path)

if __name__ == "__main__":
    main()
