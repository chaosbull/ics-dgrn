"""Retrain ICS-DGRN only with a longer budget; baselines stay fixed."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.run_p0_bundle import RESULT, load_data, run_one

OUT = os.path.join(RESULT, "ics_long_runs.csv")


def main():
    with open(os.path.join(RESULT, "ics_config.json"), encoding="utf-8") as f:
        ics_cfg = json.load(f)
    epochs, patience, bs, lr = 80, 18, 128, 1.2e-3
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    for ds in ("PEMS08", "PEMS04", "PEMS03"):
        for seed in (7, 21, 42):
            if len(done) and ((done.dataset == ds) & (done.seed == seed)).any():
                print("skip", ds, seed, flush=True)
                continue
            print(f"LONG {ds} seed {seed}", flush=True)
            data = load_data(ds, seed, 1600, 400, 500)
            out = run_one("ICS-DGRN", data, seed, epochs, bs, lr, patience, ics_cfg)
            row = out["row"]
            row["dataset"] = ds
            row["model"] = "ICS-DGRN"
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            np.save(os.path.join(RESULT, f"errlong_{ds}_ICS-DGRN_s{seed}.npy"), out["err"])
            print(
                f"  MAE={row['MAE']:.3f} RMSE={row['RMSE']:.3f} R2={row['R2']:.4f} "
                f"h1={row['MAE_h1']:.3f} epochs={row['Epochs']:.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
