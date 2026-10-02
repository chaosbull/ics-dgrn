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

"""ICS-DGRN experiment entrypoint (PEMS DL / RC / timeseries)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def main():
    parser = argparse.ArgumentParser(description="ICS-DGRN experiments")
    parser.add_argument(
        "--task",
        choices=["pems_st", "pems_rc", "timeseries", "all"],
        default="pems_st",
    )
    parser.add_argument("--pems-datasets", nargs="+", default=["PEMS08", "PEMS04", "PEMS03"])
    parser.add_argument("--ts-datasets", nargs="+", default=["logistic", "lorenz", "sunspot", "ETTh1", "weather"])
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--max-val", type=int, default=150)
    parser.add_argument("--max-test", type=int, default=250)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task in ("pems_st", "all"):
        from experiments.run_pems_200ep import main as st_main
        import sys as _sys

        argv = [
            "run_pems_200ep",
            "--datasets",
            *args.pems_datasets,
            "--max-train",
            str(args.max_train),
            "--max-val",
            str(args.max_val),
            "--max-test",
            str(args.max_test),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
        ]
        old = _sys.argv
        _sys.argv = argv
        try:
            st_main()
        finally:
            _sys.argv = old

    if args.task in ("pems_rc",):
        from experiments.run_pems import run_pems
        import pandas as pd
        from utils import viz

        frames = []
        for ds in args.pems_datasets:
            df = run_pems(ds, max_train=args.max_train, max_val=args.max_val, max_test=args.max_test, seed=args.seed)
            df = df.copy()
            df["dataset"] = ds
            frames.append(df)
        big = pd.concat(frames)
        out = os.path.join(ROOT, "result", "pems")
        os.makedirs(out, exist_ok=True)
        viz.save_table(big, os.path.join(out, "all_pems_metrics.csv"), os.path.join(out, "all_pems_metrics.xlsx"))

    if args.task in ("timeseries", "all"):
        from experiments.run_timeseries import run_one_dataset, RESULT_ROOT as TS_ROOT
        import pandas as pd
        from utils import viz

        os.makedirs(TS_ROOT, exist_ok=True)
        frames = []
        for ds in args.ts_datasets:
            df = run_one_dataset(ds, os.path.join(TS_ROOT, ds), max_samples=args.max_samples, seed=args.seed)
            df = df.copy()
            df["dataset"] = ds
            frames.append(df)
        big = pd.concat(frames)
        viz.save_table(
            big,
            os.path.join(TS_ROOT, "all_timeseries_metrics.csv"),
            os.path.join(TS_ROOT, "all_timeseries_metrics.xlsx"),
        )

    print("Done. See feifa2/result/")

if __name__ == "__main__":
    main()
