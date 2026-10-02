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

"""
Time-series experiments on ICS-DESN paper datasets.
Baselines are reservoir / classical ML (same paradigm as ICS-DESN).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.loaders import load_dataset
from model import (
    DGRNConfig,
    DeepESN,
    DeepESNConfig,
    ESN,
    ESNConfig,
    ICSDESN,
    ICSDESNConfig,
    ICSDGRN,
    correlation_graph,
    delay_embedding_graph,
)
from model.reservoir_utils import add_bias, ridge_fit, ridge_predict
from baselines import LinearSVRMulti, RidgeForecaster
from utils.metrics import evaluate_all
from utils import viz
from utils.esp_analysis import simulate_state_difference

RESULT_ROOT = os.path.join(ROOT, "result", "timeseries")

def _fit_seq_rc(model, u_tr, y_tr, u_te_full, y_te, wash: int):
    """Train on prefix; predict on test using warm-start wash from train tail."""
    model.fit(u_tr, y_tr)
    # warm start: run last `wash` train steps then continue on test inputs
    u_all = np.vstack([u_tr[-wash:], u_te_full]) if wash > 0 else u_te_full
    yp = model.predict(u_all, washout=wash)
    return yp[: len(y_te)]

def _build_graph_seq(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Returns u_graph (T,N), y (T,1), adj, n_train
    """
    if raw.shape[1] == 1:
        target = raw[:, 0]
        emb = 16
        adj = delay_embedding_graph(emb)
        U = np.stack([target[i : i + emb] for i in range(len(target) - emb)], axis=0)
        y = target[emb:].reshape(-1, 1)
        n_tr = int(0.8 * len(U))
        return U, y, adj, n_tr

    adj = correlation_graph(raw, threshold=0.25)
    u = raw[:-1]
    y = raw[1:, -1:].copy()
    n_tr = int(0.8 * len(u))
    return u, y, adj, n_tr

def _fit_dgrn_seq(dgrn: ICSDGRN, u: np.ndarray, y: np.ndarray, n_tr: int, wash: int, ridge: float):
    dgrn.reset()
    feats = []
    for t in range(n_tr):
        f = dgrn.step(u[t])
        if t >= wash:
            feats.append(f)
    Xf = add_bias(np.stack(feats, axis=1))
    Yf = y[wash:n_tr].T
    dgrn.wout = ridge_fit(Xf, Yf, ridge=ridge)

    dgrn.reset()
    all_f = [dgrn.step(u[t]) for t in range(len(u))]
    yp = ridge_predict(dgrn.wout, add_bias(np.stack(all_f, axis=1))).T
    return yp[n_tr:], y[n_tr:]

def run_one_dataset(name: str, out_dir: str, max_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    pack = load_dataset(name, window=24, horizon=1, train_ratio=0.8, max_samples=max_samples)
    raw = pack["raw"]
    target = raw[:, -1]
    u_seq = target[:-1].reshape(-1, 1)
    y_seq = target[1:].reshape(-1, 1)
    n_tr = int(0.8 * len(u_seq))
    wash = 30

    results: Dict[str, Dict] = {}
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def record(key, yt, yp, tcost):
        results[key] = evaluate_all(yt, yp)
        results[key]["TimeSec"] = tcost
        preds[key] = (np.asarray(yt), np.asarray(yp))
        print(f"  {name}/{key}: MAE={results[key]['MAE']:.6f} MSE={results[key]['MSE']:.6f}")

    # ESN
    t0 = time.time()
    esn = ESN(1, 1, ESNConfig(n_reservoir=120, washout=wash, seed=seed))
    yp = _fit_seq_rc(esn, u_seq[:n_tr], y_seq[:n_tr], u_seq[n_tr:], y_seq[n_tr:], wash)
    record("ESN", y_seq[n_tr:], yp, time.time() - t0)

    # DeepESN
    t0 = time.time()
    desn = DeepESN(1, 1, DeepESNConfig(layer_sizes=(80, 60, 40), washout=wash, seed=seed))
    yp = _fit_seq_rc(desn, u_seq[:n_tr], y_seq[:n_tr], u_seq[n_tr:], y_seq[n_tr:], wash)
    record("DeepESN", y_seq[n_tr:], yp, time.time() - t0)

    # ICS-DESN
    t0 = time.time()
    ics = ICSDESN(
        1,
        1,
        ICSDESNConfig(layer_sizes=(100, 80, 60), compression_dims=(40, 30), washout=wash, seed=seed),
    )
    yp = _fit_seq_rc(ics, u_seq[:n_tr], y_seq[:n_tr], u_seq[n_tr:], y_seq[n_tr:], wash)
    record("ICS-DESN", y_seq[n_tr:], yp, time.time() - t0)

    # Ridge windows
    Xw = pack["X"].T  # (N, F) -> samples first
    Yw = pack["Y"].T
    n_tr_w = len(pack["train_idx"])
    t0 = time.time()
    ridge = RidgeForecaster()
    ridge.fit(Xw[:n_tr_w], Yw[:n_tr_w])
    yp = ridge.predict(Xw[n_tr_w:])
    record("Ridge", Yw[n_tr_w:], yp, time.time() - t0)

    if len(Xw) <= 2500:
        t0 = time.time()
        m = min(n_tr_w, 1200)
        svr = LinearSVRMulti(C=1.0, max_iter=1200, seed=seed)
        svr.fit(Xw[:m], Yw[:m])
        yp = svr.predict(Xw[n_tr_w:])
        record("LinearSVR", Yw[n_tr_w:], yp, time.time() - t0)

    # ICS-DGRN
    t0 = time.time()
    u_g, y_g, adj, n_tr_g = _build_graph_seq(raw)
    n_nodes = u_g.shape[1]
    dgrn = ICSDGRN(
        n_nodes=n_nodes,
        n_in_feat=1,
        n_outputs=1,
        adj=adj,
        cfg=DGRNConfig(
            layer_dims=(24, 16),
            compression_dims=(10,),
            washout=0,
            seed=seed,
            ridge=1e-3,
            esp_target=0.85,
            leak=0.5,
        ),
    )
    yp, yt = _fit_dgrn_seq(dgrn, u_g, y_g, n_tr_g, wash=20, ridge=1e-3)
    record("ICS-DGRN", yt, yp, time.time() - t0)

    viz.plot_adjacency(adj, os.path.join(out_dir, f"{name}_graph.png"), title=f"{name} graph")
    with open(os.path.join(out_dir, f"{name}_esp.json"), "w", encoding="utf-8") as f:
        json.dump(dgrn.esp_report(), f, indent=2)
    decay = simulate_state_difference(dgrn, u_g[: min(400, len(u_g))])
    viz.plot_esp_decay({"ICS-DGRN": decay}, os.path.join(out_dir, f"{name}_esp_decay.png"))

    for key, (yt_, yp_) in preds.items():
        viz.plot_prediction_curve(yt_, yp_, os.path.join(out_dir, f"{name}_{key}_pred.png"), title=f"{name} | {key}")
        viz.plot_scatter(yt_, yp_, os.path.join(out_dir, f"{name}_{key}_scatter.png"), title=f"{name} | {key}")
        viz.plot_error_hist(yt_, yp_, os.path.join(out_dir, f"{name}_{key}_errhist.png"), title=f"{name} | {key}")

    for metric in ["MSE", "MAE", "RMSE", "MAPE"]:
        viz.plot_metric_bars(results, metric, os.path.join(out_dir, f"{name}_{metric.lower()}_bar.png"), title=f"{name} {metric}")
    viz.plot_radar(results, ["MSE", "MAE", "RMSE", "MAPE", "R2"], os.path.join(out_dir, f"{name}_radar.png"))

    df = pd.DataFrame(results).T
    viz.save_table(df, os.path.join(out_dir, f"{name}_metrics.csv"), os.path.join(out_dir, f"{name}_metrics.xlsx"))
    viz.plot_heatmap_metrics(df[["MSE", "MAE", "RMSE", "MAPE", "R2"]], os.path.join(out_dir, f"{name}_heatmap.png"))
    print(df.round(6))
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["logistic", "lorenz", "sunspot", "ETTh1", "weather"])
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULT_ROOT, exist_ok=True)
    frames = []
    for ds in args.datasets:
        print("=" * 60, ds)
        df = run_one_dataset(ds, os.path.join(RESULT_ROOT, ds), max_samples=args.max_samples, seed=args.seed)
        df = df.copy()
        df["dataset"] = ds
        frames.append(df)

    big = pd.concat(frames)
    viz.save_table(
        big,
        os.path.join(RESULT_ROOT, "all_timeseries_metrics.csv"),
        os.path.join(RESULT_ROOT, "all_timeseries_metrics.xlsx"),
    )
    pivot = big.reset_index().pivot_table(index="index", columns="dataset", values="MSE")
    viz.plot_heatmap_metrics(pivot, os.path.join(RESULT_ROOT, "summary_mse_heatmap.png"), title="MSE across datasets")
    print("Saved:", RESULT_ROOT)

if __name__ == "__main__":
    main()
