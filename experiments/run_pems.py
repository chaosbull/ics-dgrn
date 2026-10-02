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
Spatiotemporal traffic forecasting on PEMS (core target of ICS-DGRN).

Reservoir / classical-ML level baselines (matching ICS-DESN paradigm):
  HA, LastRepeat, Ridge, ESN-pool, GraphESN, DeepESN-flat, ICS-DESN-flat, ICS-DGRN
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

from data.pems_loader import load_pems, list_pems
from model import (
    DGRNConfig,
    DeepESN,
    DeepESNConfig,
    ESN,
    ESNConfig,
    GraphESN,
    GraphESNConfig,
    ICSDESN,
    ICSDESNConfig,
    ICSDGRN,
)
from model.reservoir_utils import add_bias, ridge_fit, ridge_predict
from baselines import HistoricalAverage, LastRepeat, RidgeForecaster, NodeWiseESNAdapter
from utils.metrics import evaluate_all, mae
from utils import viz
from utils.esp_analysis import simulate_state_difference

RESULT_ROOT = os.path.join(ROOT, "result", "pems")

def _denorm_eval(y_true, y_pred, denorm) -> Dict[str, float]:
    yt = denorm(y_true)
    yp = denorm(y_pred)
    return evaluate_all(yt, yp)

def _horizon_mae(y_true, y_pred, denorm) -> list:
    yt = denorm(y_true)
    yp = denorm(y_pred)
    # (B,H,N)
    out = []
    for h in range(yt.shape[1]):
        out.append(mae(yt[:, h], yp[:, h]))
    return out

def flat_rc_features_from_windows(model_step_fn, reset_fn, x: np.ndarray) -> np.ndarray:
    """Generic: run a vector RC on flattened spatial mean; return (F,B)."""
    B, T, N = x.shape
    feats = []
    for i in range(B):
        reset_fn()
        st = None
        u = x[i].mean(axis=1)  # (T,)
        for t in range(T):
            st = model_step_fn(np.array([u[t]]))
        feats.append(st)
    return np.stack(feats, axis=1)

def fit_flat_rc(build_model, x_tr, y_tr, ridge=1e-3, residual=True):
    model = build_model()
    F = flat_rc_features_from_windows(model.step, model.reset, x_tr)
    # append last observation mean as cue
    last = x_tr[:, -1, :].reshape(len(x_tr), -1).T  # (N, B)
    F = np.vstack([F, last])
    Fb = add_bias(F)
    if residual:
        target = y_tr - x_tr[:, -1:, :]
    else:
        target = y_tr
    yb = target.reshape(len(y_tr), -1).T
    wout = ridge_fit(Fb, yb, ridge=ridge)
    return model, wout, residual

def predict_flat_rc(model, wout, x, out_shape, residual=True):
    F = flat_rc_features_from_windows(model.step, model.reset, x)
    last = x[:, -1, :].reshape(len(x), -1).T
    F = np.vstack([F, last])
    y = ridge_predict(wout, add_bias(F)).T
    y = y.reshape((-1,) + out_shape)
    if residual:
        y = y + x[:, -1:, :]
    return y

class ResidualRidge:
    """Ridge on flattened windows predicting residual over last frame."""

    def __init__(self, alpha: float = 1.0):
        from baselines.ml_baselines import RidgeConfig

        self.inner = RidgeForecaster(cfg=RidgeConfig(alpha=alpha))

    def fit(self, x, y):
        target = y - x[:, -1:, :]
        self.inner.fit(x, target)
        return self

    def predict(self, x):
        return self.inner.predict(x) + x[:, -1:, :]

def run_pems(
    name: str = "PEMS08",
    max_train: int = 800,
    max_val: int = 200,
    max_test: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    out_dir = os.path.join(RESULT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    data = load_pems(name, max_train=max_train, max_val=max_val, max_test=max_test, seed=seed)
    x_tr, y_tr = data["x_train"], data["y_train"]
    x_va, y_va = data["x_val"], data["y_val"]
    x_te, y_te = data["x_test"], data["y_test"]
    adj = data["adj"]
    denorm = data["denorm"]
    N = data["n_nodes"]
    H = data["horizon"]
    out_shape = (H, N)

    viz.plot_adjacency(adj, os.path.join(out_dir, f"{name}_adjacency.png"), title=f"{name} adjacency")

    results: Dict[str, Dict] = {}
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    horizon_maes: Dict[str, list] = {}

    def record(key, model_fit_predict):
        t0 = time.time()
        y_hat = model_fit_predict()
        # optional val unused; report test
        met = _denorm_eval(y_te, y_hat, denorm)
        met["TimeSec"] = time.time() - t0
        results[key] = met
        preds[key] = (y_te, y_hat)
        horizon_maes[key] = _horizon_mae(y_te, y_hat, denorm)
        print(f"  {key}: MAE={met['MAE']:.4f} RMSE={met['RMSE']:.4f} time={met['TimeSec']:.1f}s")

    # ---- HA ----
    def _ha():
        m = HistoricalAverage().fit(x_tr, y_tr)
        return m.predict(x_te)

    record("HA", _ha)

    # ---- LastRepeat ----
    def _last():
        m = LastRepeat().fit(x_tr, y_tr)
        return m.predict(x_te)

    record("LastRepeat", _last)

    # ---- Ridge ----
    def _ridge():
        m = ResidualRidge(alpha=1.0).fit(x_tr, y_tr)
        return m.predict(x_te)

    record("Ridge", _ridge)

    # ---- ESN (spatial-mean drive) ----
    def _esn():
        model, wout, res = fit_flat_rc(
            lambda: ESN(1, H * N, ESNConfig(n_reservoir=150, washout=0, seed=seed, ridge=1e-3)),
            x_tr,
            y_tr,
        )
        return predict_flat_rc(model, wout, x_te, out_shape, residual=res)

    record("ESN", _esn)

    # ---- DeepESN ----
    def _desn():
        model, wout, res = fit_flat_rc(
            lambda: DeepESN(
                1,
                H * N,
                DeepESNConfig(layer_sizes=(100, 80, 60), washout=0, seed=seed, ridge=1e-3),
            ),
            x_tr,
            y_tr,
        )
        return predict_flat_rc(model, wout, x_te, out_shape, residual=res)

    record("DeepESN", _desn)

    # ---- ICS-DESN ----
    def _ics():
        model, wout, res = fit_flat_rc(
            lambda: ICSDESN(
                1,
                H * N,
                ICSDESNConfig(
                    layer_sizes=(100, 80, 60),
                    compression_dims=(40, 30),
                    washout=0,
                    seed=seed,
                    ridge=1e-3,
                ),
            ),
            x_tr,
            y_tr,
        )
        return predict_flat_rc(model, wout, x_te, out_shape, residual=res)

    record("ICS-DESN", _ics)

    # Larger reservoirs OK with node-wise readout
    g_dim, d_dims, c_dim = 48, (48, 32), 24

    def _with_graph_features(x_arr, s_op):
        # x: (B,T,N) -> (B,T,N,2) with [u, S u]
        su = np.einsum("ij,btj->bti", s_op, x_arr)
        return np.stack([x_arr, su], axis=-1)

    from model.graph_ops import normalize_adjacency

    S_feat = normalize_adjacency(adj, add_self_loop=True, mode="sym")
    x_tr_g = _with_graph_features(x_tr, S_feat)
    x_va_g = _with_graph_features(x_va, S_feat)
    x_te_g = _with_graph_features(x_te, S_feat)

    # ---- GraphESN ----
    def _gesn():
        cfg = GraphESNConfig(
            layer_dims=(g_dim,),
            washout=0,
            seed=seed,
            ridge=1e-3,
            esp_target=0.9,
            pool="mix",
            residual=True,
            append_input=True,
            graph_norm="sym",
            leak=0.55,
        )
        m = GraphESN(N, 2, H * N, adj, cfg=cfg)
        m.fit_windows(x_tr_g, y_tr)
        return m.predict_windows(x_te_g, out_shape=out_shape)

    record("GraphESN", _gesn)

    # ---- ICS-DGRN (proposed) ----
    def _dgrn():
        cfg = DGRNConfig(
            layer_dims=d_dims,
            compression_dims=(c_dim,),
            washout=0,
            seed=seed,
            ridge=1e-3,
            esp_target=0.9,
            leak=0.55,
            pool="mix",
            density=0.25,
            residual=True,
            append_input=True,
            input_scale=0.6,
            graph_norm="sym",
        )
        m = ICSDGRN(N, 2, H * N, adj, cfg=cfg)
        m.fit_windows(x_tr_g, y_tr)
        with open(os.path.join(out_dir, f"{name}_esp.json"), "w", encoding="utf-8") as f:
            json.dump(m.esp_report(), f, indent=2)
        decay = simulate_state_difference(m, x_tr_g[0])
        viz.plot_esp_decay({"ICS-DGRN": decay}, os.path.join(out_dir, f"{name}_esp_decay.png"))
        return m.predict_windows(x_te_g, out_shape=out_shape)

    record("ICS-DGRN", _dgrn)

    # ---- Ablation: DGRN without feature compression ----
    def _dgrn_nocomp():
        cfg = DGRNConfig(
            layer_dims=d_dims,
            compression_dims=(d_dims[0],),
            washout=0,
            seed=seed + 1,
            ridge=1e-3,
            esp_target=0.9,
            pool="mix",
            residual=True,
            append_input=True,
            leak=0.55,
            graph_norm="sym",
            input_scale=0.6,
        )
        m = ICSDGRN(N, 2, H * N, adj, cfg=cfg)
        m.fit_windows(x_tr_g, y_tr)
        return m.predict_windows(x_te_g, out_shape=out_shape)

    record("DGRN-NoICS", _dgrn_nocomp)

    # ---- Ablation: no graph (S = I, no spatial input feat) ----
    def _dgrn_nograph():
        cfg = DGRNConfig(
            layer_dims=d_dims,
            compression_dims=(c_dim,),
            washout=0,
            seed=seed + 2,
            ridge=1e-3,
            esp_target=0.9,
            pool="mix",
            residual=True,
            append_input=True,
            leak=0.55,
            graph_norm="none",
            input_scale=0.6,
        )
        eye = np.eye(N)
        # single-channel input without spatial diffusion
        m = ICSDGRN(N, 1, H * N, eye, cfg=cfg)
        m.fit_windows(x_tr, y_tr)
        return m.predict_windows(x_te, out_shape=out_shape)

    record("DGRN-NoGraph", _dgrn_nograph)

    # figures: predictions for best few models on node 0 horizon-0
    for key, (yt, yp) in preds.items():
        series_true = denorm(yt)[:, 0, 0]
        series_pred = denorm(yp)[:, 0, 0]
        viz.plot_prediction_curve(
            series_true,
            series_pred,
            os.path.join(out_dir, f"{name}_{key}_node0_h1.png"),
            title=f"{name} | {key} | node0 | horizon1",
            max_points=300,
        )
        viz.plot_scatter(
            denorm(yt),
            denorm(yp),
            os.path.join(out_dir, f"{name}_{key}_scatter.png"),
            title=f"{name} | {key}",
        )
        viz.plot_error_hist(
            denorm(yt),
            denorm(yp),
            os.path.join(out_dir, f"{name}_{key}_errhist.png"),
            title=f"{name} | {key}",
        )

    viz.plot_metric_bars(results, "MAE", os.path.join(out_dir, f"{name}_mae_bar.png"), title=f"{name} MAE (denorm)")
    viz.plot_metric_bars(results, "RMSE", os.path.join(out_dir, f"{name}_rmse_bar.png"), title=f"{name} RMSE")
    viz.plot_metric_bars(results, "MSE", os.path.join(out_dir, f"{name}_mse_bar.png"), title=f"{name} MSE")
    viz.plot_metric_bars(results, "MAPE", os.path.join(out_dir, f"{name}_mape_bar.png"), title=f"{name} MAPE")
    viz.plot_radar(
        results,
        ["MSE", "MAE", "RMSE", "MAPE", "R2"],
        os.path.join(out_dir, f"{name}_radar.png"),
        title=f"{name} multi-metric",
    )
    viz.plot_horizon_curves(
        horizon_maes,
        os.path.join(out_dir, f"{name}_horizon_mae.png"),
        metric_name="MAE",
        title=f"{name} MAE vs horizon",
    )

    # ablation bar
    abl_names = [k for k in ["DGRN-NoGraph", "GraphESN", "DGRN-NoICS", "ICS-DGRN", "ICS-DESN"] if k in results]
    viz.plot_ablation(
        abl_names,
        [results[k]["MAE"] for k in abl_names],
        os.path.join(out_dir, f"{name}_ablation_mae.png"),
        ylabel="MAE",
        title=f"{name} ablation (graph / depth / ICS)",
    )

    df = pd.DataFrame(results).T.sort_values("MAE")
    viz.save_table(df, os.path.join(out_dir, f"{name}_metrics.csv"), os.path.join(out_dir, f"{name}_metrics.xlsx"))
    viz.plot_heatmap_metrics(
        df[["MSE", "MAE", "RMSE", "MAPE", "R2"]],
        os.path.join(out_dir, f"{name}_heatmap.png"),
        title=f"{name} metrics heatmap",
    )

    # spatial error map for ICS-DGRN
    if "ICS-DGRN" in preds:
        yt, yp = preds["ICS-DGRN"]
        err = np.mean(np.abs(denorm(yt) - denorm(yp)), axis=(0, 1))  # (N,)
        import matplotlib.pyplot as plt

        viz.set_style()
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(err, color="#c45911", lw=1.2)
        ax.set_title(f"{name} ICS-DGRN per-node MAE")
        ax.set_xlabel("Node")
        ax.set_ylabel("MAE")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_ICS-DGRN_node_mae.png"), bbox_inches="tight")
        plt.close(fig)

    print(df.round(4))
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["PEMS08", "PEMS04"])
    parser.add_argument("--max-train", type=int, default=600)
    parser.add_argument("--max-val", type=int, default=150)
    parser.add_argument("--max-test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULT_ROOT, exist_ok=True)
    frames = []
    for ds in args.datasets:
        print("=" * 60, ds)
        df = run_pems(
            ds,
            max_train=args.max_train,
            max_val=args.max_val,
            max_test=args.max_test,
            seed=args.seed,
        )
        df = df.copy()
        df["dataset"] = ds
        frames.append(df)

    big = pd.concat(frames)
    viz.save_table(
        big,
        os.path.join(RESULT_ROOT, "all_pems_metrics.csv"),
        os.path.join(RESULT_ROOT, "all_pems_metrics.xlsx"),
    )
    # cross-dataset MAE heatmap
    pivot = big.reset_index().pivot_table(index="index", columns="dataset", values="MAE")
    viz.plot_heatmap_metrics(pivot, os.path.join(RESULT_ROOT, "summary_mae_heatmap.png"), title="PEMS MAE summary")
    print("Saved to", RESULT_ROOT)

if __name__ == "__main__":
    main()
