# Copyright 2026 ZengWenquan
# https://github.com/chaosbull
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict

import numpy as np


def _flat(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return yt, yp


def mse(y_true, y_pred) -> float:
    yt, yp = _flat(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def mae(y_true, y_pred) -> float:
    yt, yp = _flat(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-5) -> float:
    # blows up when the true flow is 0, so don't lean on this number
    yt, yp = _flat(y_true, y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def r2_score(y_true, y_pred) -> float:
    yt, yp = _flat(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def evaluate_all(y_true, y_pred) -> Dict[str, float]:
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
