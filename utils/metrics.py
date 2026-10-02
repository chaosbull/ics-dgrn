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

"""Evaluation metrics (ICS-DESN style: MSE, MAE, RMSE, MAPE, R2)."""

from __future__ import annotations

from typing import Dict

import numpy as np

def _as_1d(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return yt, yp

def mse(y_true, y_pred) -> float:
    yt, yp = _as_1d(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))

def mae(y_true, y_pred) -> float:
    yt, yp = _as_1d(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def mape(y_true, y_pred, eps: float = 1e-5) -> float:
    yt, yp = _as_1d(y_true, y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)

def r2_score(y_true, y_pred) -> float:
    yt, yp = _as_1d(y_true, y_pred)
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

def masked_mae(y_true, y_pred, null_val: float = 0.0) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = yt != null_val
    mask = mask.astype(np.float64)
    mask /= np.mean(mask) + 1e-12
    loss = np.abs(yt - yp) * mask
    return float(np.mean(loss))
