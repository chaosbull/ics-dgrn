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

"""Generate synthetic benchmarks used by ICS-DESN paper (logistic / Lorenz / sunspot)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(__file__)
DS = os.path.join(ROOT, "datasets")

def logistic_map(n: int = 5000, r: float = 4.0, x0: float = 0.1, warmup: int = 200) -> np.ndarray:
    x = x0
    for _ in range(warmup):
        x = r * x * (1 - x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        x = r * x * (1 - x)
        out[i] = x
    return out

def lorenz_system(
    n: int = 10000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    warmup: int = 1000,
) -> np.ndarray:
    """Return (n, 3) trajectory of x,y,z."""
    x, y, z = 1.0, 1.0, 1.0
    for _ in range(warmup):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x, y, z = x + dt * dx, y + dt * dy, z + dt * dz
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x, y, z = x + dt * dx, y + dt * dy, z + dt * dz
        out[i] = (x, y, z)
    return out

def synthetic_sunspot(n: int = 3000, seed: int = 0) -> np.ndarray:
    """
    Approximate 11-year solar cycle + noise (placeholder when SILSO csv unavailable).
    Amplitude-modulated sinusoid; good for smoke tests, not astronomy research.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cycle = 11.0 * 12.0  # monthly-like
    base = 60 + 50 * np.sin(2 * np.pi * t / cycle)
    second = 10 * np.sin(2 * np.pi * t / (cycle / 2))
    noise = rng.normal(0, 8, size=n)
    return np.clip(base + second + noise, 0, None)

def generate_all(seed: int = 2025) -> None:
    os.makedirs(DS, exist_ok=True)

    logi = logistic_map(n=5000)
    pd.DataFrame({"value": logi}).to_csv(os.path.join(DS, "logistic.csv"), index=False)

    lor = lorenz_system(n=10000)
    pd.DataFrame(lor, columns=["x", "y", "z"]).to_csv(os.path.join(DS, "lorenz.csv"), index=False)

    sun = synthetic_sunspot(n=3000, seed=seed)
    pd.DataFrame({"sunspot": sun}).to_csv(os.path.join(DS, "sunspot.csv"), index=False)

    print("Wrote logistic.csv, lorenz.csv, sunspot.csv ->", DS)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    generate_all(seed=args.seed)

if __name__ == "__main__":
    main()
