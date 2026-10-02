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

"""ESP empirical verification utilities (2.pdf)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from model.dgrn import DGRNConfig, ICSDGRN

def simulate_state_difference(
    model: ICSDGRN,
    u_seq: np.ndarray,
    seed_a: int = 0,
    seed_b: int = 1,
) -> np.ndarray:
    """
    Drive two copies with identical input but different initial states.
    Returns Frobenius norm of state difference over time for the last layer.
    """
    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)

    model.reset()
    states_a = [rng_a.normal(0, 1, size=s.shape) for s in model.states]
    model.states = [s.copy() for s in states_a]

    # clone config/weights into second trajectory by reusing same matrices
    diffs = []
    # trajectory A
    hist_a = []
    for t in range(len(u_seq)):
        model.step(u_seq[t])
        hist_a.append([s.copy() for s in model.states])

    # trajectory B
    model.states = [rng_b.normal(0, 1, size=s.shape) for s in model.states]
    hist_b = []
    for t in range(len(u_seq)):
        model.step(u_seq[t])
        hist_b.append([s.copy() for s in model.states])

    for t in range(len(u_seq)):
        d = 0.0
        for sa, sb in zip(hist_a[t], hist_b[t]):
            d += np.linalg.norm(sa - sb) ** 2
        diffs.append(np.sqrt(d))
    return np.asarray(diffs, dtype=np.float64)

def verify_esp_condition(model: ICSDGRN) -> Dict:
    rep = model.esp_report()
    return rep
