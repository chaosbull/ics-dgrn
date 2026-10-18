"""Single-layer Graph Echo State Network baseline (no depth / no ICS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .dgrn import DGRNConfig, ICSDGRN


@dataclass
class GraphESNConfig(DGRNConfig):
    layer_dims: tuple = (32,)
    compression_dims: tuple = ()


class GraphESN(ICSDGRN):
    """Graph reservoir with a single layer (ablation of depth & compression)."""

    def __init__(
        self,
        n_nodes: int,
        n_in_feat: int,
        n_outputs: int,
        adj: np.ndarray,
        cfg: Optional[GraphESNConfig] = None,
    ):
        cfg = cfg or GraphESNConfig()
        cfg.layer_dims = (cfg.layer_dims[0],) if cfg.layer_dims else (32,)
        cfg.compression_dims = ()
        super().__init__(n_nodes, n_in_feat, n_outputs, adj, cfg=cfg)
