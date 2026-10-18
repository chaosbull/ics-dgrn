# Mechanism checks

These files use the same ICS-DGRN configuration as `result/pems_st200` (`layer_dims=[48, 36, 24]`, `compression_dims=[24, 16]`, `esp_target=0.9`) unless a row says otherwise. They are not the older 300-epoch single-seed sweep.

## Ablation (`ablation/`)

PEMS08, three seeds. Mean test MAE:

| Variant | MAE | std |
|---------|----:|----:|
| Full ICS-DGRN | 21.641 | 0.237 |
| No graph propagation | 21.615 | 0.185 |
| Rewired graph | 21.424 | 0.090 |
| No compression | 21.605 | 0.152 |
| Trainable reservoir | 21.547 | 0.124 |
| Single layer | 21.874 | 0.177 |

On this split, the single-layer model is worse than the full model. Dropping graph propagation or compression does not raise MAE. A degree-preserving rewired graph is not worse than the road graph here. Figure: `ablation/F_ablation.png`.

## Compression ratio (`compression_ratio/`)

PEMS08 sweep of the interlayer compression ratio. MAE stays between 22.31 (`ratio=0.25`) and 22.56 (`ratio=0.75`). Figure: `compression_ratio/F_compression.png`.

## ESP (`esp_state_decay/`)

State-difference diagnostic versus spectral scale `tau`. `esp_ok` is true for `tau` in {0.5, 0.7, 0.9} and false for 1.05 and 1.3. The recorded state difference at the end of the trace increases with `tau` (about 0.0034 at 0.5 to 0.019 at 1.3). Forecast MAE in this short sweep stays near 23.1 and does not by itself show that violating the sufficient condition hurts accuracy. Figure: `esp_state_decay/F_esp.png`.

## Efficiency (`compute_overhead/`)

`F_efficiency_bars.png` and `F_efficiency_scatter.png` plot training time, parameter counts, and error for the main comparison. Read them together with `result/pems_st200/main_summary.csv`.

## Horizon (`multi_horizon/`)

`F_horizon.png` is the 1–12 step MAE curve for the main models.

## Other tables

| File | Content |
|------|---------|
| `block_bootstrap_vs_agcrn.csv` | Block-bootstrap MAE gap of ICS-DGRN versus AGCRN, per dataset and seed |
| `ridge_graph_ablation_summary.csv` | Ridge reservoir with the real graph, no graph, and a rewired graph |

Scenario curves from the previous release were not recomputed with `layer_dims=[48, 36, 24]`, so they are not included.

Ridge graph ablation (mean MAE): removing the graph, or rewiring it, does not produce a consistent loss. On PEMS08, rewired 21.88, no-graph 22.22, full ridge 22.28.
