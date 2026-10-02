# ICS-DGRN 补充实验结果说明（`result/pems_newexp`）

本目录与 `result/pems_st200`（主对比）、`result/pems`（Ridge/消融）相互独立。  
下面按子目录说明：**每个文件画了什么 / 表里每列是什么 / 怎么读**。

**实验设置**：PEMS08 为主（多 seed 另含 PEMS03/04）；hist=12 → horizon=12；300 epoch；SmoothL1；Adam+Cosine；子采样 800/150/250；默认 `layer_dims=[56,40]`，`compression_dims=[28]`，`esp_target=0.9`；conda `py312` + CUDA。

---

## 目录总览

```
result/pems_newexp/
  REPORT.md                 # 本说明
  compression_ratio/        # 层间压缩比敏感性
  sparse_density/           # 压缩矩阵稀疏度敏感性
  esp_state_decay/          # 状态差衰减（ESP 直接观察）
  esp_forecast_joint/       # 谱缩放 τ × 预测误差联合
  fixed_vs_trainable/       # 固定 / 可训练水库对比
  multi_seed/               # 多种子均值±标准差
  multi_horizon/            # 1–12 步误差增长
  compute_overhead/         # 参数量 / 时间 / 显存
  qualitative/              # 三场景预测曲线
```

---

## 1. `compression_ratio/` — 层间压缩比

固定水库层维 `[56,40]`，只改压缩维 \(d_{\mathrm{comp}}=\mathrm{round}(56\cdot r)\)，\(r\in\{1.0,0.8,0.6,0.4,0.2\}\)。

### 数值表

| 文件 | 含义 |
|------|------|
| `PEMS08_compression_metrics.csv` | 每个 \(r\) 一行：测试集 MSE/MAE/RMSE/MAPE/R²、训练时间、可训练参数、压缩维、ESP 相关量等 |
| `PEMS08_compression_esp.json` | 同上结果的 JSON 备份 |

**本次数值摘要**

| \(r\) | 压缩维 | MAE | RMSE | \(R^2\) | 可训练参数 |
|------:|-------:|----:|-----:|----------:|-----------:|
| 1.0 | 56 | 20.177 | 31.390 | 0.9526 | 41182 |
| 0.8 | 45 | 20.162 | 31.336 | 0.9528 | 40126 |
| 0.6 | 34 | 20.185 | 31.352 | 0.9527 | 39070 |
| 0.4 | 22 | 20.180 | 31.330 | 0.9528 | 37918 |
| 0.2 | 11 | 20.224 | 31.417 | 0.9525 | 36862 |

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_compression_curves.png` | 三张子图：横轴 \(r\)，纵轴分别为 MAE、RMSE、\(R^2\) | 看压缩变强时精度怎么变；本次最低 MAE 在 \(r=0.8\) |
| `PEMS08_compression_params_vs_mae.png` | 左轴：可训练参数随 \(r\)；右轴：MAE 随 \(r\) | 同一张图上看「参数变少」和「误差变大/变小」是否同步 |

---

## 2. `sparse_density/` — 压缩矩阵稀疏度

压缩维固定为 28，只改 \(\Phi\) 的非零比例 \(\rho\in\{1.0,0.5,0.2,0.1,0.05\}\)。

### 数值表

| 文件 | 含义 |
|------|------|
| `PEMS08_density_metrics.csv` | 每个 \(\rho\) 一行；列 `rho` 即稀疏度；另有 `phi_spec_norm_l0`（压缩矩阵谱范数） |

**本次数值摘要**

| \(\rho\) | MAE | RMSE | \(R^2\) | \(\|\Phi\|_2\) |
|--------:|----:|-----:|----------:|---------------:|
| 1.0 | 20.197 | 31.348 | 0.9527 | 1.97 |
| 0.5 | 20.255 | 31.508 | 0.9523 | 1.46 |
| 0.2 | 20.240 | 31.506 | 0.9523 | 1.07 |
| 0.1 | 20.301 | 31.608 | 0.9519 | 0.91 |
| 0.05 | 20.353 | 31.665 | 0.9518 | 0.72 |

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_density_curves.png` | 横轴 \(\rho\)，纵轴 MAE / RMSE / \(R^2\) | \(\rho\) 变小 = 更稀疏；看误差是否上升 |
| `PEMS08_density_params_vs_mae.png` | 参数量与 MAE 双轴 | 本实验 Parameter 计数不随 mask 减少时，参数曲线可能几乎水平，主要看 MAE |

---

## 3. `esp_state_decay/` — 状态差衰减（ESP 直接实验）

同一段输入、两个不同随机初态，记录每层状态差随时间变化。  
\(\hat D_l(t)=\|X_a-X_b\|_F\big/\|X_a(0)-X_b(0)\|_F\)。纵轴一般用对数。

谱缩放 \(\tau\in\{0.5,0.7,0.9,1.0,1.1\}\)（即 `esp_target`）。

### 数值表 / JSON

| 文件 | 含义 |
|------|------|
| `PEMS08_esp_decay_summary.csv` / `.json` | 每个 \(\tau\) 一行：`max_kappa`（\(\max_l\|S\|_2\|W_l\|_2\)）、`esp_ok`、各层 late/early 比、终态 \(\hat D\)、衰减强弱标签 |
| `PEMS08_tau{τ}_layer{k}_Dhat.csv` | 三列：`t`, `D_hat`, `D`。第 \(k\) 层在时刻 \(t\) 的归一化 / 未归一化状态差 |

**本次数值摘要**

| \(\tau\) | \(\max\kappa\) | esp_ok | L1 late/early | L2 late/early | 终态 \(\hat D\) | 标签 |
|--------:|---------------:|:------:|--------------:|--------------:|----------------:|:----:|
| 0.5 | 0.500 | True | 0.0076 | 0.0232 | 0.0007 | Strong |
| 0.7 | 0.714 | True | 0.0099 | 0.0317 | 0.0012 | Strong |
| 0.9 | 0.897 | True | 0.0143 | 0.0448 | 0.0022 | Strong |
| 1.0 | 1.048 | False | 0.0164 | 0.0446 | 0.0023 | Moderate |
| 1.1 | 1.088 | False | 0.0190 | 0.0560 | 0.0032 | Moderate |

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_tau{τ}_state_decay.png` | 横轴时间步 \(t\)，纵轴 \(\hat D_l(t)\)（log）；曲线 = Layer 1 / Layer 2 | 曲线往下掉 = 初态影响消失；掉得越快越符合回声稳定 |
| `PEMS08_state_decay_vs_tau.png` | 只画**最后一层**；多条曲线对应不同 \(\tau\) | 同一张图比较：\(\tau\) 越大，衰减通常越慢、终值越高 |

---

## 4. `esp_forecast_joint/` — 谱缩放 × 预测

对每个 \(\tau\) 完整训练 ICS-DGRN，同时记录 \(\kappa\)、状态衰减标签、测试 MAE/RMSE、12 步误差与 \(G_{\mathrm{horizon}}\)。

### 数值表

| 文件 | 含义 |
|------|------|
| `PEMS08_esp_forecast_table.csv` | 短表：τ、max_kappa、StateDecay、MAE、RMSE、R²、MAE_h12、G_horizon、esp_ok |
| `PEMS08_esp_forecast_full.csv` | 完整指标（含训练时间、参数等） |
| `PEMS08_tau{τ}_horizon_mae.csv` | 两列：`h`（1..12）、该步 MAE |

**本次数值摘要**

| \(\tau\) | \(\max\kappa\) | StateDecay | MAE | MAE\(_{12}\) | \(G_{\mathrm{horizon}}\) |
|--------:|---------------:|:----------:|----:|-------------:|-------------------------:|
| 0.5 | 0.511 | Strong | 20.221 | 25.673 | 0.827 |
| 0.7 | 0.715 | Strong | 20.178 | 25.575 | 0.823 |
| 0.9 | 0.919 | Strong | 20.168 | 25.556 | 0.823 |
| 1.0 | 1.022 | Moderate | 20.177 | 25.601 | 0.828 |
| 1.1 | 1.124 | Moderate | 20.158 | 25.584 | 0.830 |

\(G_{\mathrm{horizon}}=(MAE_{12}-MAE_1)/MAE_1\)。

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_esp_forecast_joint.png` | 左：\(\tau\) vs \(\max\kappa\)（灰虚线 = 1）；右：\(\tau\) vs 测试 MAE | 左看理论收缩量是否 <1；右看预测误差随谱缩放怎么变 |

---

## 5. `fixed_vs_trainable/` — 固定 vs 可训练水库

| 变体 | 含义 |
|------|------|
| FixedReservoir | 稀疏 \(W\) 固定，仅训 Win / Φ / leak / 读出等（默认 ICS-DGRN） |
| TrainableReservoir | recurrent \(W\) 也参与梯度更新 |

### 数值表

| 文件 | 含义 |
|------|------|
| `PEMS08_fixed_vs_trainable.csv` | 两行：MAE/RMSE/R²、TrainableParams、TotalParams、TrainTimeSec、PeakGPUMemMB、esp_ok 等 |

**本次数值摘要**

| variant | MAE | RMSE | 可训练参数 | 总参数 | 训练(s) | 峰值显存(MB) | esp_ok |
|---------|----:|-----:|-----------:|-------:|--------:|-------------:|:------:|
| Fixed | 20.168 | 31.354 | 38494 | 49534 | 252.4 | 197.9 | True |
| Trainable | 20.010 | 31.106 | 43230 | 49534 | 262.2 | 198.4 | False |

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_fixed_vs_trainable.png` | 三柱图：MAE、可训练参数、训练时间 | 左看精度；中看要优化多少参数；右看墙钟时间 |

---

## 6. `multi_seed/` — 多种子

ICS-DGRN：PEMS03/04/08 各 3 个 seed（42,43,44）。  
AGCRN：仅 PEMS08 同样 3 个 seed（与主表近并列对照）。

### 数值表

| 文件 | 含义 |
|------|------|
| `all_seed_runs.csv` | 每一次独立训练一行（含 seed） |
| `multi_seed_summary.csv` | 按 dataset×model 聚合：MAE/RMSE/R² 的 mean 与 std |
| `REPORT_snippet.md` | 从 summary 生成的一行行 `mean±std` 文本 |

**本次数值摘要**

| dataset | model | MAE | RMSE | \(R^2\) |
|---------|-------|----:|-----:|--------:|
| PEMS03 | ICS-DGRN | 18.764 ± 0.246 | 29.483 ± 0.149 | 0.9560 ± 0.0022 |
| PEMS04 | ICS-DGRN | 25.014 ± 0.192 | 38.258 ± 0.378 | 0.9419 ± 0.0012 |
| PEMS08 | ICS-DGRN | 20.261 ± 0.084 | 31.219 ± 0.389 | 0.9539 ± 0.0012 |
| PEMS08 | AGCRN | 20.335 ± 0.173 | 31.138 ± 0.247 | 0.9542 ± 0.0013 |

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_multiseed_mae.png` | 柱 = MAE 均值，误差棒 = 标准差；ICS-DGRN 与 AGCRN | 看均值差和误差棒是否重叠 |
| `PEMS04_multiseed_mae.png` | 仅 ICS-DGRN | 该集上的波动大小 |
| `PEMS03_multiseed_mae.png` | 仅 ICS-DGRN | 同上 |

---

## 7. `multi_horizon/` — 多步预测误差增长

在 PEMS08 上训 ICS-DGRN + STGCN/TGCN/GWNet/AGCRN，按预测步 \(h=1..12\) 分别算 MAE。

### 数值表

| 文件 | 含义 |
|------|------|
| `PEMS08_horizon_mae_detail.csv` | 行 = horizon 1..12，列 = 各模型该步 MAE |
| `PEMS08_horizon_growth.csv` | 每个模型一行：整体 MAE、MAE_h1、MAE_h12、DeltaMAE、G_horizon |

**指标定义**

- \(\Delta MAE = MAE_{12}-MAE_1\)
- \(G_{\mathrm{horizon}}=(MAE_{12}-MAE_1)/MAE_1\)

**本次数值摘要**

| model | MAE | MAE\(_1\) | MAE\(_{12}\) | \(\Delta\)MAE | \(G\) |
|-------|----:|----------:|-------------:|--------------:|------:|
| ICS-DGRN | 20.168 | 14.017 | 25.556 | 11.540 | 0.823 |
| AGCRN | 20.245 | 14.444 | 25.980 | 11.536 | 0.799 |
| STGCN | 21.066 | 13.957 | 27.874 | 13.917 | 0.997 |
| TGCN | 23.749 | 16.559 | 30.865 | 14.305 | 0.864 |
| GWNet | 21.420 | 14.074 | 28.402 | 14.328 | 1.018 |

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_horizon_mae.png` | 横轴 \(h=1..12\)，纵轴该步 MAE；每条线一个模型 | 曲线越往上 = 远步误差越大；比绝对高度和斜率 |
| `PEMS08_G_horizon_bar.png` | 各模型的 \(G_{\mathrm{horizon}}\) 柱状图 | 只比「相对增长幅度」，不看绝对 MAE |

---

## 8. `compute_overhead/` — 参数 / 时间 / 显存

同一协议下五个模型各训一遍，记录可训练参数、总参数、训练时间、峰值 GPU 显存。

### 数值表

| 文件 | 含义 |
|------|------|
| `PEMS08_compute_overhead.csv` | 每模型一行：MAE、TrainableParams、TotalParams、TrainTimeSec、PeakGPUMemMB 等 |

**本次数值摘要**

| model | MAE | 可训练参数 | 总参数 | 训练(s) | 峰值显存(MB) |
|-------|----:|-----------:|-------:|--------:|-------------:|
| ICS-DGRN | 20.168 | 38494 | 49534 | 259.2 | 198.6 |
| STGCN | 21.067 | 25740 | 25740 | 322.3 | 897.3 |
| TGCN | 23.749 | 3564 | 3564 | 162.7 | 86.3 |
| GWNet | 21.417 | 12876 | 12876 | 212.4 | 181.0 |
| AGCRN | 20.245 | 4156 | 4156 | 122.7 | 55.0 |

### 图

| 文件 | 图上画什么 | 怎么读 |
|------|------------|--------|
| `PEMS08_compute_overhead.png` | 三子图柱状：可训练参数、训练秒数、峰值显存(MB) | 左=模型大小；中=墙钟；右=GPU 占用峰值 |

---

## 9. `qualitative/` — 三场景预测曲线

在测试集 node0、horizon=1 上，自动选取三段窗口，对比 Ground Truth / ICS-DGRN / AGCRN。

| 文件 | 图上画什么 | 场景怎么选的 |
|------|------------|--------------|
| `PEMS08_scenario_A_normal.png` | 蓝=真值，橙=ICS-DGRN，绿虚线=AGCRN；约 80 步窗口 | 滑动窗口内流量变化最小（较平稳） |
| `PEMS08_scenario_B_rapid_rise.png` | 同上 | 窗口内正向增量最大（快速上升） |
| `PEMS08_scenario_C_rapid_fall.png` | 同上 | 窗口内负向增量最大（快速下降） |
| `PEMS08_scenarios.json` | `start`/`end` 索引 | 三段窗口在测试序列中的起止下标 |

怎么读：看峰谷是否跟上、有没有明显滞后或过冲；只作定性对照。

---

## 10. 根目录其它文件

| 文件 | 含义 |
|------|------|
| `REPORT.md` | 本说明文档 |

---

## 常用列名速查（CSV 通用）

| 列名 | 含义 |
|------|------|
| MAE / RMSE / R2 / MSE / MAPE | 反标准化后的测试集指标（MAPE 在近零流量上会很大，慎用） |
| TrainTimeSec / InferTimeSec | 训练墙钟 / 测试推理秒数 |
| TrainableParams / TotalParams | 可训练参数 / 含 buffer 的总浮点参数约数 |
| PeakGPUMemMB | `torch.cuda.max_memory_allocated` 峰值（MB） |
| MAE_h1 / MAE_h12 | 仅第 1 / 第 12 预测步的 MAE |
| DeltaMAE / G_horizon | 见 §7 |
| max_kappa / esp_ok / esp_target | \(\max_l\|S\|_2\|W\|_2\)；是否 <1；设定的 \(\tau\) |
| compression_ratio / compression_dim | \(r\) 与实际压缩维 |
| rho / phi_density | 压缩矩阵非零比例 |
| train_recurrent | `True` = \(W\) 可训练 |
| seed | 随机种子 |
