# ICS-DGRN

图上的多层储层，用来做短时交通流预测。层和层之间用一个可训练的压缩把状态变短，循环权重每次前向都会按谱范数缩放到 `||W||_2 * ||S||_2 <= 0.95`，避免状态发散。读出是一个小 MLP，最后把当前观测值加回去。

对照模型是 STGCN、T-GCN、Graph WaveNet、AGCRN。都是按论文结构写的缩小版，隐层宽度基本是 32，不是官方仓库里的那一套权重。

作者 [ZengWenquan](https://github.com/chaosbull)

## 已有结果

`results/` 里的表是之前一次 20 epoch 跑出来的，三个数据集都跑完了。脚本现在最多训 1000 个 epoch，再大的数会被截断。权重写到 `results/<数据集>/*_模型名.pt`。

- 历史 12 步，预测 12 步
- 损失 Smooth L1，Adam，余弦退火到 1e-5，梯度裁剪 5
- ICS-DGRN 学习率 8e-4，其余 1e-3，weight decay 1e-4，batch 32
- 训练 / 验证 / 测试各随机抽 800 / 150 / 250 条，种子 42
- 用训练集的均值和标准差做标准化，表里的 MAE、RMSE、R2 是还原到原始流量之后算的

MAE 上 ICS-DGRN 三个集合都最低，参数和训练时间也最大。PEMS08 上和 STGCN 很接近（22.30 对 22.61），PEMS04 上拉开大约 1.2。

### PEMS03

| Model | MAE | RMSE | R2 | Train (s) |
| --- | ---: | ---: | ---: | ---: |
| ICS-DGRN | 20.13 | 30.86 | 0.950 | 195.9 |
| STGCN | 20.78 | 32.16 | 0.945 | 25.5 |
| AGCRN | 21.59 | 35.52 | 0.933 | 11.1 |
| GWNet | 22.03 | 33.33 | 0.941 | 28.5 |
| TGCN | 36.30 | 53.48 | 0.848 | 16.4 |

### PEMS04

| Model | MAE | RMSE | R2 | Train (s) |
| --- | ---: | ---: | ---: | ---: |
| ICS-DGRN | 26.93 | 41.41 | 0.932 | 175.3 |
| STGCN | 28.17 | 43.46 | 0.925 | 22.1 |
| AGCRN | 28.57 | 43.18 | 0.926 | 9.5 |
| GWNet | 28.65 | 43.86 | 0.924 | 24.6 |
| TGCN | 36.39 | 55.57 | 0.877 | 14.6 |

### PEMS08

| Model | MAE | RMSE | R2 | Train (s) |
| --- | ---: | ---: | ---: | ---: |
| ICS-DGRN | 22.30 | 34.38 | 0.943 | 119.0 |
| STGCN | 22.61 | 35.11 | 0.941 | 20.9 |
| AGCRN | 23.54 | 35.91 | 0.938 | 7.7 |
| GWNet | 23.63 | 36.14 | 0.937 | 15.4 |
| TGCN | 40.39 | 62.34 | 0.813 | 10.8 |

MAPE 在 csv 里有，但流量会出现 0，这个数没有参考价值，上面就没列。重跑时会再画曲线和散点，那些 png 不放进仓库。

## 环境

Python 3.9+。有 CUDA 会用 GPU，没有就走 CPU，会慢很多。

```bash
pip install -r requirements.txt
```

## 数据

npz 太大，没有放进仓库。用的是 [STSGCN](https://github.com/Davidham3/STSGCN) / ASTGCN 那套 PEMS03、PEMS04、PEMS08，放到 `data/`：

```
data/PEMS03/train.npz
data/PEMS03/val.npz
data/PEMS03/test.npz
data/PEMS03/adj_PEMS03.pkl
```

PEMS04、PEMS08 同样命名。`x` 和 `y` 的形状是 `(样本, 12, 节点, 通道)`，代码只用第 0 个通道。邻接矩阵是 DCRNN 的 pickle，矩阵在第三个元素。

## 重跑

在仓库根目录：

```bash
python experiments/run_compare.py --datasets PEMS08 PEMS04 PEMS03 --epochs 1000
```

`--epochs` 最大 1000。耐心和轮数一样，验证集还在降就会把设的轮数跑完。每个模型训完存一份 `.pt`，指标仍是 csv。

## 代码

```
model/ics_dgrn.py            ICS-DGRN
baselines/st_models.py       四个对照，以及训练、测试
data/pems.py                 读 npz
utils/metrics.py             MAE / RMSE / R2
utils/viz.py                 曲线和表
experiments/run_compare.py   入口
results/                     指标表；新跑的权重是 .pt
```

## 许可

[Apache License 2.0](LICENSE)。Copyright 2026 ZengWenquan.
