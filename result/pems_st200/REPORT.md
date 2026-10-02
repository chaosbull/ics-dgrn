# 300-epoch DL comparison

Models: ICS-DGRN(v3), STGCN, TGCN, GWNet, AGCRN
Epochs (max): 300, loss: SmoothL1, optimizer: Adam + CosineAnnealing, device: CUDA auto

## PEMS08

              MAE     RMSE      R2  Epochs  TrainTimeSec
ICS-DGRN  20.2443  31.5044  0.9523     300       300.886
STGCN     21.0574  33.0564  0.9474     300       343.317
TGCN      23.7485  36.1055  0.9373     300       246.900
GWNet     21.4218  33.5818  0.9458     300       245.257
AGCRN     20.2452  31.3812  0.9526     300       155.875

- MAE gap (STGCN - ICS-DGRN) = +0.8131
- MAE gap (AGCRN - ICS-DGRN) = +0.0009

## PEMS04

              MAE     RMSE      R2  Epochs  TrainTimeSec
ICS-DGRN  25.1387  38.7516  0.9403     300       424.061
STGCN     26.7156  41.3850  0.9320     300       363.407
TGCN      30.4626  46.2726  0.9149     300       359.199
GWNet     26.7423  41.4624  0.9317     300       392.574
AGCRN     25.3343  39.0147  0.9395     300       196.853

- MAE gap (STGCN - ICS-DGRN) = +1.5769
- MAE gap (AGCRN - ICS-DGRN) = +0.1956

## PEMS03

              MAE     RMSE      R2  Epochs  TrainTimeSec
ICS-DGRN  18.6707  29.7050  0.9532     300       434.009
STGCN     19.2406  30.2605  0.9515     300       388.562
TGCN      25.1424  37.5139  0.9254     300       296.746
GWNet     19.3259  30.3540  0.9512     300       435.390
AGCRN     18.6019  30.9962  0.9491     300       213.377

- MAE gap (STGCN - ICS-DGRN) = +0.5699
- MAE gap (AGCRN - ICS-DGRN) = -0.0688
