# PEMS comparison, 20 epoch

ICS-DGRN, STGCN, TGCN, GWNet, AGCRN.
Smooth L1, Adam, cosine decay. Numbers below are on the original flow scale.
Train / val / test subsample: 800 / 150 / 250, seed 42.

MAPE is in the csv files. Traffic flow hits zero, so that column is not useful.

## PEMS08

              MAE     RMSE      R2  Epochs  TrainTimeSec
ICS-DGRN  22.2977  34.3800  0.9431      20      118.9910
STGCN     22.6077  35.1140  0.9407      20       20.8569
TGCN      40.3915  62.3364  0.8131      20       10.7682
GWNet     23.6256  36.1385  0.9372      20       15.3917
AGCRN     23.5417  35.9078  0.9380      20        7.6538

STGCN MAE minus ICS-DGRN MAE = +0.3100

## PEMS04

              MAE     RMSE      R2  Epochs  TrainTimeSec
ICS-DGRN  26.9263  41.4102  0.9319      20      175.2780
STGCN     28.1653  43.4602  0.9250      20       22.1423
TGCN      36.3868  55.5721  0.8773      20       14.6093
GWNet     28.6480  43.8617  0.9236      20       24.5691
AGCRN     28.5696  43.1754  0.9259      20        9.5120

STGCN MAE minus ICS-DGRN MAE = +1.2390

## PEMS03

              MAE     RMSE      R2  Epochs  TrainTimeSec
ICS-DGRN  20.1309  30.8583  0.9496      20      195.9240
STGCN     20.7771  32.1621  0.9452      20       25.4625
TGCN      36.2967  53.4842  0.8484      20       16.3639
GWNet     22.0316  33.3262  0.9412      20       28.5056
AGCRN     21.5890  35.5209  0.9332      20       11.0987

STGCN MAE minus ICS-DGRN MAE = +0.6462
