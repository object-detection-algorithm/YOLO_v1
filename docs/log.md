
# 日志

## 训练参数

* `S=7, B=2, C=3`
* 缩放至`(448, 448)`，进行数据标准化处理
* 优化器：`SGD`，学习率`1e-3`，动量大小`0.9`
* 衰减器：每隔`4`轮衰减`4%`，学习因子`0.96`

## 训练日志

```
$ python train.py 
Epoch 0/49
----------
train Loss: 4.3550
save model

Epoch 1/49
----------
train Loss: 3.4803
save model

Epoch 2/49
----------
train Loss: 3.3921
save model

Epoch 3/49
----------
train Loss: 3.0650
save model

Epoch 4/49
----------
train Loss: 2.9081
save model

Epoch 5/49
----------
train Loss: 2.5893
save model

Epoch 6/49
----------
train Loss: 2.5640
save model

Epoch 7/49
----------
train Loss: 2.4905
save model

Epoch 8/49
----------
train Loss: 2.1913
save model

Epoch 9/49
----------
train Loss: 2.1152
save model

Epoch 10/49
----------
train Loss: 1.9428
save model

Epoch 11/49
----------
train Loss: 1.7271
save model

Epoch 12/49
----------
train Loss: 1.4372
save model

Epoch 13/49
----------
train Loss: 1.8185

Epoch 14/49
----------
train Loss: 1.5460

Epoch 15/49
----------
train Loss: 1.1692
save model

Epoch 16/49
----------
train Loss: 1.0540
save model

Epoch 17/49
----------
train Loss: 0.9028
save model

Epoch 18/49
----------
train Loss: 0.7702
save model

Epoch 19/49
----------
train Loss: 0.7176
save model

Epoch 20/49
----------
train Loss: 0.7485

Epoch 21/49
----------
train Loss: 0.6307
save model

Epoch 22/49
----------
train Loss: 0.5581
save model

Epoch 23/49
----------
train Loss: 0.5320
save model

Epoch 24/49
----------
train Loss: 0.5893

Epoch 25/49
----------
train Loss: 0.5185
save model

Epoch 26/49
----------
train Loss: 0.6156

Epoch 27/49
----------
train Loss: 0.5096
save model

Epoch 28/49
----------
train Loss: 0.5403

Epoch 29/49
----------
train Loss: 0.4653
save model

Epoch 30/49
----------
train Loss: 0.3850
save model

Epoch 31/49
----------
train Loss: 0.3609
save model

Epoch 32/49
----------
train Loss: 0.4063

Epoch 33/49
----------
train Loss: 0.3349
save model

Epoch 34/49
----------
train Loss: 0.2629
save model

Epoch 35/49
----------
train Loss: 0.3319

Epoch 36/49
----------
train Loss: 0.2790

Epoch 37/49
----------
train Loss: 0.2487
save model

Epoch 38/49
----------
train Loss: 0.2325
save model

Epoch 39/49
----------
train Loss: 0.2146
save model

Epoch 40/49
----------
train Loss: 0.2087
save model

Epoch 41/49
----------
train Loss: 0.1626
save model

Epoch 42/49
----------
train Loss: 0.1446
save model

Epoch 43/49
----------
train Loss: 0.1372
save model

Epoch 44/49
----------
train Loss: 0.1260
save model

Epoch 45/49
----------
train Loss: 0.1231
save model

Epoch 46/49
----------
train Loss: 0.1232

Epoch 47/49
----------
train Loss: 0.1475

Epoch 48/49
----------
train Loss: 0.1226
save model

Epoch 49/49
----------
train Loss: 0.1000
save model

Training complete in 6m 28s
```

## 检测结果

```
compute mAP
{'cucumber': 63, 'mushroom': 61, 'eggplant': 62}
99.43% = cucumber AP 
98.39% = eggplant AP 
99.22% = mushroom AP 
mAP = 99.01%
```