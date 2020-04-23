
# 07 VOC训练日志

## 训练参数

* `S=7, B=2, C=3`
* 缩放至`(448, 448)`，进行数据标准化处理
* 优化器：`SGD`，学习率`1e-3`，动量大小`0.9`
* 衰减器：每隔`4`轮衰减`4%`，学习因子`0.96`

## 检测结果

```
。。。
```

## 训练日志

```
$ python train.py 
Epoch 0/49
----------
train Loss: 9.3544
save model

Epoch 1/49
----------
train Loss: 7.4459
save model

Epoch 2/49
----------
train Loss: 7.1511
save model

Epoch 3/49
----------
train Loss: 6.9450
save model

Epoch 4/49
----------
train Loss: 6.7656
save model

Epoch 5/49
----------
train Loss: 6.6077
save model

Epoch 6/49
----------
train Loss: 6.4264
save model

Epoch 7/49
----------
train Loss: 6.2750
save model

Epoch 8/49
----------
train Loss: 6.0318
save model

Epoch 9/49
----------
train Loss: 5.7777
save model

Epoch 10/49
----------
train Loss: 5.4760
save model

Epoch 11/49
----------
train Loss: 5.1784
save model

Epoch 12/49
----------
train Loss: 4.8067
save model

Epoch 13/49
----------
train Loss: 4.4603
save model

Epoch 14/49
----------
train Loss: 4.1291
save model

Epoch 15/49
----------
train Loss: 3.7810
save model

Epoch 16/49
----------
train Loss: 3.4259
save model

Epoch 17/49
----------
train Loss: 3.1166
save model

Epoch 18/49
----------
train Loss: 2.8041
save model

Epoch 19/49
----------
train Loss: 2.4853
save model

Epoch 20/49
----------
train Loss: 2.2107
save model

Epoch 21/49
----------
train Loss: 1.9424
save model

Epoch 22/49
----------
train Loss: 1.7307
save model

Epoch 23/49
----------
train Loss: 1.5518
save model

Epoch 24/49
----------
train Loss: 1.3712
save model

Epoch 25/49
----------
train Loss: 1.2174
save model

Epoch 26/49
----------
train Loss: 1.1142
save model

Epoch 27/49
----------
train Loss: 1.0305
save model

Epoch 28/49
----------
train Loss: 0.9270
save model

Epoch 29/49
----------
train Loss: 0.8465
save model

Epoch 30/49
----------
train Loss: 0.7739
save model

Epoch 31/49
----------
train Loss: 0.7228
save model

Epoch 32/49
----------
train Loss: 0.6765
save model

Epoch 33/49
----------
train Loss: 0.6264
save model

Epoch 34/49
----------
train Loss: 0.5932
save model

Epoch 35/49
----------
train Loss: 0.5646
save model

Epoch 36/49
----------
train Loss: 0.5359
save model

Epoch 37/49
----------
train Loss: 0.4941
save model

Epoch 38/49
----------
train Loss: 0.4682
save model

Epoch 39/49
----------
train Loss: 0.4482
save model

Epoch 40/49
----------
train Loss: 0.4276
save model

Epoch 41/49
----------
train Loss: 0.4048
save model

Epoch 42/49
----------
train Loss: 0.3874
save model

Epoch 43/49
----------
train Loss: 0.3759
save model

Epoch 44/49
----------
train Loss: 0.3607
save model

Epoch 45/49
----------
train Loss: 0.3415
save model

Epoch 46/49
----------
train Loss: 0.3327
save model

Epoch 47/49
----------
train Loss: 0.3222
save model

Epoch 48/49
----------
train Loss: 0.3141
save model

Epoch 49/49
----------
train Loss: 0.2978
save model

Training complete in 214m 30s
```
