
# YOLO模型

论文给出了一个`24`层卷积的`CNN`模型

![](./imgs/figure-3.png)

## 实现

* `py/lib/models/basic_conv2d.py`
* `py/lib/models/yolo_v1.py`

模型输入`4`维数据（大小为`[N, C, H, W]`）；输出`3`维数据（大小为`[N, S*S, B*5+C]`)