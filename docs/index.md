# YOLO_v1

实现`YOLO_v1`算法

## 数据集

使用`3`类定位数据集，参考[[数据集]Image Localization Dataset](https://blog.zhujian.life/posts/a2d65e1.html)

```
{'cucumber': 63, 'mushroom': 61, 'eggplant': 62}
```

## 实现流程

1. 创建训练数据集
2. 定义`YoLo_v1`模型
3. 定义损失函数`Multi-part Loss`
4. 训练模型
5. 计算`mAP`