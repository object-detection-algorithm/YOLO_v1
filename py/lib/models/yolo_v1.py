# -*- coding: utf-8 -*-

"""
@date: 2020/4/16 下午12:50
@file: yolo_v1.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from models.basic_conv2d import BasicConv2d


class YOLO_v1(nn.Module):

    def __init__(self):
        super(YOLO_v1, self).__init__()
        conv_block = BasicConv2d

        self.features = nn.Sequential(
            conv_block(3, 64, kernel_size=7, stride=2),
            # reduction
            nn.MaxPool2d(2, stride=2),
            conv_block(64, 192, kernel_size=3, padding=1),
            # reduction
            nn.MaxPool2d(2, stride=2),
            conv_block(192, 128, kernel_size=1),
            conv_block(128, 256, kernel_size=3, padding=1),
            conv_block(256, 256, kernel_size=1),
            conv_block(256, 512, kernel_size=3, padding=1),
            # reduction
            nn.MaxPool2d(2, stride=2),
            conv_block(512, 256, kernel_size=1),
            conv_block(256, 512, kernel_size=3, padding=1),
            conv_block(512, 256, kernel_size=1),
            conv_block(256, 512, kernel_size=3, padding=1),
            conv_block(512, 256, kernel_size=1),
            conv_block(256, 512, kernel_size=3, padding=1),
            conv_block(512, 256, kernel_size=1),
            conv_block(256, 512, kernel_size=3, padding=1),
            conv_block(512, 512, kernel_size=1),
            conv_block(512, 1024, kernel_size=3, padding=1),
            # reduction
            nn.MaxPool2d(2, stride=2),
            conv_block(1024, 512, kernel_size=1),
            conv_block(512, 1024, kernel_size=3, padding=1),
            conv_block(1024, 512, kernel_size=1),
            conv_block(512, 1024, kernel_size=3, padding=1),
            conv_block(1024, 1024, kernel_size=3, padding=1),
            # reduction
            conv_block(1024, 1024, kernel_size=3, stride=2),
            conv_block(1024, 1024, kernel_size=3, padding=1),
            conv_block(1024, 1024, kernel_size=3, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 7 * 7 * 30)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    data = torch.randn((1, 3, 448, 448))
    # data = torch.randn((1, 3, 224, 224))
    model = YOLO_v1()

    outputs = model(data)
    print(outputs.shape)
