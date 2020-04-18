# -*- coding: utf-8 -*-

"""
@date: 2020/4/18 上午11:19
@file: draw.py
@author: zj
@description: 
"""

import cv2
import matplotlib.pyplot as plt


def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title('loss')
    plt.savefig('./loss.png')
