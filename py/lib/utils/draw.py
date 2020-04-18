# -*- coding: utf-8 -*-

"""
@date: 2020/4/18 上午11:19
@file: draw.py
@author: zj
@description: 
"""

import copy
import cv2
import matplotlib.pyplot as plt


def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title('loss')
    plt.savefig('./loss.png')


def plot_box(img, bndboxs, name_list):
    dst = copy.deepcopy(img)

    for i in range(len(name_list)):
        bndbox = bndboxs[i]
        name = name_list[i]

        xmin, ymin, xmax, ymax = bndbox
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
        cv2.putText(dst, name, (xmin, ymax), 1, cv2.FONT_HERSHEY_PLAIN, (255, 0, 0), thickness=1)

    return dst


def show(img, title=None):
    fig = plt.figure()

    if title:
        plt.title(title)
    plt.imshow(img)
    plt.show()
