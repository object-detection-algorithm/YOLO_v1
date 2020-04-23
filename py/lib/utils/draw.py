# -*- coding: utf-8 -*-

"""
@date: 2020/4/18 上午11:19
@file: draw.py
@author: zj
@description: 
"""

import copy
import numpy as np
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


def plot_bboxs(img, bndboxs, name_list, cate_list, pred_boxs, pred_cates, pred_probs):
    dst = copy.deepcopy(img)

    for i in range(len(name_list)):
        bndbox = bndboxs[i]
        name = name_list[i]

        xmin, ymin, xmax, ymax = bndbox
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=1)
        cv2.putText(dst, name, (xmin, ymax), 1, cv2.FONT_HERSHEY_PLAIN, (255, 0, 0), thickness=1)

    for i in range(len(pred_probs)):
        cate = pred_cates[i]
        prob = pred_probs[i]
        bbox = pred_boxs[i]

        if prob < 0.5:
            continue

        xmin, ymin, xmax, ymax = np.array(bbox, dtype=np.int)
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
        cv2.putText(dst, '%s_%.3f' % (cate_list[cate], prob), (xmin, ymin), 1, cv2.FONT_HERSHEY_PLAIN, (0, 0, 255),
                    thickness=1)

    return dst


def show(img, title=None):
    fig = plt.figure()

    if title:
        plt.title(title)
    plt.imshow(img)
    plt.show()
