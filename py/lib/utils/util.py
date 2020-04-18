# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午7:31
@file: util.py
@author: zj
@description: 
"""

import numpy as np


def iou(pred_box, target_box):
    """
    计算候选建议和标注边界框的IoU
    :param pred_box: 大小为[4]
    :param target_box: 大小为[N, 4]
    :return: [N]
    """
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]

    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * (target_box[:, 3] - target_box[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores


def compute_ious(rects, bndboxs):
    iou_list = list()
    for rect in rects:
        scores = iou(rect, bndboxs)
        iou_list.append(max(scores))
    return iou_list


def parse_output(outputs, S, B, C):
    """
    每个网格保存置信度最高的检测边界框
    :param outputs: (N, S*S, B*5+C)
    :return: probs, bboxs
    probs: (N, S*S, C)
    bboxs: (N, S*S, 5)
    """
    N = outputs.shape[0]

    probs = outputs[:, :, :C]
    confidences = outputs[:, :, C:(C + B)].reshape(-1, B)
    bboxs = outputs[:, :, (C + B):].reshape(-1, 4 * B)

    idxs = torch.argmax(confidences, dim=1)

    probs *= confidences[range(len(idxs)), idxs]
    obj_boxs = bboxs[range(len(idxs)), idxs]
    return probs, obj_boxs.reshape(N, S * S, -1)
