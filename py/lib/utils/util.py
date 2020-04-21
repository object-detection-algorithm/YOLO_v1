# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午7:31
@file: util.py
@author: zj
@description: 
"""

import numpy as np
import torch
import sys


def error(msg):
    print(msg)
    sys.exit(0)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    intersection = np.maximum(0.0, xB - xA + 1) * np.maximum(0.0, yB - yA + 1)
    # 计算两个边界框面积
    boxAArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    boxBArea = (target_box[:, 2] - target_box[:, 0] + 1) * (target_box[:, 3] - target_box[:, 1] + 1)

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
    :return: cates, probs, bboxs
    cates: (N, S*S)
    probs: (N, S*S)
    bboxs: (N, S*S, 4)
    """
    N = outputs.shape[0]

    # (N*S*S, C)
    probs = outputs[:, :, :C].reshape(-1, C)
    # (N*S*S, B)
    confidences = outputs[:, :, C:(C + B)].reshape(-1, B)
    # (N*S*S, 4*B)
    bboxs = outputs[:, :, (C + B):].reshape(-1, 4 * B)

    # 计算每个网格所属类别 (N*S*S)
    cates = torch.argmax(probs, dim=1)
    # 计算每个网格最高置信度 (N*S*S)
    idxs = torch.argmax(confidences, dim=1)

    # 计算分类概率 (N*S*S)
    cate_probs = probs[range(len(cates)), cates] * confidences[range(len(idxs)), idxs]
    # 计算对应边界框坐标 (N*S*S, 4)
    obj_boxs = bboxs[range(len(idxs)), idxs * 4: (idxs + 1) * 4]

    return cates.reshape(N, S * S), cate_probs(N, S * S), obj_boxs(N, S * S, 4)


def bbox_corner_to_center(bboxs):
    """
    [xmin, ymin, xmax, ymax] -> [x_center, y_center, w, h]
    :param bboxs: [N, 4]
    """
    assert len(bboxs.shape) == 2
    tmp = np.zeros(bboxs.shape)

    # w
    tmp[:, 2] = bboxs[:, 2] - bboxs[:, 0] + 1
    # h
    tmp[:, 3] = bboxs[:, 3] - bboxs[:, 1] + 1
    # x_center
    tmp[:, 0] = bboxs[:, 0] + tmp[:, 2] / 2
    # y_center
    tmp[:, 1] = bboxs[:, 1] + tmp[:, 3] / 2

    return tmp


def bbox_center_to_corner(bboxs):
    """
    [x_center, y_center, w, h] -> [xmin, ymin, xmax, ymax]
    :param bboxs: [N, 4]
    """
    assert len(bboxs.shape) == 2
    tmp = np.zeros(bboxs.shape)

    # xmin
    tmp[:, 0] = bboxs[:, 0] - bboxs[:, 2] / 2
    # ymin
    tmp[:, 1] = bboxs[:, 1] - bboxs[:, 3] / 2
    # xmax
    tmp[:, 2] = bboxs[:, 0] + bboxs[:, 2] / 2
    # ymax
    tmp[:, 3] = bboxs[:, 1] + bboxs[:, 3] / 2

    return tmp


def nms(cates, probs, bboxs):
    """
    non-maximum suppression
    :param cates:
    :param probs:
    :param bboxs:
    :return:
    """
    pass
