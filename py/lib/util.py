# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午7:31
@file: util.py
@author: zj
@description: 
"""

import os
import numpy as np
import xmltodict
import torch
import matplotlib.pyplot as plt


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, 'car.csv')
    samples = np.loadtxt(csv_path, dtype=np.str)
    return samples


def parse_location_xml(xml_path):
    """
    解析xml文件，返回标注边界框信息（中心点坐标 + 长宽）
    """
    # print(xml_path)
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        # print(xml_dict)

        bndboxs = list()
        name_list = list()
        objects = xml_dict['annotation']['object']
        if isinstance(objects, list):
            for obj in objects:
                difficult = int(obj['difficult'])
                if difficult != 1:
                    name_list.append(obj['name'])

                    bndbox = obj['bndbox']
                    xmin = int(bndbox['xmin'])
                    ymin = int(bndbox['ymin'])
                    xmax = int(bndbox['xmax'])
                    ymax = int(bndbox['ymax'])

                    w = xmax - xmin
                    h = ymax - ymin
                    x = xmin + w / 2
                    y = ymin + h / 2
                    bndboxs.append((x, y, w, h))
        elif isinstance(objects, dict):
            difficult = int(objects['difficult'])

            if difficult != 1:
                name_list.append(objects['name'])

                bndbox = objects['bndbox']
                xmin = int(bndbox['xmin'])
                ymin = int(bndbox['ymin'])
                xmax = int(bndbox['xmax'])
                ymax = int(bndbox['ymax'])

                w = xmax - xmin
                h = ymax - ymin
                x = xmin + w / 2
                y = ymin + h / 2
                bndboxs.append((x, y, w, h))
        else:
            pass

        return np.array(bndboxs), name_list


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


def save_model(model, model_save_path):
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(model.state_dict(), model_save_path)


def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title('loss')
    plt.savefig('./loss.png')
