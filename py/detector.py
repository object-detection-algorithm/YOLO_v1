# -*- coding: utf-8 -*-

"""
@date: 2020/4/18 上午11:23
@file: detector.py
@author: zj
@description: 
"""

import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from utils import file
from utils import util
from utils import draw
from models.location_dataset import LocationDataset
from models.yolo_v1 import YOLO_v1

S = 7
B = 2
C = 3


def load_data(img_path, xml_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    src = cv2.imread(img_path)
    bndboxs, name_list = file.parse_location_xml(xml_path)
    # dst = draw.plot_box(src, bndboxs, name_list)
    # draw.show(dst)

    h, w = src.shape[:2]
    img = transform(src)
    scale_h, scale_w = img.shape[1:]
    ratio_h = scale_h / h
    ratio_w = scale_w / w

    # [C, H, W] -> [N, C, H, W]
    img = img.unsqueeze(0)

    data_dict = {}
    data_dict['src'] = src
    data_dict['src_size'] = (h, w)
    data_dict['bndboxs'] = bndboxs
    data_dict['name_list'] = name_list

    data_dict['img'] = img
    data_dict['scale_size'] = (scale_h, scale_w)
    data_dict['ratio'] = (ratio_h, ratio_w)

    return img, data_dict


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    img, data_dict = load_data('../imgs/cucumber_9.jpg', '../imgs/cucumber_9.xml')
    model = file.load_model(device, S, B, C)
    # 计算
    outputs = model.forward(img.to(device)).cpu().squeeze(0)
    print(outputs.shape)

    # (S*S, C)
    pred_probs = outputs[:, :C]
    # (S*S, C:(C+B))
    pred_confidences = outputs[:, C:(C + B)]
    # (S*S, (C+B):(C+5B))
    pred_bboxs = outputs[:, (C + B):]

    # 计算类别
    pred_cates = torch.argmax(pred_probs, dim=1)
    # 计算分类概率
    pred_confidences_idxs = torch.argmax(pred_confidences, dim=1)
    pred_cate_probs = pred_probs[range(S * S), pred_cates] \
                      * pred_confidences[range(S * S), pred_confidences_idxs]
    # 计算预测边界框
    pred_cate_bboxs = torch.zeros(S * S, 4)
    pred_cate_bboxs[:, 0] = pred_bboxs[range(S * S), pred_confidences_idxs * 4]
    pred_cate_bboxs[:, 1] = pred_bboxs[range(S * S), pred_confidences_idxs * 4 + 1]
    pred_cate_bboxs[:, 2] = pred_bboxs[range(S * S), pred_confidences_idxs * 4 + 2]
    pred_cate_bboxs[:, 3] = pred_bboxs[range(S * S), pred_confidences_idxs * 4 + 3]

    # 预测边界框的缩放，回到原始图像
    pred_bboxs = util.deform_bboxs(pred_cate_bboxs, data_dict, S)
    # 在原图绘制标注边界框和预测边界框
    dst = draw.plot_bboxs(data_dict['src'], data_dict['bndboxs'], data_dict['name_list'], pred_bboxs, pred_cates,
                          pred_cate_probs)
    cv2.imwrite('./detect.png', dst)
    # BGR -> RGB
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    draw.show(dst)
