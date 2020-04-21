# -*- coding: utf-8 -*-

"""
@date: 2020/4/19 下午3:07
@file: batch_detect.py.py
@author: zj
@description: 批量检测数据（for mAP）
"""

import os
import glob
import time
import shutil
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

cate_list = ['cucumber', 'eggplant', 'mushroom']


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform


def load_data(root_dir):
    img_path_list = glob.glob(os.path.join(root_dir, '*.jpg'))
    annotation_path_list = [os.path.join(root_dir, os.path.splitext(os.path.basename(img_path))[0] + ".xml")
                            for img_path in img_path_list]

    return img_path_list, annotation_path_list


def parse_data(img_path, xml_path, transform):
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


def load_model(device):
    model_path = './models/checkpoint_yolo_v1.pth'
    model = YOLO_v1(S=7, B=2, C=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    return model


def deform_bboxs(pred_bboxs, data_dict):
    """
    :param pred_bboxs: [S*S, 4]
    :return:
    """
    scale_h, scale_w = data_dict['scale_size']
    grid_w = scale_w / S
    grid_h = scale_h / S

    bboxs = np.zeros(pred_bboxs.shape)
    for i in range(S * S):
        row = int(i / S)
        col = int(i % S)

        x_center, y_center, box_w, box_h = pred_bboxs[i]
        bboxs[i, 0] = (col + x_center) * grid_w
        bboxs[i, 1] = (row + y_center) * grid_h
        bboxs[i, 2] = box_w * scale_w
        bboxs[i, 3] = box_h * scale_h
    # (x_center, y_center, w, h) -> (xmin, ymin, xmax, ymax)
    bboxs = util.bbox_center_to_corner(bboxs)

    ratio_h, ratio_w = data_dict['ratio']
    bboxs[:, 0] /= ratio_w
    bboxs[:, 1] /= ratio_h
    bboxs[:, 2] /= ratio_w
    bboxs[:, 3] /= ratio_h

    # 最大最小值
    h, w = data_dict['src_size']
    bboxs[:, 0] = np.maximum(bboxs[:, 0], 0)
    bboxs[:, 1] = np.maximum(bboxs[:, 1], 0)
    bboxs[:, 2] = np.minimum(bboxs[:, 2], w)
    bboxs[:, 3] = np.minimum(bboxs[:, 3], h)

    return bboxs.astype(int)


def save_data(img_name, img, target_cates, target_bboxs, pred_cates, pred_probs, pred_bboxs):
    """
    保存检测结果
    :param img_name: 图像名
    :param img: 原始图像
    :param target_cates: 标注边界框所属类别
    :param target_bboxs: 标注边界框坐标
    :param pred_cates: 预测边界框类别
    :param pred_probs: 预测边界框置信度
    :param pred_bboxs: 预测边界框坐标
    """
    dst_root_dir = './data/outputs'
    dst_target_dir = os.path.join(dst_root_dir, 'targets')
    dst_pred_dir = os.path.join(dst_root_dir, 'preds')
    dst_img_dir = os.path.join(dst_root_dir, 'imgs')

    file.check_dir(dst_root_dir)
    file.check_dir(dst_target_dir)
    file.check_dir(dst_pred_dir)
    file.check_dir(dst_img_dir)

    img_path = os.path.join(dst_img_dir, img_name + ".png")
    cv2.imwrite(img_path, img)
    annotation_path = os.path.join(dst_target_dir, img_name + ".txt")
    with open(annotation_path, 'w') as f:
        for i in range(len(target_cates)):
            target_cate_name = target_cates[i]
            xmin, ymin, xmax, ymax = target_bboxs[i]

            f.write('%s %d %d %d %d' % (target_cate_name, xmin, ymin, xmax, ymax))
            if i != (len(target_cates) - 1):
                f.write('\n')
    pred_path = os.path.join(dst_pred_dir, img_name + ".txt")
    with open(pred_path, 'w') as f:
        for i in range(len(pred_cates)):
            pred_cate_idx = pred_cates[i]
            pred_prob = pred_probs[i]
            xmin, ymin, xmax, ymax = pred_bboxs[i]

            f.write('%s %.3f %d %d %d %d' % (cate_list[pred_cate_idx], pred_prob, xmin, ymin, xmax, ymax))
            if i != (len(pred_cates) - 1):
                f.write('\n')


if __name__ == '__main__':
    # device = util.get_device()
    device = "cpu"
    model = load_model(device)

    transform = get_transform()
    img_path_list, annotation_path_list = load_data('./data/training_images')
    # print(img_path_list)

    N = len(img_path_list)
    for i in range(N):
        img_path = img_path_list[i]
        print(i, img_path)
        annotation_path = annotation_path_list[i]

        img, data_dict = parse_data(img_path, annotation_path, transform)

        # 计算
        outputs = model.forward(img.to(device)).cpu().squeeze(0).numpy()

        # (S*S, C)
        pred_probs = outputs[:, :C]
        # (S*S, C:(C+B))
        pred_confidences = outputs[:, C:(C + B)]
        # (S*S, (C+B):(C+5B))
        pred_bboxs = outputs[:, (C + B):]

        # 计算类别
        pred_cates = np.argmax(pred_probs, axis=1).astype(int)
        # 计算分类概率
        pred_confidences_idxs = np.argmax(pred_confidences, axis=1)
        pred_cate_probs = pred_probs[range(S * S), pred_cates] \
                          * pred_confidences[range(S * S), pred_confidences_idxs]
        # 计算预测边界框
        pred_cate_bboxs = np.zeros((S * S, 4))
        pred_cate_bboxs[:, 0] = pred_bboxs[range(S * S), pred_confidences_idxs * 4]
        pred_cate_bboxs[:, 1] = pred_bboxs[range(S * S), pred_confidences_idxs * 4 + 1]
        pred_cate_bboxs[:, 2] = pred_bboxs[range(S * S), pred_confidences_idxs * 4 + 2]
        pred_cate_bboxs[:, 3] = pred_bboxs[range(S * S), pred_confidences_idxs * 4 + 3]

        # 预测边界框的缩放，回到原始图像
        pred_bboxs = deform_bboxs(pred_cate_bboxs, data_dict)

        # 保存图像/标注边界框/预测边界框
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_data(img_name, data_dict['src'], data_dict['name_list'], data_dict['bndboxs'],
                  pred_cates, pred_cate_probs, pred_bboxs)
    print('done')
