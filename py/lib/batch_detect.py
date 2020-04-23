# -*- coding: utf-8 -*-

"""
@date: 2020/4/19 下午3:07
@file: batch_detect.py.py
@author: zj
@description: 批量检测数据（for mAP）
"""

import os
import glob
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from utils import file
from utils import util
from utils import voc_map
from models.yolo_v1 import YOLO_v1

S = 7
B = 2
# C = 3
# cate_list = ['cucumber', 'eggplant', 'mushroom']

C = 20
cate_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

dst_root_dir = '../../data/outputs'
dst_target_dir = os.path.join(dst_root_dir, 'targets')
dst_pred_dir = os.path.join(dst_root_dir, 'preds')
dst_img_dir = os.path.join(dst_root_dir, 'imgs')
tmp_json_dir = os.path.join(dst_root_dir, '.tmp_files')

file.make_dir(dst_root_dir, is_rm=True)
file.make_dir(dst_target_dir, is_rm=True)
file.make_dir(dst_pred_dir, is_rm=True)
file.make_dir(dst_img_dir, is_rm=True)
file.make_dir(tmp_json_dir, is_rm=True)


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform


def load_data(root_dir):
    img_path_list = glob.glob(os.path.join(root_dir, 'imgs', '*.jpg'))
    img_path_list.sort()
    annotation_path_list = [
        os.path.join(root_dir, 'annotations', os.path.splitext(os.path.basename(img_path))[0] + ".xml")
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
    model = file.load_model(device, S, B, C)

    transform = get_transform()
    # img_path_list, annotation_path_list = load_data('./../data/location_dataset')
    img_path_list, annotation_path_list = load_data('../../data/VOC_dataset')
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
        pred_bboxs = util.deform_bboxs(pred_cate_bboxs, data_dict, S)

        # 保存图像/标注边界框/预测边界框
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_data(img_name, data_dict['src'], data_dict['name_list'], data_dict['bndboxs'],
                  pred_cates, pred_cate_probs, pred_bboxs)
    print('compute mAP')
    voc_map.voc_map(dst_target_dir, dst_pred_dir, tmp_json_dir)
