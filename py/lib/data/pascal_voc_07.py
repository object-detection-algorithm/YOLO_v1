# -*- coding: utf-8 -*-

"""
@date: 2020/4/22 下午2:05
@file: pascal_voc_07.py
@author: zj
@description: 下载并解压PASCAL VOC数据集
"""

import cv2
import os
import glob
import shutil
import numpy as np
from torchvision.datasets import VOCDetection

from utils import util


def get_dataset(root_dir):
    """
    下载PASCAL VOC数据集
    """
    dataset = VOCDetection(root_dir, year='2007', image_set='trainval', download=True)

    # img, target = dataset.__getitem__(1000)
    # img = np.array(img)
    # print(target)
    # print(img.shape)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return dataset


def pretreat(src_root_dir, dst_root_dir):
    """
    判断源文件目录是否为空
    清空结果文件目录，并新建图像和标注文件夹
    :return:
    """
    if not os.path.exists(src_root_dir):
        util.error("%s doesn't exist" % src_root_dir)
    if os.path.exists(dst_root_dir):
        shutil.rmtree(dst_root_dir)
    os.mkdir(dst_root_dir)

    dst_img_dir = os.path.join(dst_root_dir, 'imgs')
    dst_annotation_dir = os.path.join(dst_root_dir, 'annotations')
    os.mkdir(dst_img_dir)
    os.mkdir(dst_annotation_dir)

    return src_root_dir, dst_root_dir, dst_img_dir, dst_annotation_dir


if __name__ == '__main__':
    data_dir = '../../data'
    data_set = get_dataset(data_dir)

    src_root_dir = '../../data/VOCdevkit/VOC2007'
    dst_root_dir = '../../data/VOC_dataset'
    src_root_dir, dst_root_dir, dst_img_dir, dst_annotation_dir = pretreat(src_root_dir, dst_root_dir)

    img_path_list = glob.glob(os.path.join(src_root_dir, 'JPEGImages', '*.jpg'))
    annotation_path_list = glob.glob(os.path.join(src_root_dir, 'Annotations', '*.xml'))

    for img_path in img_path_list:
        dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, dst_img_path)

    for annotation_path in annotation_path_list:
        dst_annotation_path = os.path.join(dst_annotation_dir, os.path.basename(annotation_path))
        shutil.copyfile(annotation_path, dst_annotation_path)

    print('done')