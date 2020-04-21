# -*- coding: utf-8 -*-

"""
@date: 2020/4/21 下午1:30
@file: parse_location.py
@author: zj
@description: 解析定位数据集，分别保存图像和标注数据
"""

import os
import shutil
import glob

from utils import util


def pretreat():
    """
    判断源文件目录是否为空
    清空结果文件目录，并新建图像和标注文件夹
    :return:
    """
    src_root_dir = '../../data/training_images'
    dst_root_dir = '../../data/location_dataset'

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
    src_root_dir, dst_root_dir, dst_img_dir, dst_annotation_dir = pretreat()

    img_path_list = glob.glob(os.path.join(src_root_dir, '*.jpg'))
    annotation_path_list = glob.glob(os.path.join(src_root_dir, '*.xml'))

    for img_path in img_path_list:
        dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, dst_img_path)

    for annotation_path in annotation_path_list:
        dst_annotation_path = os.path.join(dst_annotation_dir, os.path.basename(annotation_path))
        shutil.copyfile(annotation_path, dst_annotation_path)
