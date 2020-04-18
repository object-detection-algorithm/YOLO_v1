# -*- coding: utf-8 -*-

"""
@date: 2020/4/18 下午2:41
@file: file.py
@author: zj
@description: 
"""

import os
import xmltodict
import numpy as np
import torch


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


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

                    bndboxs.append((xmin, ymin, xmax, ymax))
                    # w = xmax - xmin
                    # h = ymax - ymin
                    # x = xmin + w / 2
                    # y = ymin + h / 2
                    # bndboxs.append((x, y, w, h))
        elif isinstance(objects, dict):
            difficult = int(objects['difficult'])

            if difficult != 1:
                name_list.append(objects['name'])

                bndbox = objects['bndbox']
                xmin = int(bndbox['xmin'])
                ymin = int(bndbox['ymin'])
                xmax = int(bndbox['xmax'])
                ymax = int(bndbox['ymax'])

                bndboxs.append((xmin, ymin, xmax, ymax))
                # w = xmax - xmin
                # h = ymax - ymin
                # x = xmin + w / 2
                # y = ymin + h / 2
                # bndboxs.append((x, y, w, h))
        else:
            pass

        return np.array(bndboxs), name_list


def save_model(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)


def save_checkpoint(model_save_path, epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, model_save_path)
