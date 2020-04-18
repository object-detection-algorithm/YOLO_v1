# -*- coding: utf-8 -*-

"""
@date: 2020/4/16 下午7:56
@file: location_dataset.py
@author: zj
@description: 
"""

import cv2
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import file
import torchvision.transforms as transforms

cate_list = ['cucumber', 'eggplant', 'mushroom']


class LocationDataset(Dataset):

    def __init__(self, root_dir, transform=None, S=7, B=2, C=20):
        """
        保存图像以及标注框性能
        :param root_dir: 根目录
        :param transform:
        :param S: 每行／每列网格数
        :param B: 单个网格边界框数
        :param C:　类别数
        """
        super(LocationDataset, self).__init__()
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        jpeg_path_list = []
        xml_path_list = []
        for name in cate_list:
            for i in range(1, 61):
                jpeg_path_list.append(os.path.join(root_dir, '%s_%d.jpg' % (name, i)))
                xml_path_list.append(os.path.join(root_dir, '%s_%d.xml' % (name, i)))
        self.jpeg_path_list = jpeg_path_list
        self.xml_path_list = xml_path_list

    def __getitem__(self, index):
        """
        返回image, target
        image：图像数据
        target：[S*S, C + B*5]
        单个网格信息示例：[C1, C2, C3, ..., Cn, confidenceA, confidenceB, .., xA, yA, wA, hA, xB, yB, wB, hB, ...]
        """
        assert index < len(self.jpeg_path_list), 'image length: %d' % len(self.jpeg_path_list)

        # print(self.jpeg_path_list[index])
        image = cv2.imread(self.jpeg_path_list[index])
        img_h, img_w = image.shape[:2]
        ratio_h = 1
        ratio_w = 1
        if self.transform:
            image = self.transform(image)
            # 计算图像缩放比例
            dst_img_h, dst_img_w = image.shape[1:3]
            ratio_h = 1.0 * dst_img_h / img_h
            ratio_w = 1.0 * dst_img_w / img_w

            img_h, img_w = image.shape[1:3]

        # 单个网格长宽
        grid_h = img_h / self.S
        grid_w = img_w / self.S

        target = torch.zeros((self.S * self.S, self.C + self.B * 5))
        bndboxs, name_list = file.parse_location_xml(self.xml_path_list[index])
        # 缩放边界框坐标（x, y, w, h）
        bndboxs[:, 0] = bndboxs[:, 0] * ratio_w
        bndboxs[:, 1] = bndboxs[:, 1] * ratio_h
        bndboxs[:, 2] = bndboxs[:, 2] * ratio_w
        bndboxs[:, 3] = bndboxs[:, 3] * ratio_h

        # Note：每个网格仅包含单个标注边界框
        grid_nums = torch.zeros((self.S, self.S))
        for i in range(len(bndboxs)):
            bndbox = bndboxs[i]
            name = name_list[i]

            box_x, box_y, box_w, box_h = bndbox
            # 边界框中心位于哪个网格
            grid_x = int(box_x / grid_w)
            grid_y = int(box_y / grid_h)
            # 边界框中心相对于网格的比例（0,1）
            x = (box_x % grid_w) / grid_w
            y = (box_y % grid_h) / grid_h
            # 边界框长宽相对于图像的比例（0,1）
            w = box_w / img_w
            h = box_h / img_h
            # 该网格内是否已填充（每个网格1个标注边界框）
            if grid_nums[grid_x, grid_y] > 1:
                print('网格(%d, %d)已填充：%s' % (grid_x, grid_y, str(target[grid_x, grid_y])))
            else:
                # 转换类别和标签
                cate_idx = cate_list.index(name)
                # 指定类别概率为1
                target[grid_x * grid_y, cate_idx] = 1
                for j in range(self.B):
                    # 置信度
                    target[grid_x * grid_y, self.C + j] = 1
                    # 相应的边界框坐标
                    target[grid_x * grid_y, self.C + self.B + 4 * j] = x
                    target[grid_x * grid_y, self.C + self.B + 4 * j + 1] = y
                    target[grid_x * grid_y, self.C + self.B + 4 * j + 2] = w
                    target[grid_x * grid_y, self.C + self.B + 4 * j + 3] = h

                grid_nums[grid_x, grid_y] += 1

        return image, target

    def __len__(self):
        return len(self.jpeg_path_list)


if __name__ == '__main__':
    root_dir = '../../data/training_images/'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_set = LocationDataset(root_dir, transform, 7, 2, 3)
    # print(data_set)
    # print(len(data_set))
    #
    # image, target = data_set.__getitem__(3)
    # print(image.shape)
    # print(target.shape)
    # print(target)

    data_loader = DataLoader(data_set, shuffle=True, batch_size=8, num_workers=8)
    items = next(iter(data_loader))
    inputs, labels = items
    print(inputs.shape)
    print(labels.shape)
