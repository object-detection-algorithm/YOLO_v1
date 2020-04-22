# -*- coding: utf-8 -*-

"""
@date: 2020/4/16 下午8:53
@file: multi_part_loss.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.location_dataset import LocationDataset
from models.yolo_v1 import YOLO_v1


class MultiPartLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(MultiPartLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.coord = lambda_coord
        self.noobj = lambda_noobj

    def forward(self, preds, targets):
        """
        :param preds: (N, S*S, B*5+C) 其中
        [N, S*S, :C] 表示每个网格中预测目标的类别（如果存在则指定类别设置为概率１，不存在为0）
        [N, S*S: C:(C+B)] 表示每个网格中目标的置信度（如果存在为1，不存在为0）
        [N, S*S, (C+B):(C+B*5)] 表示网格中目标的边界框（center_x/center_y/w/h，如果不存在则为0）
        :param targets: (N, S*S, (B*5+C))
        :return:
        """
        N = preds.shape[0]
        ## 预测
        # 提取每个网格的分类概率
        # [N, S*S, C] -> [N*S*S, C]
        pred_probs = preds[:, :, :self.C].reshape(-1, self.C)
        # 提取每个网格的置信度
        # [N, S*S, B] -> [N*S*S, B]
        pred_confidences = preds[:, :, self.C: (self.B + self.C)].reshape(-1, self.B)
        # 提取每个网格的预测边界框坐标
        # [N, S*S, B*4] -> [N*S*S, B*4] -> [N*S*S, B, 4]
        pred_bboxs = preds[:, :, (self.B + self.C): (self.B * 5 + self.C)] \
            .reshape(-1, self.B * 4) \
            .reshape(-1, self.B, 4)

        ## 目标
        # 提取每个网格的分类概率
        # [N, S*S, C] -> [N*S*S, C]
        target_probs = targets[:, :, :self.C].reshape(-1, self.C)
        # 提取每个网格的置信度
        # [N, S*S, B] -> [N*S*S, B]
        target_confidences = targets[:, :, self.C: (self.B + self.C)].reshape(-1, self.B)
        # 提取每个网格的边界框坐标
        # [N, S*S, B*4] -> [N*S*S, B*4] -> [N*S*S, B, 4]
        target_bboxs = targets[:, :, (self.B + self.C): (self.B * 5 + self.C)] \
            .reshape(-1, self.B * 4) \
            .reshape(-1, self.B, 4)

        ## 首先计算所有边界框的置信度损失（假定不存在obj）
        loss = self.noobj * self.sum_squared_error(pred_confidences, target_confidences)

        # 计算每个预测边界框与对应目标边界框的IoU
        iou_scores = self.iou(pred_bboxs.reshape(-1, 4), target_bboxs.reshape(-1, 4)).reshape(-1, 2)
        # 选取每个网格中IoU最高的边界框
        top_idxs = torch.argmax(iou_scores, dim=1)
        top_len = len(top_idxs)
        # 获取相应的置信度以及边界框
        top_pred_confidences = pred_confidences[range(top_len), top_idxs]
        top_pred_bboxs = pred_bboxs[range(top_len), top_idxs]

        top_target_confidences = target_confidences[range(top_len), top_idxs]
        top_target_bboxs = target_bboxs[range(top_len), top_idxs]
        # print(top_pred_confidences.shape)
        # print(top_pred_bboxs.shape)

        # 选取存在目标的网格
        obj_idxs = torch.sum(target_probs, dim=1) == 1
        # print(obj_idxs)

        obj_pred_confidences = top_pred_confidences[obj_idxs]
        obj_pred_bboxs = top_pred_bboxs[obj_idxs]
        obj_pred_probs = pred_probs[obj_idxs]

        obj_target_confidences = top_target_confidences[obj_idxs]
        obj_target_bboxs = top_target_bboxs[obj_idxs]
        obj_target_probs = target_probs[obj_idxs]

        ## 计算目标边界框的置信度损失
        loss += (1 - self.noobj) * self.sum_squared_error(obj_pred_confidences, obj_target_confidences)
        ## 计算分类概率损失
        loss += self.sum_squared_error(obj_pred_probs, obj_target_probs)
        ## 计算边界框坐标损失
        loss += self.sum_squared_error(obj_pred_bboxs[:, :2], obj_target_bboxs[:, :2])
        loss += self.sum_squared_error(torch.sqrt(obj_pred_bboxs[:, 2:]), torch.sqrt(obj_target_bboxs[:, 2:]))

        return loss / N

        # N = preds.shape[0]
        # total_loss = 0.0
        # print(preds.shape)
        # print(targets.shape)
        # for pred, target in zip(preds, targets):
        #     """
        #     逐个图像计算
        #     pred: [S*S, (B*5+C)]
        #     target: [S*S, (B*5+C)]
        #     """
        #     # 分类概率
        #     pred_probs = pred[:, :self.C]
        #     target_probs = target[:, :self.C]
        #     # 置信度
        #     pred_confidences = pred[:, self.C:(self.C + self.B)]
        #     target_confidences = target[:, self.C:(self.C + self.B)]
        #     # 边界框坐标
        #     pred_bboxs = pred[:, (self.C + self.B):]
        #     target_bboxs = target[:, (self.C + self.B):]
        #
        #     for i in range(self.S * self.S):
        #         """
        #         逐个网格计算
        #         """
        #         pred_single_probs = pred_probs[i]
        #         target_single_probs = target_probs[i]
        #
        #         pred_single_confidences = pred_confidences[i]
        #         target_single_confidences = target_confidences[i]
        #
        #         pred_single_bboxs = pred_bboxs[i]
        #         target_single_bboxs = target_bboxs[i]
        #
        #         # 是否存在置信度（如果存在，则target的置信度必然大于0）
        #         is_obj = target_single_confidences[0] > 0
        #         # 计算置信度损失 假定该网格不存在对象
        #         total_loss += self.noobj * self.sum_squared_error(pred_single_confidences, target_single_confidences)
        #         print(total_loss)
        #         if is_obj:
        #             print('i = %d' % (i))
        #             # 如果存在
        #             # 计算分类损失
        #             total_loss += self.sum_squared_error(pred_single_probs, target_single_probs)
        #             print(total_loss)
        #
        #             # 计算所有预测边界框和标注边界框的IoU
        #             pred_single_bboxs = pred_single_bboxs.reshape(-1, 4)
        #             target_single_bboxs = target_single_bboxs.reshape(-1, 4)
        #
        #             scores = self.iou(pred_single_bboxs, target_single_bboxs)
        #             # 提取IoU最大的下标
        #             bbox_idx = torch.argmax(scores)
        #             # 计算置信度损失
        #             total_loss += (1 - self.noobj) * \
        #                           self.sum_squared_error(pred_single_confidences[bbox_idx],
        #                                                  target_single_confidences[bbox_idx])
        #             print(total_loss)
        #             # 计算边界框损失
        #             total_loss += self.coord * self.bbox_loss(pred_single_bboxs[bbox_idx].reshape(-1, 4),
        #                                                       target_single_bboxs[bbox_idx].reshape(-1, 4))
        #             print(total_loss)
        #
        #             print('done')
        #
        # return total_loss / N

    def sum_squared_error(self, preds, targets):
        return torch.sum((preds - targets) ** 2)

    def bbox_loss(self, pred_boxs, target_boxs):
        """
        :param pred_boxs: 大小为[N, 4] (center_x, center_y, w, h)
        :param target_boxs: 大小为[N, 4] (center_x, center_y , w, h)
        """
        loss = 0.0
        pred_boxs = pred_boxs.float()
        target_boxs = target_boxs.float()

        loss += self.sum_squared_error(pred_boxs[:, :2], target_boxs[:, :2])
        loss += self.sum_squared_error(torch.sqrt(pred_boxs[:, 2:4]), torch.sqrt(target_boxs[:, 2:4]))

        return loss

    def iou(self, pred_boxs, target_boxs):
        """
        计算候选建议和标注边界框的IoU
        :param pred_box: 大小为[N, 4] (center_x, center_y, w, h)
        :param target_box: 大小为[N, 4] (center_x, center_y , w, h)
        :return: [N]
        """
        pred_boxs = pred_boxs.cpu().detach().numpy()
        target_boxs = target_boxs.cpu().detach().numpy()

        xA = np.maximum(pred_boxs[:, 0] - pred_boxs[:, 2] / 2, target_boxs[:, 0] - target_boxs[:, 2] / 2)
        yA = np.maximum(pred_boxs[:, 1] - pred_boxs[:, 3] / 2, target_boxs[:, 1] - target_boxs[:, 3] / 2)
        xB = np.minimum(pred_boxs[:, 0] + pred_boxs[:, 2] / 2, target_boxs[:, 0] + target_boxs[:, 2] / 2)
        yB = np.minimum(pred_boxs[:, 1] + pred_boxs[:, 3] / 2, target_boxs[:, 1] + target_boxs[:, 3] / 2)
        # 计算交集面积
        intersection = np.maximum(0.0, xB - xA + 1) * np.maximum(0.0, yB - yA + 1)
        # 计算两个边界框面积
        boxAArea = pred_boxs[:, 2] * pred_boxs[:, 3]
        boxBArea = target_boxs[:, 2] * target_boxs[:, 3]

        scores = intersection / (boxAArea + boxBArea - intersection)
        return torch.from_numpy(scores)


def load_data(data_root_dir, cate_list, S=7, B=2, C=20):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_set = LocationDataset(data_root_dir, cate_list, transform=transform, S=S, B=B, C=C)
    data_loader = DataLoader(data_set, batch_size=1, num_workers=8)

    return data_loader


if __name__ == '__main__':
    S = 7
    B = 2
    C = 3
    cate_list = ['cucumber', 'eggplant', 'mushroom']

    criterion = MultiPartLoss(S=7, B=2, C=3)
    # preds = torch.arange(637).reshape(1, 7 * 7, 13) * 0.01
    # targets = torch.ones((1, 7 * 7, 13)) * 0.01
    # loss = criterion(preds, targets)
    # print(loss)
    data_loader = load_data('../../data/location_dataset', cate_list, S=S, B=B, C=C)
    model = YOLO_v1(S=S, B=B, C=C)

    for inputs, labels in data_loader:
        inputs = inputs
        labels = labels
        print(inputs.shape)
        print(labels.shape)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(loss)
            exit(0)
