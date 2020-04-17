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
        [N, S*S, :C] 表示每个网格中预测目标的类别（如果存在为[0, self.C-1]，不存在为-1）
        [N, S*S: C:(C+B)] 表示每个网格中目标的置信度（如果存在为1，不存在为-1）
        [N, S*S, (C+B):(C+B*5)] 表示网格中目标的边界框（center_x/center_y/w/h，如果不存在则为0）
        :param targets: (N, S*S*(B*5+C))
        :return:
        """
        # ## 预测
        # # 提取每个网格的分类概率
        # pred_probs = preds[-1, :self.S * self.S * self.C].reshape(-1, self.S, self.S, self.C)
        # # 提取每个网格的置信度
        # pred_confidences = preds[-1, self.S * self.S * self.C: self.S * self.S * (self.B + self.C)] \
        #     .reshape(-1, self.S, self.S, self.B)
        # # 提取每个网格的预测边界框坐标
        # pred_bboxs = preds[-1, self.S * self.S * (self.B + self.C): self.S * self.S * (self.B * 5 + self.C)] \
        #     .reshape(-1, self.S, self.S, 4)
        #
        # ## 目标
        # # 每个网格的分类
        # target_probs = targets[-1, :self.S * self.S].reshape(-1, self.S, self.S)
        # # 置信度
        # target_confidences = targets[-1, self.S * self.S: self.S * self.S * 2].reshape(-1, self.S, self.S)
        # # 坐标
        # target_bboxs = targets[-1, self.S * self.S * 2:self.S * self.S * 6].reshape(-1, self.S, self.S, 4)
        #
        # # 图像中哪些网格包含了目标（根据分类判断）
        # objs = torch.where(target_probs != -1)
        # # 哪些不包含目标
        # nobjs = torch.where(target_probs == -1)
        #
        # ## 首先计算包含了分类的

        N = preds.shape[0]
        total_loss = 0.0
        for pred, target in zip(preds, targets):
            """
            逐个图像计算
            pred: [S*S, (B*5+C)]
            target: [S*S, (B*5+C)]
            """
            # 分类概率
            pred_probs = pred[:, :self.C]
            target_probs = target[:, :self.C]
            # 置信度
            pred_confidences = pred[:, self.C:(self.C + self.B)]
            target_confidences = target[:, self.C:(self.C + self.B)]
            # 边界框坐标
            pred_bboxs = pred[:, (self.C + self.B):]
            target_bboxs = target[:, (self.C + self.B):]

            for i in range(self.S * self.S):
                """
                逐个网格计算
                """
                pred_single_probs = pred_probs[i]
                target_single_probs = target_probs[i]

                pred_single_confidences = pred_confidences[i]
                target_single_confidences = target_confidences[i]

                pred_single_bboxs = pred_bboxs[i]
                target_single_bboxs = target_bboxs[i]

                # 是否存在置信度（如果存在，则target的置信度必然大于0）
                is_obj = target_single_confidences[0] > 0
                # 计算置信度损失 假定所有目标都不存在
                total_loss += self.noobj * self.sum_squared_error(pred_single_confidences, target_single_confidences)
                if is_obj:
                    # 如果存在
                    # 计算分类损失
                    total_loss += self.sum_squared_error(pred_single_probs, target_single_probs)

                    # 计算所有预测边界框和标注边界框的IoU
                    pred_single_bboxs = pred_single_bboxs.reshape(-1, 4)
                    target_single_bboxs = target_single_bboxs.reshape(-1, 4)

                    scores = self.iou(pred_single_bboxs, target_single_bboxs)
                    # 提取IoU最大的下标
                    bbox_idx = torch.argmax(scores)
                    # 计算置信度损失
                    total_loss += (1 - self.noobj) * \
                                  self.sum_squared_error(pred_single_confidences[bbox_idx],
                                                         target_single_confidences[bbox_idx])
                    # 计算边界框损失
                    total_loss += self.coord * self.bbox_loss(pred_single_bboxs[bbox_idx].reshape(-1, 4),
                                                              target_single_bboxs[bbox_idx].reshape(-1, 4))

        return total_loss / N

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

        loss += torch.sum((pred_boxs[:, 0] - target_boxs[:, 0]) ** 2)
        loss += torch.sum((pred_boxs[:, 1] - target_boxs[:, 1]) ** 2)

        loss += torch.sum((torch.sqrt(pred_boxs[:, 2]) - torch.sqrt(target_boxs[:, 2])) ** 2)
        loss += torch.sum((torch.sqrt(pred_boxs[:, 3]) - torch.sqrt(target_boxs[:, 3])) ** 2)

        return loss

    def iou(self, pred_boxs, target_boxs):
        """
        计算候选建议和标注边界框的IoU
        :param pred_box: 大小为[N, 4] (center_x, center_y, w, h)
        :param target_box: 大小为[N, 4] (center_x, center_y , w, h)
        :return: [N]
        """
        pred_boxs = pred_boxs.detach().numpy()
        target_boxs = target_boxs.detach().numpy()

        xA = np.maximum(pred_boxs[:, 0] - pred_boxs[:, 2] / 2, target_boxs[:, 0] - target_boxs[:, 2] / 2)
        yA = np.maximum(pred_boxs[:, 1] - pred_boxs[:, 3] / 2, target_boxs[:, 1] - target_boxs[:, 3] / 2)
        xB = np.minimum(pred_boxs[:, 0] + pred_boxs[:, 2] / 2, target_boxs[:, 0] + target_boxs[:, 2] / 2)
        yB = np.minimum(pred_boxs[:, 1] + pred_boxs[:, 3] / 2, target_boxs[:, 1] + target_boxs[:, 3] / 2)
        # 计算交集面积
        intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
        # 计算两个边界框面积
        boxAArea = pred_boxs[:, 2] * pred_boxs[:, 3]
        boxBArea = target_boxs[:, 2] * target_boxs[:, 3]

        scores = intersection / (boxAArea + boxBArea - intersection)
        return torch.from_numpy(scores)


if __name__ == '__main__':
    criterion = MultiPartLoss(S=7, B=2, C=3)

    preds = torch.arange(637).reshape(1, 7 * 7, 13) * 0.01
    targets = torch.ones((1, 7 * 7, 13)) * 0.01

    loss = criterion.forward(preds, targets)
    print(loss)
