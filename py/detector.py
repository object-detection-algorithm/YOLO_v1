# -*- coding: utf-8 -*-

"""
@date: 2020/4/18 上午11:23
@file: detector.py
@author: zj
@description: 
"""

import time
import cv2
import torch
from torch.utils.data import DataLoader
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

    img = cv2.imread(img_path)
    bndboxs, name_list = file.parse_location_xml(xml_path)

    return img, bndboxs, name_list


def val_model(data_loader, model, device=None):
    since = time.time()

    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        # 计算mAP
        cates, probs, bboxs = util.parse_output(outputs, S, B, C)
        util.nms(cates, probs, bboxs)

    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    img, bndboxs, name_list = load_data('../imgs/cucumber_9.jpg', '../imgs/cucumber_9.xml')
    dst = draw.plot_box(img, bndboxs, name_list)
    draw.show(dst)

    # model_path = '../models/checkpoint_yolo_v1_24.pth'
    # model = YOLO_v1(S=7, B=2, C=3)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False
    # model = model.to(device)
    #
    # val_model(data_loader, model, device=device)
