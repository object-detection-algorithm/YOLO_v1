# -*- coding: utf-8 -*-

"""
@date: 2020/4/17 下午4:43
@file: train.py
@author: zj
@description: 
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import util
from models.location_dataset import LocationDataset
from models.yolo_v1 import YOLO_v1
from models.multi_part_loss import MultiPartLoss


def load_data(data_root_dir, S=7, B=2, C=20):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_set = LocationDataset(data_root_dir, transform=transform, S=S, B=B, C=C)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)

    return data_loader


def train_model(data_loader, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    # best_model_weights = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    loss = criterion(outputs, labels)
                    print(loss.item())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            exit(0)
            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    data_loader = load_data('../data/training_images', S=7, B=2, C=3)
    print(len(data_loader))

    model = YOLO_v1(S=7, B=2, C=3)
    model = model.to(device)

    criterion = MultiPartLoss(S=7, B=2, C=3)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(data_loader, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=device)

    # 保存最好的模型参数
    # util.check_dir('./models')
    # torch.save(best_model.state_dict(), 'models/alexnet_car.pth')
