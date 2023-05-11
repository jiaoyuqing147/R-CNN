# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 上午9:54
@file: finetune.py
@author: zj
@description: 
"""

import os
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from d2l.mxnet import show_images
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import AlexNet_Weights#预训练的时候加载AlexNet时会用到

from py.utils.data.custom_finetune_dataset import CustomFinetuneDataset
from py.utils.data.custom_batch_sampler import CustomBatchSampler
from py.utils.util import check_dir
import mxnet as mx#为了使用show_images 函数

def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)
        data_set = CustomFinetuneDataset(data_dir, transform=transform)#自定义处理数据集
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)#128组成一组（batch）
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True)#最后的数据不够128，扔掉drop_last=True

        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()

    return data_loaders, data_sizes


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders, data_sizes = load_data('./data/classifier_car')

    #PyTorch中的AlexNet模型分为5个卷积层和3个全连接层，
    #model = models.alexnet(pretrained=True)#为什么要做预训练,这种写法已经被改动了，换成了下面这行代码
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)#使用预训练的参数

    print(model)
    '''
    torchvision模块中的models子模块中的alexnet函数提供了一个预训练的AlexNet模型，
    该模型已经在ImageNet数据集上进行了预训练，并获得了很高的性能。
    当我们设置pretrained=True时，函数会自动下载并加载在ImageNet数据集上预训练的权重参数，
    这些参数包含了在ImageNet数据集上提取特征的经验，可以在其他任务上进行微调。
    因此，使用预训练模型，我们可以避免从头开始训练模型所需的大量计算资源和时间，
    并且可以利用在ImageNet数据集上获得的优秀的性能来提高模型在其他数据集上的性能。
    '''


    data_loader = data_loaders["train"]
    print("一次迭代取得所有的正负数据，如果是多个类则取得多类数据集合")

    inputs, targets = next(data_loader.__iter__())
    print(inputs[0].size(), type(inputs[0]))
    trans = transforms.ToPILImage()#用于将输入数据转换为PIL.Image格式，以便于可视化或者其他处理
    print(type(trans(inputs[0])))


    print(targets)
    print(inputs.shape)
    titles = ["TRUE" if i.item() else "False" for i in targets[0:60]]
    #images = [np.array(trans(i)) for i in inputs[0:60]]#show_images函数只能处理MXNet NDArray，而不显示NumPy数组，所以改用下面的代码
    images = [mx.nd.array(trans(i)) for i in inputs[0:60]]
    show_images(images, titles=titles, num_rows=5, num_cols=12)

    #把AlexNet变成二分类模型，在最后一行改为分成2类
    num_features = model.classifier[6].in_features#获取原始AlexNet模型中全连接层的输入特征数
    model.classifier[6] = nn.Linear(num_features, 2)# 修改AlexNet模型的最后一个全连接层，将其输出从1000个类别改为2个类别（汽车和非汽车）。
    # print(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()#定义交叉熵损失函数。
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)#定义随机梯度下降（SGD）优化器，用于优化模型参数，学习率为0.001，动量为0.9。
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)#定义学习率调度器，每7个epochs将学习率降低为原来的0.1。

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=25)
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.pth')
