import numpy  as np


import py
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from py.utils.util import parse_car_csv

#custom_finetune_dataset中的尝试
# annotation_path = "E:/R-CNN/py/data/classifier_car/train/Annotations/000091_1.csv"
# positive_rects=list()
# positive_sizes=list()
# rects = np.loadtxt(annotation_path, dtype=int, delimiter=' ')
# print(rects.shape[0])
# positive_rects.extend(rects)
# positive_sizes.append(len(rects))
# print(positive_rects)
# print(positive_sizes)
# print(len(positive_rects))

#finetune.py中加载预训练的模型有点问题
import torchvision.models as models
from torchvision.models import AlexNet_Weights

model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
print(model)

# print(rects.shape[1])