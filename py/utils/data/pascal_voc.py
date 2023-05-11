# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午2:51
@file: pascal_voc.py
@author: zj
@description: 加载PASCAL VOC 2007数据集
"""

import cv2
import numpy as np
#VOCDetection是PyTorch数据集类，
# 用于PASCAL VOC数据集。
from torchvision.datasets import VOCDetection

'''
下面的函数用来画包围盒
'''
def draw_box_with_text(img, xmin, ymin, xmax, ymax, text):
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
    cv2.putText(img, "{}".format(text), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

if __name__ == '__main__':
    """
    下载PASCAL VOC数据集
    """
    #../../data是当前脚本的上上级文件夹中的data文件夹
    #dataset = VOCDetection('../../data', year='2007', image_set='trainval', download=True)#没下载数据集的时候用这个文件
    dataset = VOCDetection('../../data', year='2007', image_set='trainval')#本地已经下载好了数据集，用这个文件

    print("数据集数目",len(dataset))
    img, target = dataset.__getitem__(1023)
    img = np.array(img)

    print("目标检测数据集的目标",target)
    xmin = int(target['annotation']['object'][0]['bndbox']['xmin'])
    ymin = int(target['annotation']['object'][0]['bndbox']['ymin'])
    xmax = int(target['annotation']['object'][0]['bndbox']['xmax'])
    ymax = int(target['annotation']['object'][0]['bndbox']['ymax'])
    text="sofa"

    draw_box_with_text(img, xmin, ymin, xmax, ymax,text)
    print("选取图片的大小",img.shape)

    cv2.imshow('img', img)
    cv2.waitKey(0)
