# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午2:43
@file: pascal_voc_car.py
@author: zj
@description: 从PASCAL VOC 2007数据集中抽取类别Car。保留1/10的数目
"""
import os
import sys
import shutil
import random
import numpy as np
import xmltodict



'''
#这个导入语句的含义是：“从当前包的父级包中导入 module1 模块”。这个运行也会出错，弃用
from ...utils.util import check_dir
'''

from py.utils.util import check_dir
#作者给的原导入方式不行
# from utils.util import check_dir

suffix_xml = '.xml'
suffix_jpeg = '.jpg'
car_train_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
car_val_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'
voc_annotation_dir = '../../data/VOCdevkit/VOC2007/Annotations/'
voc_jpeg_dir = '../../data/VOCdevkit/VOC2007/JPEGImages/'
car_root_dir = '../../data/voc_car/'

def parse_train_val(data_path):
    """
    解析的最终目的：把所有正样本的图像的编号，都存入samples列表中
    """
    samples = []
    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            res = line.strip().split(' ')
            # 如果是正样本如'000012  1'可以拆分成['000012', '', '1']
            if len(res) == 3 and int(res[2]) == 1:
                samples.append(res[0])

    return np.array(samples)#返回的是正样本的合集

#随机采样样本，减少数据集个数（留下1/10）下面
def sample_train_val(samples):
    """
    随机采样样本，减少数据集个数（留下1/10）
    """
    for name in ['train', 'val']:
        dataset = samples[name]
        length = len(dataset)
        '''
        这行代码使用“random.sample()”函数生成一个随机整数列表，范围为0到“length-1”。
        该函数的第二个参数指定要生成的随机整数数量，在此代码中为“int(length / 10)”（即原数据集的10%）
        '''
        random_samples = random.sample(range(length), int(length / 10))
        # print(random_samples)
        new_dataset = dataset[random_samples]
        samples[name] = new_dataset

    return samples


# def parse_car(sample_list):
#     """
#     遍历所有的标注文件，筛选包含car的样本
#     """
#
#     car_samples = list()
#     for sample_name in sample_list:
#         annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
#         with open(annotation_path, 'rb') as f:
#             xml_dict = xmltodict.parse(f)
#             # print(xml_dict)
#
#             bndboxs = list()
#             objects = xml_dict['annotation']['object']
#             if isinstance(objects, list):
#                 for obj in objects:
#                     obj_name = obj['name']
#                     difficult = int(obj['difficult'])
#                     if 'car'.__eq__(obj_name) and difficult != 1:
#                         car_samples.append(sample_name)
#             elif isinstance(objects, dict):
#                 obj_name = objects['name']
#                 difficult = int(objects['difficult'])
#                 if 'car'.__eq__(obj_name) and difficult != 1:
#                     car_samples.append(sample_name)
#             else:
#                 pass
#
#     return car_samples


def save_car(car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    """
    保存类别Car的样本图片和标注文件
    """
    for sample_name in car_samples:
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml)
        shutil.copyfile(src_annotation_path, dst_annotation_path)

        src_jpeg_path = os.path.join(voc_jpeg_dir, sample_name + suffix_jpeg)
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpeg)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    csv_path = os.path.join(data_root_dir, 'car.csv')
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')


if __name__ == '__main__':
    '''
    下面一行代码把car_train_path训练数据集中的正样本和’train‘对应，
    把car_val_path验证数据集中的正样本和’val‘对应
    '''
    #parse_train_val此函数作用在于把所有正样本存到一个列表中，train包含的是训练样本数据的
    samples = {'train': parse_train_val(car_train_path), 'val': parse_train_val(car_val_path)}
    print(samples)
    print('train\'s lengh',len(samples['train']))#376条数据，训练
    print('val\'s lengh', len(samples['val']))#377条数据，验证

    '''
    
    '''

    # samples = sample_train_val(samples)#通过sample_train_val(samples)函数，样本规模已缩小为原1/10
    # print('after,train\'s lengh',len(samples['train']))
    # print('after,val\'s lengh', len(samples['val']))
    # print(samples)


#把samples数据全部存入到新的文件夹中，方便后续处理。
    check_dir(car_root_dir)
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)
        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')

        check_dir(data_root_dir)
        check_dir(data_annotation_dir)
        check_dir(data_jpeg_dir)
        save_car(samples[name], data_root_dir, data_annotation_dir, data_jpeg_dir)

    print('done')
