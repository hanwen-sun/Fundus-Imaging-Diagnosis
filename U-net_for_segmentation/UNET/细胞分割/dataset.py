# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午1:28
"""
使用的是ISBI细胞分割的数据集，训练集就三十张图像
"""
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Dataset(Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.image_path = glob.glob(os.path.join(data_path,'image/*.png'))
    def augment(self,image,mode):
        """
        :param image:
        :param mode: 1 :水平翻转 0 : 垂直翻转 -1 水平+垂直翻转
        :return:
        """
        file = cv2.flip(image,mode)
        return file
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):
        image_path = self.image_path[index]
        label_path = image_path.replace("image","label")

        #读取
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        #转为灰度图
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        image = image.reshape(1,image.shape[0],image.shape[1])
        label = label.reshape(1,label.shape[0],label.shape[1])

        #标签二值化 ，将255 -> 1
        label = label / 255

        # 随机进行数据增强,2时不做数据增强
        mode = random.choice([-1,0,1,2])
        if mode != 2:
            image = self.augment(image,mode)
            label=self.augment(label,mode)
        return image, label

isbi = ISBI_Dataset("data/train/")
print(len(isbi))
train_loader = torch.utils.data.DataLoader(isbi,
                                           batch_size=2,
                                           shuffle=True)
for image ,label in train_loader:
    print(image.shape)