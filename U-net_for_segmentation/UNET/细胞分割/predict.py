# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午2:07
import glob
import numpy as np
import torch
import os
import cv2
from unet_model.unet import UNET

if __name__ == "__main__":
    device = torch.device('cuda:0')
    net = UNET(n_channels=1,n_classes=1)
    net.to(device)
    net.load_state_dict(torch.load('best_model.pth'))
    net.eval()
    testpaths = glob.glob('data/test/*.png')
    for test_path in testpaths:
        save_res_path = 'result/' + test_path.split('.')[0].split('/')[-1] + '_res.png'
        print(save_res_path)
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1,1,img.shape[0],img.shape[1])
        img = torch.from_numpy(img)
        img = img.to(device,dtype = torch.float32)

        pred = net(img)
        pred = np.array(pred.data.cpu()[0])[0]

        #从二值还原为灰度图
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        cv2.imwrite(save_res_path,pred)
