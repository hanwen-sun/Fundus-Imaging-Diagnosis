# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午1:54

from unet_model.unet import UNET
from dataset import ISBI_Dataset
import torch.optim as optim
import torch.nn as nn
import torch

def train_net(net,device,data_path,epochs=40,batch_size=1,lr=1e-5):
    isbi_dataset = ISBI_Dataset(data_path)
    train_loader = torch.utils.data.DataLoader(isbi_dataset,
                                               batch_size,
                                               shuffle = True)
    #使用RMSprop优化
    optimizer = optim.RMSprop(net.parameters(),lr,weight_decay=1e-8,momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.to(device,dtype = torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            pred = net(images)

            loss = criterion(pred,labels)
            print('epoch:%d  train loss:%f' % (epoch+1,loss.item()))
            if loss <best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    device = torch.device('cuda:0')
    net = UNET(n_channels=1,n_classes=1)
    net.to(device)
    data_path = r'data/train/'
    train_net(net,device,data_path)
