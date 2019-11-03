# -*- coding:utf-8 -*-
# Created Time: 2018/05/24 10:28
# Author: Xi Zhang <zhangxi2019@ia.ac.cn>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os,time
from PIL import Image

class Config:
    @property
    def data_dir(self):
        data_dir = './datasets/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    @property
    def model_dir(self):
        model_dir ='./models/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    nchw = [12,3,256,256]
    label_copy = 20


config = Config()

class Config_VGG:
    ncwh = [12,3,256,256]
    num_workers = 4
    shuffle = True
    lr = 0.001

config_VGG = Config_VGG()

class SingleDataset_VGG(Dataset):
    def __init__(self, im_names, labels, config, datadir):
        self.im_names = im_names
        self.labels = labels
        self.config = config
        self.datadir = datadir

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self,idx):
        image = Image.open(self.datadir + self.im_names[idx])
        fx = torch.FloatTensor(2,4).zero_()

        image = self.transform_test(image)
        label = (self.labels[idx])
        return image, fx, label

    @property
    def transform_test(self):
        transform_test = transforms.Compose([
            transforms.Resize(self.config.ncwh[-2:]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        return transform_test


