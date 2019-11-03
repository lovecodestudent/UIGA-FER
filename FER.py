# -*- coding:utf-8 -*-
# Created Time: 2018/05/24 10:18
# Author: Xi Zhang <zhangxi2019@ia.ac.cn>

import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib

import os
import argparse
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from nets import vgg19_bn_fx as VGG
from nets import Encoder

from dataset_nature import config_VGG, SingleDataset_VGG
from dataset_nature import config as config_ele

class Classify(object):
    def __init__(self, args, config=config_VGG, config_ele=config_ele, net=VGG):
        self.args = args
        self.config = config_VGG
        self.config_ele = config_ele
        if self.args.raf:
            image_f = open(self.config_ele.data_dir + '/raf/images_test.list')
            datadir = self.config_ele.data_dir + "/raf/data/"
            checkpoint = torch.load(self.config_ele.model_dir + '/FER_raf.t7')
            ckpt_file_enc = self.config_ele.model_dir + '/FER_enc_raf.pth'
        if self.args.multipie:
            image_f = open(self.config_ele.data_dir + "/multipie/images_test.list")
            datadir = self.config_ele.data_dir + "/multipie/data/"
            checkpoint = torch.load(self.config_ele.model_dir + '/FER_multipie.t7')
            ckpt_file_enc = self.config_ele.model_dir + '/FER_enc_multipie.pth'

        # deal with the dataloader
        self.test_im_names = []
        self.test_labels = []
        for line in image_f:
            pic_name = line.strip().split()[0]
            if pic_name[:4] == 'test': 
                self.test_im_names.append(line.strip().split()[0])#[:-4]+'.png')
                self.test_labels.append(int(line.strip().split()[1]))        
        self.dataset_test = SingleDataset_VGG(self.test_im_names, self.test_labels, self.config, datadir)
        self.test_loader = DataLoader(dataset = self.dataset_test, batch_size = self.config.ncwh[0], shuffle = self.config.shuffle, num_workers = self.config.num_workers)

        self.gpu = args.gpu
        self.use_cuda = torch.cuda.is_available()
        self.correct_list = [0,0,0,0,0,0,0]

        self.net = VGG(pretrained=False)
        self.net.load_state_dict(checkpoint['net'])
        self.Enc = Encoder()
        assert os.path.exists(ckpt_file_enc)
        self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
        self.Enc.eval()
        self.Enc.cuda()
        print("Load pre-trained model successfully!")

        if self.use_cuda:
            with torch.cuda.device(0):
                self.net.cuda()
            if len(self.args.gpu)>1 :
                self.net = torch.nn.DataParallel(self.net, device_ids=range(len(self.gpu)))
            cudnn.benchmark = True

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=5e-4)

    def plotCM(self, classes, matrix, savname):
        matrix = matrix.astype(numpy.float)
        linesum = matrix.sum(1)
        linesum = numpy.dot(linesum.reshape(-1, 1), numpy.ones((1, matrix.shape[1])))
        matrix /= linesum
        # plot
        plt.switch_backend('agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        for i in range(matrix.shape[0]):
            ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
        ax.set_xticklabels([''] + classes, rotation=90)
        ax.set_yticklabels([''] + classes)
        plt.savefig(savname)
        print("CM end!")


    def test(self):
        self.net.eval()
        for test_iter,(inputs_,_,targets_) in enumerate(self.test_loader):
            inputs = Variable(inputs_)
            targets_ = list(targets_)
            targets = Variable(torch.Tensor(targets_))
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            fx_test = self.Enc(inputs, return_skip=False)
            fx_test = torch.cat([fx_test, fx_test],1)
            outputs = self.net(inputs, fx_test)

            _, predicted = torch.max(outputs.data, 1)

            if test_iter == 0:
                all_pred = predicted.cpu()
                all_targ = targets.long().data.cpu()
            else:
                all_pred = torch.cat([all_pred, predicted.cpu()])
                all_targ = torch.cat([all_targ, targets.long().data.cpu()])

            # RAF-DB
            if self.args.raf:
                exp_list = ['surp','fear','disg','happy','sad','angry','nature']
                num_list = [329, 74, 160, 1185, 478, 162, 680]
            # Multi-PIE
            if self.args.multipie:
                exp_list = ['disgust', 'scream', 'smile', 'squint', 'surprise', 'neutral']
                num_list = [230, 239, 319, 203, 203, 337] #140

            for idx,item in enumerate(predicted):
                if item == targets.long().data[idx]:
                    self.correct_list[item] += 1
        # confision matrix
        M = confusion_matrix(all_targ.numpy().tolist(), all_pred.numpy().tolist()).astype(numpy.float32)
        self.plotCM(exp_list, M, './CM-result.jpg')

        acc_list = []
        for i in range(len(exp_list)):
            acc_list.append(round(100*M[i][i]/num_list[i],2))

        print(exp_list)
        print("The accuracy for each expression is : ")
        print(acc_list)
        print("The average accuracy is : " + str(sum(self.correct_list) / sum(num_list)))

def main():
    # 获取参数
    parser = argparse.ArgumentParser(description='PyTorch VGG19 Training')
    parser.add_argument('-g', '--gpu', default=[], type=str, help='Specify GPU ids.')  
    parser.add_argument('--multipie', action='store_true', help='resume from checkpoint')
    parser.add_argument('--raf', action='store_true', help='resume from checkpoint')

    args = parser.parse_args()
    print(args)

    assert args.multipie + args.raf == 1
  
    if len(args.gpu)>0: 
      os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu[0]

    model = Classify(args)

    model.test()
    torch.cuda.empty_cache() 

if __name__ == "__main__":
    main()

