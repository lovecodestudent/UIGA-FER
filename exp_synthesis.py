# -*- coding:utf-8 -*-
# Created Time: 2018/05/24 10:18
# Author: Xi Zhang <zhangxi2019@ia.ac.cn>

from config_dataset import config

from nets import Encoder, Decoder, Decoder_label, Discriminator, VGGLoss

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from itertools import chain
import random
from torch.utils.data import Dataset, DataLoader
import time


class ELEGANT(object):
    def __init__(self, args,
                 config=config, \
                 encoder=Encoder, decoder=Decoder, decoder_label=Decoder_label, discriminator=Discriminator):

        self.args = args
        self.attributes = args.attributes
        self.n_attributes = len(self.attributes)
        self.gpu = args.gpu

        # init dataset and networks
        self.config = config
        
        self.Enc = encoder()
        self.Dec = decoder_label(self.config.label_copy, self.n_attributes)
        
        self.restore_from_file()
        self.set_mode_and_gpu()

    def restore_from_file(self):
        ckpt_file_enc = self.config.model_dir + '/Enc_raf.pth' # '/Enc_multipie.pth' 
        assert os.path.exists(ckpt_file_enc)
        ckpt_file_dec = self.config.model_dir + '/Dec_raf.pth' # 'Dec_multipie.pth'
        assert os.path.exists(ckpt_file_dec)
        if self.gpu:
            self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
            self.Dec.load_state_dict(torch.load(ckpt_file_dec), strict=False)
        else:
            self.Enc.load_state_dict(torch.load(ckpt_file_enc, map_location='cpu'), strict=False)
            self.Dec.load_state_dict(torch.load(ckpt_file_dec, map_location='cpu'), strict=False)
        print("Load pre-trained model successfully!")

    def set_mode_and_gpu(self):
        self.Enc.eval()
        self.Dec.eval()
        if self.gpu:
            with torch.cuda.device(0):
                self.Enc.cuda()
                self.Dec.cuda()
        if len(self.gpu) > 1:
            self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
            self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))

    def tensor2var(self, tensors, volatile=False):
        if not hasattr(tensors, '__iter__'): tensors = [tensors]
        out = []
        for tensor in tensors:
            if len(self.gpu):
                tensor = tensor.cuda(0)
            out.append(tensor)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_attr_chs(self, encodings, attribute_id):
        num_chs = encodings.size(1)
        per_chs = float(num_chs) / self.n_attributes
        start = int(np.rint(per_chs * attribute_id))
        end = int(np.rint(per_chs * (attribute_id + 1)))
        return encodings.narrow(1, start, end-start)
    
    def get_attr_chs_B(self, encodings, block_id):
        num_chs = encodings.size(0)
        per_chs = float(num_chs) / self.n_attributes
        start = int(np.rint(per_chs * block_id))
        end = int(np.rint(per_chs * (block_id + 1)))
        return encodings.narrow(0, start, end-start)

    def forward_G(self):
        self.z_A, self.A_skip = self.Enc(self.A, return_skip=True)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)

        self.z_C = torch.cat([self.get_attr_chs(self.z_A, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_B, i)  for i in range(self.n_attributes)], 1)
        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_A, i)  for i in range(self.n_attributes)], 1)
        
        for idx in range(self.z_C.size(0)):
            self.z_C[idx] = torch.cat([self.get_attr_chs_B(self.z_C[idx], i) if i != self.attribute_y_B[idx] \
                                  else self.get_attr_chs_B(self.z_B[idx], i) for i in range(self.n_attributes)], 0)
            self.z_D[idx] = torch.cat([self.get_attr_chs_B(self.z_D[idx], i) if i != self.attribute_y_B[idx] \
                                  else self.get_attr_chs_B(self.z_A[idx], i) for i in range(self.n_attributes)], 0)

        self.change_vector_A = torch.FloatTensor(self.z_A.shape[0],self.n_attributes).zero_()
        self.change_vector_C = torch.FloatTensor(self.z_C.shape[0],self.n_attributes).zero_()
        for i in range(self.z_A.shape[0]):
            self.change_vector_C[i][self.attribute_id] = 1
            self.change_vector_C[i][self.attribute_y_B[i]] = 1

        self.change_vector_C_new = self.change_vector_C
        self.change_vector_A_new = self.change_vector_A

        for i in range(self.config.label_copy-1):
            self.change_vector_C_new = torch.cat([self.change_vector_C_new, self.change_vector_C], 1)
            self.change_vector_A_new = torch.cat([self.change_vector_A_new, self.change_vector_A], 1)
          
        if self.gpu:
            self.change_vector_C = self.change_vector_C_new.cuda(0)
            self.change_vector_A = self.change_vector_A_new.cuda(0)

        self.R_A = self.Dec(self.z_A, self.z_A, self.change_vector_A, skip=self.A_skip)
        self.R_B = self.Dec(self.z_B, self.z_B, self.change_vector_A, skip=self.B_skip)
        self.R_C = self.Dec(self.z_C, self.z_A, self.change_vector_C, skip=self.A_skip)
        self.R_D = self.Dec(self.z_D, self.z_B, self.change_vector_C, skip=self.B_skip)

        self.A1 = torch.clamp(self.A + self.R_A, -1, 1)
        self.B1 = torch.clamp(self.B + self.R_B, -1, 1)
        self.C  = torch.clamp(self.A + self.R_C, -1, 1)
        self.D  = torch.clamp(self.B + self.R_D, -1, 1)


    def img_denorm(self, img, scale=255):
        return (img + 1) * scale / 2.

    def transform(self, *images):
        transform1 = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        transform2 = lambda x: x.view(1, *x.size()) * 2 - 1
        out = [transform2(transform1(image)) for image in images]
        return out

    def swap(self,attribute_id, attribute_id_b,test_input, test_target):
        '''
        swap attributes of two images.
        '''
        self.attribute_id = attribute_id
        self.attribute_y_B = torch.IntTensor(1).zero_()

        self.attribute_y_B[0] = attribute_id_b
        self.B, self.A = self.tensor2var(self.transform(Image.open(test_input), Image.open(test_target)), volatile=True)

        self.forward_G()
        img = torch.cat((self.B, self.A, self.D, self.C), -1)
        img = np.transpose(self.img_denorm(img.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('./result.jpg')
        print("Done!The generated image is saved as result.jpg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attributes', nargs='+', type=str, help='Specify attribute names.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
    parser.add_argument('--swap_list', default=[], nargs='+', type=int, help='Specify the attributes ids for swapping.')
    parser.add_argument('--input',type=str,help='Specify the input image(Ia).')
    parser.add_argument('--target',type=str,help='Specify the target image(Ib).')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    model = ELEGANT(args)

    assert len(args.swap_list) == 2
    attribute_id = int(args.swap_list[0])
    attribute_id_b = int(args.swap_list[1])

    model.swap(attribute_id, attribute_id_b, args.input, args.target)


if __name__ == "__main__":
    main()
