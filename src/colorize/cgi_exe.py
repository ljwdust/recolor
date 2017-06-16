#!/usr/bin/env python


import numpy as np
import chainer
import cv2

#import chainer.functions as F
#import chainer.links as L
#import six
#import os

from chainer import cuda, serializers, Variable  # , optimizers, training
#from chainer.training import extensions
#from train import Image2ImageDataset

from img2imgDataset import ImageAndRefDataset

import unet
import lnet
# import pdb
# pdb.set_trace()


class Painter:

    def __init__(self, gpu=0, colormode='LAB', normalized=False):

        print("start")
        self.root = "./images/"
        self.batchsize = 1
        self.outdir = self.root + "out/"
        self.outdir_min = self.root + "out_min/"
        self.gpu = gpu
        self._dtype = np.float32
        self._colormode = colormode
        self._norm = normalized

        print("load model")
        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            cuda.set_max_workspace_size(64 * 1024 * 1024)  # 64MB
            chainer.Function.type_check_enable = False
        if self._colormode == 'YUV':
            self.cnn_128 = unet.UNET()
            # self.cnn_512 = unet.UNET()
        elif self._colormode == 'LAB':
            self.cnn_128 = unet.UNET(inputChannel=3, outputChannel=2)
            # self.cnn_512 = unet.UNET()

        if self.gpu >= 0:
            self.cnn_128.to_gpu()
            # self.cnn_512.to_gpu()
        if self._colormode == 'YUV':
            serializers.load_npz("./src/colorize/models/unet_128_standard-YUV", self.cnn_128)
        elif self._colormode == 'LAB':
            serializers.load_npz("./src/colorize/models/cnn_128_iter_370000", self.cnn_128)
        # serializers.load_npz(
        #     "./cgi-bin/paint_x2_unet/models/unet_512_standard", self.cnn_512)


    def save_as_img(self, array, name):
        array = array.transpose(1, 2, 0)
        array = array.clip(0, 255).astype(np.uint8)
        if self.gpu >= 0:
            array = cuda.to_cpu(array)
        (major, minor, _) = cv2.__version__.split(".")
        if major == '3':
            img = cv2.cvtColor(array, cv2.COLOR_YUV2RGB)
        else:
            img = cv2.cvtColor(array, cv2.COLOR_YUV2BGR)
        cv2.imwrite(name, img)


    def colorize(self, id_str, step='C', blur=0, s_size=128, colorize_format="jpg"):
        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()

        dataset = ImageAndRefDataset(
            [id_str + ".png"], self.root + "line/", self.root + "ref/", 
            colormode=self._colormode, normalized=self._norm)

        # s_size is the size of final result, could be modified
        sample = dataset.get_example(0, minimize=True, blur=blur, s_size=128)

        if self._colormode == 'YUV':
            sample_container = np.zeros(
                (1, 4, sample[0].shape[1], sample[0].shape[2]), dtype='f')
        elif self._colormode == 'LAB':
            sample_container = np.zeros(
                (1, 3, sample[0].shape[1], sample[0].shape[2]), dtype='f')
        sample_container[0, :] = sample[0]

        if self.gpu >= 0:
            sample_container = cuda.to_gpu(sample_container)

        image_conv2d_layer = self.cnn_128.calc(Variable(sample_container, volatile='on'), test=True)
        del sample_container

        # if step == 'C':
        #     input_bat = np.zeros((1, 4, sample[1].shape[1], sample[1].shape[2]), dtype='f')
        #     print(input_bat.shape)
        #     input_bat[0, 0, :] = sample[1]

        #     output = cuda.to_cpu(image_conv2d_layer.data[0])
        #     del image_conv2d_layer  # release memory

        #     for channel in range(3):
        #         input_bat[0, 1 + channel, :] = cv2.resize(
        #             output[channel, :], 
        #             (sample[1].shape[2], sample[1].shape[1]), 
        #             interpolation=cv2.INTER_CUBIC)

        #     if self.gpu >= 0:
        #         link = cuda.to_gpu(input_bat, None)
        #     else:
        #         link = input_bat
        #     image_conv2d_layer = self.cnn_512.calc(Variable(link, volatile='on'), test=True)
        #     del link  # release memory

        image_out_path = self.outdir + id_str + "_0." + colorize_format

        if self._colormode == 'YUV':
            convdata = image_conv2d_layer.data[0]
            self.save_as_img(convdata, image_out_path)
        elif self._colormode == 'LAB':
            if self._norm:
                convdata = image_conv2d_layer.data[0] + 128
            else:
                convdata = image_conv2d_layer.data[0]

            ab = convdata.transpose(1, 2, 0)
            ab = ab.clip(0, 255).astype(np.uint8)
            if self.gpu >= 0:
                ab = cuda.to_cpu(ab)

            L = sample[1].transpose(1, 2, 0)
            lab = np.zeros((L.shape[0], L.shape[1], 3), dtype='f')
            lab[:,:,0] = L[:,:,0]
            lab[:,:,1:] = cv2.resize(ab, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_AREA)
            lab = lab.astype(np.uint8)

            res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            cv2.imwrite(image_out_path, res)

        del image_conv2d_layer



if __name__ == '__main__':
    for n in range(1):
        p = Painter()
        print(n)
        p.colorize(n * p.batchsize)
