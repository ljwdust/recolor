#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os
from PIL import Image

from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import argparse

import unet
import lnet
import cv2

#from images_dict import img_dict
from img2imgDataset import Image2ImageDataset

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


def main():
    parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='/media/ljw/Research/research/Deep_Learning/data/Places2/',
                        help='Directory of image files.')
    # parser.add_argument('--dataset', '-i', default='/home/ljw/deep_learning/intercolorize/data/farm/',
    #                     help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=5000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--colormode', default='LAB',
                        help='Color mode')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    root = args.dataset
    #model = "./model_paint"

    if args.colormode == 'YUV':
        cnn = unet.UNET()
        dis = unet.DIS()
    elif args.colormode == 'LAB':
        cnn = unet.UNET(inputChannel=3, outputChannel=2)
        dis = unet.DIS(inputChannel=2)
    else:
        print('ERROR! Unexpected color mode!!!')

    # l = lnet.LNET()
    # serializers.load_npz("../models/liner_f", l)   # load pre-trained model to l

    dataset = Image2ImageDataset(
        "/media/ljw/Research/research/Deep_Learning/data/Places2/filelist_places365-standard/places365_train_outdoor_color512-all.txt", 
        root + "/", root + "data_large", train=True, colormode=args.colormode)  # the class of dataset
    # dataset = Image2ImageDataset(
    #     "/home/ljw/deep_learning/intercolorize/data/farm/color_512.txt",
    #     root + "gray/", root + "color/", train=True, colormode=args.colormode)  # the class of dataset
    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        cnn.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()  # Copy the model to the GPU
        # l.to_gpu()

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.0001)  # use the Adam
    opt.setup(cnn)
    opt.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_cnn')  # what does this used for???

    opt_d = chainer.optimizers.Adam(alpha=0.0001)
    opt_d.setup(dis)
    opt_d.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')

    # Set up a trainer
    updater = ganUpdater(
        colormode=args.colormode,
        models=(cnn, dis),
        iterator={
            'main': train_iter,
            #'test': test_iter
        },
        optimizer={
            'cnn': opt,
            'dis': opt_d},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    snapshot_interval2 = (args.snapshot_interval * 2, 'iteration')
    trainer.extend(extensions.dump_graph('cnn/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval2)
    trainer.extend(extensions.snapshot_object(
        cnn, 'cnn_128_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'cnn_128_dis_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'cnn/loss', 'cnn/loss_rec', 'cnn/loss_adv', 'cnn/loss_tag', 'cnn/loss_l', 'dis/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(os.path.join(args.out, args.resume), trainer)

    trainer.run()

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(args.out, 'model_final'), cnn)
    chainer.serializers.save_npz(os.path.join(args.out, 'optimizer_final'), opt)


class ganUpdater(chainer.training.StandardUpdater):

    def __init__(self, colormode, *args, **kwargs):
        self.cnn, self.dis = kwargs.pop('models')
        self._iter = 0
        super(ganUpdater, self).__init__(*args, **kwargs)
        self._colormode = colormode

    def YUV2Gray(self, img):
        img = img.data.get()
        img = img.transpose(0, 2, 3, 1)
        imgr = np.zeros(img.shape[0:3])
        for i in range(img.shape[0]):
            imgtemp = img[i,:]
            imgtemp = cv2.cvtColor( imgtemp, cv2.COLOR_YUV2RGB )
            imgr[i,:] = cv2.cvtColor( imgtemp, cv2.COLOR_RGB2GRAY )

        xp = self.cnn.xp
        img = xp.zeros(imgr.shape).astype("f")
        img.set(imgr.astype(np.float32))
        return Variable(img)

    def Lab2Gray(self, img, x_in):
        img = img.data.get()        
        img = img.transpose(0, 2, 3, 1)
        # img += 128            # NORMALIZATION: uncommon this line

        x_in = x_in.data.get()
        x_in = x_in.transpose(0, 2, 3, 1)
        # x_in += 128           # NORMALIZATION: uncommon this line

        imgres = np.zeros(img.shape[0:3])
        imgtemp = np.zeros((img.shape[1], img.shape[2], 3))
        for i in range(img.shape[0]):
            imgtemp[:,:,0] = x_in[i,:,:,0]
            imgtemp[:,:,1:] = img[i,:]
            imgtemp = np.asarray(imgtemp, np.uint8)
            imgtemp = cv2.cvtColor( imgtemp, cv2.COLOR_LAB2RGB )
            imgres[i,:] = cv2.cvtColor( imgtemp, cv2.COLOR_RGB2GRAY )

        xp = self.cnn.xp
        img = xp.zeros(imgres.shape).astype("f")
        img.set(imgres.astype(np.float32))
        return Variable(img)


    def loss_cnn(self, cnn, x_out, t_out, y_out, x_in, lam1=1, lam2=1, lam3=10):   # important!!!============
        loss_rec = lam1 * (F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2 * y_out   # y_out is the loss itself
        if self._colormode == 'YUV':
            l_t = self.YUV2Gray(t_out)
            l_x = self.YUV2Gray(x_out)
        elif self._colormode == 'LAB':
            l_t = self.Lab2Gray(t_out, x_in)
            l_x = self.Lab2Gray(x_out, x_in)
        loss_l = lam3 * (F.mean_absolute_error(l_x, l_t))
        loss = loss_rec + loss_adv + loss_l
        chainer.report({'loss': loss, "loss_rec": loss_rec,
                        'loss_adv': loss_adv, "loss_l": loss_l}, cnn)
        # chainer.report({'loss': loss, "loss_rec": loss_rec,
        #                 'loss_adv': loss_adv}, cnn)

        return loss

    def loss_dis(self, dis, y_in, y_out):
        L1 = y_in
        L2 = y_out
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):  # override
        xp = self.cnn.xp  # array module for this link, return numpy or cuda.cupy (what does xp mean???)
        self._iter += 1

        # get the next batch of 'main'
        batch = self.get_iterator('main').next()  
        batchsize = len(batch)

        # size of in and out
        s_in = batch[0][0].shape
        s_out = batch[0][1].shape

        # x_in is the training input, t_out is the groundtruth label
        x_in = xp.zeros((batchsize, s_in[0], s_in[1], s_in[2])).astype("f")
        t_out = xp.zeros((batchsize, s_out[0], s_out[1], s_out[2])).astype("f")

        # get the training data and label
        for i in range(batchsize):   
            x_in[i, :] = xp.asarray(batch[i][0])
            t_out[i, :] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)
        t_out = Variable(t_out)

        x_out = self.cnn.calc(x_in, test=False)  # forward process, x_out is the result

        cnn_optimizer = self.get_optimizer('cnn')
        dis_optimizer = self.get_optimizer('dis')

        # invoke __call__ in class DIS (discriminator)
        # get the loss of x_out and 0
        y_target = self.dis(x_out, Variable(
            xp.zeros(batchsize, dtype=np.int32)), test=False)

        # update the parameters, call loss function and the backward() method to compute the gradients
        # "cnn, x_out, t_out, y_target" is the arguments of the loss function
        # x_out: gen_result, t_out: groundtruth, 
        cnn_optimizer.update(self.loss_cnn, self.cnn, x_out, t_out, y_target, x_in)  

        x_out.unchain_backward()
        # fake: set to ones; real: set to zeros
        y_fake = self.dis(x_out,  Variable(
            xp.ones(batchsize, dtype=np.int32)), test=False)
        y_real = self.dis(t_out,  Variable(
            xp.zeros(batchsize, dtype=np.int32)), test=False)
        dis_optimizer.update(self.loss_dis, self.dis, y_real, y_fake)

if __name__ == '__main__':
    main()
