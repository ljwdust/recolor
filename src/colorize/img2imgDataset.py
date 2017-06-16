#!/usr/bin/env python

import numpy as np
import chainer
'''
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
'''
import six
import os

from chainer import cuda, optimizers, serializers, Variable
import cv2
import utility
# import pdb
# pdb.set_trace()


def cvt2YUV(img):
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor( img, cv2.COLOR_RGB2YUV )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2YUV )
    return img


def cvt2LAB(img, normalized=False):
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor( img, cv2.COLOR_RGB2LAB )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2LAB )
    if normalized:
        img = np.asarray(img, np.float32)
        img[:,:,0] = img[:,:,0] / 255 * 100
        img[:,:,1:] -= 128
    return img


def RGB2HSV(img):
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    img = np.asarray(img, np.float32)
    img[:,:,0] *= 2  # from [0, 180] to [0, 360]
    # img[:,:,1:] /= 255  # from [0, 255] to [0, 1]
    return img


def HSV2RGB(img):
    img[:,:,0] /= 2
    # img[:,:,1:] *= 255
    img = np.asarray(img, np.uint8)
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor( img, cv2.COLOR_HSV2RGB )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_HSV2RGB )
    return img


def resizeImage(img):
    sy = img.shape[0] 
    sy = sy - sy % 16
    sx = img.shape[1]
    sx = sx - sx % 16
    img = cv2.resize(img, (sx, sy), interpolation=cv2.INTER_AREA)
    return img


# sampling discrete dots from strokes
# TODO: dispose different colors respectively
def cvtStroke2Dot(img):
    imgdot = np.zeros(img.shape, dtype=np.uint8)
    b, g, r, a = cv2.split(img)

    cnt = 0
    selected = True
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if a[x][y] != 0:
                cnt += 1
                if selected:
                    for ch in range(4):
                        imgdot[x][y][ch] = img[x][y][ch]
                    selected = False
                if cnt > 10:    # sampling interval
                    selected = True
                    cnt = 0

    return imgdot


class ImageAndRefDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./ref', dtype=np.float32, colormode='LAB',
                        normalized=False):
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._dtype = dtype
        self._colormode = colormode
        self._norm = normalized

    def __len__(self):
        return len(self._paths)

    def get_example(self, i, minimize=False, blur=0, s_size=128):
        path1 = os.path.join(self._root1, self._paths[i])
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        print("load:" + path1, os.path.isfile(path1), image1 is None)
        image1 = np.asarray(image1, self._dtype)

        # resize image
        _image1 = image1.copy()
        if s_size < np.max(image1.shape[0:2]):
            if image1.shape[0] < image1.shape[1]:
                s0 = s_size
                s1 = int(image1.shape[1] * (s_size / image1.shape[0]))
                s1 = s1 - s1 % 16    #  s1 can be devided by 16 
                # _s0 = 4 * s0
                # _s1 = int(image1.shape[1] * ( _s0 / image1.shape[0]))
                # _s1 = (_s1+8) - (_s1+8) % 16
            else:
                s1 = s_size
                s0 = int(image1.shape[0] * (s_size / image1.shape[1]))
                s0 = s0 - s0 % 16
                # _s1 = 4 * s1
                # _s0 = int(image1.shape[0] * ( _s1 / image1.shape[1]))
                # _s0 = (_s0+8) - (_s0+8) % 16

            # _image1 = cv2.resize(_image1, (_s1, _s0), interpolation=cv2.INTER_AREA)
            image1 = cv2.resize(image1, (s1, s0), interpolation=cv2.INTER_AREA)
        else:
            image1 = utility.resizeImage(image1, 16)


        if self._colormode == 'LAB':
            if self._norm:
                image1 -= 128

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if _image1.ndim == 2:
            _image1 = _image1[:, :, np.newaxis]

        if self._colormode == 'YUV':
            image1 = np.insert(image1, 1, -512, axis=2)
            image1 = np.insert(image1, 2, 128, axis=2)
            image1 = np.insert(image1, 3, 128, axis=2)
        elif self._colormode == 'LAB':
            if self._norm:
                image1 = np.insert(image1, 1, 0, axis=2)
                image1 = np.insert(image1, 2, 0, axis=2)
            else:
                image1 = np.insert(image1, 1, 128, axis=2)
                image1 = np.insert(image1, 2, 128, axis=2)
            

        # add color ref image
        path_ref = os.path.join(self._root2, self._paths[i])

        ########### INTER_NEAREST may cause tiny stroke disappear #######
        image_ref = cv2.imread(path_ref, cv2.IMREAD_UNCHANGED)
        image_ref = cv2.resize(image_ref, (image1.shape[1], image1.shape[
                               0]), interpolation=cv2.INTER_NEAREST)
        # TODO: turn strokes to dots
        image_ref = cvtStroke2Dot(image_ref)

        b, g, r, a = cv2.split(image_ref)
        if self._colormode == 'YUV':
            image_ref = cvt2YUV( cv2.merge((b, g, r)) )
        elif self._colormode == 'LAB':
            image_ref = cvt2LAB( cv2.merge((b, g, r)), self._norm )

        if self._colormode == 'YUV':
            for x in range(image1.shape[0]):
                for y in range(image1.shape[1]):
                    if a[x][y] != 0:
                        for ch in range(3):
                            image1[x][y][ch + 1] = image_ref[x][y][ch]
        elif self._colormode == 'LAB':
            for x in range(image1.shape[0]):
                for y in range(image1.shape[1]):
                    if a[x][y] != 0:
                        for ch in range(2):
                            image1[x][y][ch + 1] = image_ref[x][y][ch + 1]

        return image1.transpose(2, 0, 1), _image1.transpose(2, 0, 1)


class Image2ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./target', dtype=np.float32, leak=(16, 32), train=False, colormode='LAB'):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._dtype = dtype
        self._leak = leak
        self._img_dict = {}
        self._train = train
        self._colormode = colormode

    def __len__(self):
        return len(self._paths)

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        bin_r = 0.9     # the rate of reading argmeated data

        readed = False
        # if np.random.rand() < bin_r:     # read argmented data
        #     if np.random.rand() < 0.3:
        #         path1 = os.path.join(self._root1 + "_b2r/", self._paths[i])
        #     else:
        #         path1 = os.path.join(self._root1 + "_cnn/", self._paths[i])
        #     path2 = os.path.join(self._root2 + "_b2r/", self._paths[i])
        #     image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)     # read gray image
        #     image2 = cv2.imread(path2, cv2.IMREAD_COLOR)            # read color image
        #     if image1 is not None and image2 is not None:
        #         if image1.shape[0] > 0 and image1.shape[1] and image2.shape[0] > 0 and image2.shape[1]:
        #             readed = True
        if not readed:            # read original data
            # path1 = os.path.join(self._root1, self._paths[i])
            # path2 = os.path.join(self._root2, self._paths[i])
            path1 = self._root1 + self._paths[i]
            path2 = self._root2 + self._paths[i]
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)    # read gray image
            image2 = cv2.imread(path2, cv2.IMREAD_COLOR)           # read color image

        if image1 is None:
            image1 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # crop and resize
        cropsize = 512
        finalsize = 128  # if -1, not resize (remain cropsize), or 128, 256
        if image1.shape[0] == cropsize:
            cropy = 0
        else:
            cropy = np.random.randint(0, image1.shape[0] - cropsize)

        if image1.shape[1] == cropsize:
            cropx = 0
        else:
            cropx = np.random.randint(0, image1.shape[1] - cropsize)
            
        image1 = image1[cropy:cropy+cropsize, cropx:cropx+cropsize]
        image2 = image2[cropy:cropy+cropsize, cropx:cropx+cropsize]
        if finalsize > 0:
            image1 = cv2.resize(image1, (finalsize, finalsize), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (finalsize, finalsize), interpolation=cv2.INTER_AREA)

        # image1 = cv2.resize(image1, (finalsize, finalsize), interpolation=cv2.INTER_AREA)
        # image2 = cv2.resize(image2, (finalsize, finalsize), interpolation=cv2.INTER_AREA)

        # if self._train and np.random.rand() < 0.2:    # data argmentation: image thresholding of gray image
        #     ret, image1 = cv2.threshold(
        #         image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # add flip
        if np.random.rand() > 0.5:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
        # if np.random.rand() > 0.9:
        #     image1 = cv2.flip(image1, 0)
        #     image2 = cv2.flip(image2, 0)

        # change color in HSV color space
        image2 = RGB2HSV(image2)
        r = np.random.rand()
        if r < 0.4:
            n = 0 
        elif r < 0.7:
            n = np.random.randint(-60, 60) 
        else:
            n = np.random.randint(60, 300) 
        image2[:,:,0] += n
        image2[:,:,0] %= 360
        image2 = HSV2RGB(image2)

        # cvt color must in type uint8
        if self._colormode == 'YUV':
            image2 = cvt2YUV( image2 )
        elif self._colormode == 'LAB':
            image2 = cvt2LAB( image2 )
            image2 = image2[:,:,1:]      # only ab channels
        else:
            print('ERROR! Unexpected color mode!!!')

        image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2, self._dtype)

        # add noise
        noise = np.random.normal(
            0, 5 * np.random.rand(), image1.shape).astype(self._dtype)
        image1 += noise
        noise = np.random.normal(
            0, 5 * np.random.rand(), image2.shape).astype(self._dtype)
        image2 += noise
        if self._colormode == 'YUV':
            noise = np.random.normal(0, 16)
            image1 += noise
            image1[image1 < 0] = 0
        elif self._colormode == 'LAB':
            noise = np.random.normal(0, 16)
            image1 += noise
            image1[image1 < 0] = 0
            image1[image1 > 255] = 255
        else:
            print('ERROR! Unexpected color mode!!!')

        # abstract mean                         # NORMALIZATION: uncommon these lines
        # if self._colormode == 'LAB':
        #     image1 -= 128
        # else:
        #     print('ERROR! Unexpected color mode!!!')

        # dimension 3
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]

        # insert stroke channels
        if self._colormode == 'YUV':
            image1 = np.insert(image1, 1, -512, axis=2) # why -512???
            image1 = np.insert(image1, 2, 128, axis=2)
            image1 = np.insert(image1, 3, 128, axis=2)
        elif self._colormode == 'LAB':
            image1 = np.insert(image1, 1, 128, axis=2)
            image1 = np.insert(image1, 2, 128, axis=2)
            # image1 = np.insert(image1, 1, 0, axis=2)          # NORMALIZATION: replace above with these 2 lines
            # image1 = np.insert(image1, 2, 0, axis=2)
        else:
            print('ERROR! Unexpected color mode!!!')

        # randomly add target image px
        if self._leak[1] > 0:
            # random the stroke number n
            r = np.random.rand()
            if r < 0.4:
                n = 0       # no stroke
            elif r < 0.7:
                n = np.random.randint(1, self._leak[0])    # a few strokes
            else:
                n = np.random.randint(self._leak[0], self._leak[1])  # n is the number of strokes

            x = np.random.randint(1, image1.shape[0] - 1, n)   # generate n number randomly
            y = np.random.randint(1, image1.shape[1] - 1, n)
            if self._colormode == 'YUV':
                for i in range(n):
                    for ch in range(3):
                        d = 20
                        v = image2[x[i]][y[i]][ch] + np.random.normal(0, 5)     # these two lines add noise, but don't know how???
                        v = np.floor(v / d + 0.5) * d
                        image1[x[i]][y[i]][ch + 1] = v      # set color as strokes
                        if np.random.rand() > 0.5:
                            image1[x[i]][y[i] + 1][ch + 1] = v
                            image1[x[i]][y[i] - 1][ch + 1] = v
                        if np.random.rand() > 0.5:
                            image1[x[i] + 1][y[i]][ch + 1] = v
                            image1[x[i] - 1][y[i]][ch + 1] = v
            elif self._colormode == 'LAB':
                for i in range(n):
                    for ch in range(2):
                        v = image2[x[i]][y[i]][ch] + np.random.normal(0, 2)
                        image1[x[i]][y[i]][ch + 1] = v      # set color as strokes
                        if np.random.rand() > 0.5:
                            image1[x[i]][y[i] + 1][ch + 1] = v
                            image1[x[i]][y[i] - 1][ch + 1] = v
                        if np.random.rand() > 0.5:
                            image1[x[i] + 1][y[i]][ch + 1] = v
                            image1[x[i] - 1][y[i]][ch + 1] = v
            else:
                print('ERROR! Unexpected color mode!!!')

        image1 = (image1.transpose(2, 0, 1))
        image2 = (image2.transpose(2, 0, 1))
        #image1 = (image1.transpose(2, 0, 1) -128) /128
        #image2 = (image2.transpose(2, 0, 1) -128) /128

        return image1, image2  # ,vec


class Image2ImageDatasetX2(Image2ImageDataset):

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        path1 = os.path.join(self._root1, self._paths[i])
        path2 = os.path.join(self._root2, self._paths[i])
        #image1 = ImageDataset._read_image_as_array(path1, self._dtype)
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        image2 = cvt2YUV(image2)
        image2 = np.asarray(image2, self._dtype)
        name1 = os.path.basename(self._paths[i])
        vec = self.get_vec(name1)

        # add flip and noise
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.8:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        if self._train:
            bin_r = 0.3
        if np.random.rand() < bin_r:
            ret, image1 = cv2.threshold(
                image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _image1 = image1.copy()
        _image2 = image2.copy()
        image1 = cv2.resize(image1, (128, 128), interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(image2, (128, 128), interpolation=cv2.INTER_AREA)

        image1 = np.asarray(image1, self._dtype)
        _image1 = np.asarray(_image1, self._dtype)

        if self._train:
            noise = np.random.normal(0, 5, image1.shape).astype(self._dtype)
            image1 = image1 + noise
            noise = np.random.normal(0, 5, image2.shape).astype(self._dtype)
            image2 = image2 + noise
            noise = np.random.normal(
                0, 4 * np.random.rand(), _image1.shape).astype(self._dtype)
            noise += np.random.normal(0, 24)
            _image1 = _image1 + noise
            _image1[_image1 < 0] = 0
            _image1[_image1 > 255] = 255

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]
        if _image1.ndim == 2:
            _image1 = _image1[:, :, np.newaxis]
        if _image2.ndim == 2:
            _image2 = _image2[:, :, np.newaxis]

        image1 = np.insert(image1, 1, -512, axis=2)
        image1 = np.insert(image1, 2, 128, axis=2)
        image1 = np.insert(image1, 3, 128, axis=2)

        # randomly add terget image px
        if self._leak[1] > 0:
            image0 = image1
            n = np.random.randint(self._leak[0], self._leak[1])
            x = np.random.randint(1, image1.shape[0] - 1, n)
            y = np.random.randint(1, image1.shape[1] - 1, n)
            for i in range(n):
                for ch in range(3):
                    d = 20
                    v = image2[x[i]][y[i]][ch] + np.random.normal(0, 5)
                    #v = np.random.normal(128,40)
                    v = np.floor(v / d + 0.5) * d
                    image1[x[i]][y[i]][ch + 1] = v
                    if np.random.rand() > 0.5:
                        image1[x[i]][y[i] + 1][ch + 1] = v
                        image1[x[i]][y[i] - 1][ch + 1] = v
                    if np.random.rand() > 0.5:
                        image1[x[i] + 1][y[i]][ch + 1] = v
                        image1[x[i] - 1][y[i]][ch + 1] = v

        return image1.transpose(2, 0, 1), image2.transpose(2, 0, 1), _image1.transpose(2, 0, 1), _image2.transpose(2, 0, 1)
