import numpy as np
import chainer
import cv2

from chainer import cuda, serializers, Variable  # , optimizers, training

import unet
import lnet

from img2imgDataset import ImageAndRefDataset

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='/home/ljw/deep_learning/intercolorize/data/farm/',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    root = args.dataset
    #model = "./model_paint"


    # dataset = Image2ImageDataset(
    #     "/home/ljw/deep_learning/intercolorize/data/farm/color_512.txt", root + "gray/", root + "color/", train=True)  # the class of dataset
    # # dataset.set_img_dict(img_dict)
    # train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

    print("load model")
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        # cnn.to_gpu()  # Copy the model to the GPU

    # cnn_128 = unet.UNET()
    cnn_128 = unet.UNET(inputChannel=3, outputChannel=2)
    # cnn_512 = unet.UNET()
    if args.gpu >= 0:
        cnn_128.to_gpu()
        # cnn_512.to_gpu()
    serializers.load_npz("result/cnn_128_iter_235000", cnn_128)  #test4_LAB/cnn_128_iter_62000
    # serializers.load_npz(
    #     "./cgi-bin/paint_x2_unet/models/unet_512_standard", self.cnn_512)


    path1 = 'test/2.jpg'
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_AREA)
    image1 = np.asarray(image1, np.float32)
    l = image1.copy()
    # image1 -= 128

    if image1.ndim == 2:
        image1 = image1[:, :, np.newaxis]

    image1 = np.insert(image1, 1, 128, axis=2)
    image1 = np.insert(image1, 2, 128, axis=2)
    img = image1.transpose(2, 0, 1)


    x = np.zeros((1, 3, img.shape[1], img.shape[2]), dtype='f')
    x[0,:] = img

    if args.gpu >= 0:
        x = cuda.to_gpu(x)

    # lnn = lnet.LNET()
    y = cnn_128.calc(Variable(x, volatile='on'), test=True)


    ab = y.data[0].transpose(1, 2, 0)
    ab = ab.clip(0, 255).astype(np.uint8)
    if args.gpu >= 0:
        ab = cuda.to_cpu(ab)

    lab = np.zeros((ab.shape[0], ab.shape[1], 3), dtype='f')
    lab[:,:,0] = l
    lab[:,:,1:] = ab
    lab = lab.astype(np.uint8)
    
    res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cv2.imwrite('test/lab.jpg', res)

    # save_as_img(lab, "test/" + "11.jpg")


def save_as_img(array, name):
    array = array.transpose(1, 2, 0)
    array = array.clip(0, 255).astype(np.uint8)
    array = cuda.to_cpu(array)
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor(array, cv2.COLOR_YUV2RGB)
    else:
        img = cv2.cvtColor(array, cv2.COLOR_YUV2BGR)
    cv2.imwrite(name, img)







def get_example(self, i, minimize=False, blur=0, s_size=128):
    path1 = os.path.join(self._root1, self._paths[i])
    #image1 = ImageDataset._read_image_as_array(path1, self._dtype)

    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    print("load:" + path1, os.path.isfile(path1), image1 is None)
    image1 = np.asarray(image1, self._dtype)

    _image1 = image1.copy()
    if minimize:   # resize to 128 and 512
        if image1.shape[0] < image1.shape[1]:
            s0 = s_size
            s1 = int(image1.shape[1] * (s_size / image1.shape[0]))
            s1 = s1 - s1 % 16    #  s1 can be devided by 16 
            _s0 = 4 * s0
            _s1 = int(image1.shape[1] * ( _s0 / image1.shape[0]))
            _s1 = (_s1+8) - (_s1+8) % 16
        else:
            s1 = s_size
            s0 = int(image1.shape[0] * (s_size / image1.shape[1]))
            s0 = s0 - s0 % 16
            _s1 = 4 * s1
            _s0 = int(image1.shape[0] * ( _s1 / image1.shape[1]))
            _s0 = (_s0+8) - (_s0+8) % 16

        _image1 = image1.copy()
        _image1 = cv2.resize(_image1, (_s1, _s0),
                             interpolation=cv2.INTER_AREA)
        #noise = np.random.normal(0,5*np.random.rand(),_image1.shape).astype(self._dtype)

        if blur > 0:
            blured = cv2.blur(_image1, ksize=(blur, blur))
            image1 = _image1 + blured - 255

        image1 = cv2.resize(image1, (s1, s0), interpolation=cv2.INTER_AREA)

    # image is grayscale
    if image1.ndim == 2:
        image1 = image1[:, :, np.newaxis]
    if _image1.ndim == 2:
        _image1 = _image1[:, :, np.newaxis]

    image1 = np.insert(image1, 1, -512, axis=2)
    image1 = np.insert(image1, 2, 128, axis=2)
    image1 = np.insert(image1, 3, 128, axis=2)

    # add color ref image
    path_ref = os.path.join(self._root2, self._paths[i])

    if minimize:
        image_ref = cv2.imread(path_ref, cv2.IMREAD_UNCHANGED)
        image_ref = cv2.resize(image_ref, (image1.shape[1], image1.shape[
                               0]), interpolation=cv2.INTER_NEAREST)
        b, g, r, a = cv2.split(image_ref)
        image_ref = cvt2YUV( cv2.merge((b, g, r)) )

        for x in range(image1.shape[0]):
            for y in range(image1.shape[1]):
                if a[x][y] != 0:
                    for ch in range(3):
                        image1[x][y][ch + 1] = image_ref[x][y][ch]

    else:
        image_ref = cv2.imread(path_ref, cv2.IMREAD_COLOR)
        image_ref = cvt2YUV(image_ref)
        image1 = cv2.resize(
            image1, (4 * image_ref.shape[1], 4 * image_ref.shape[0]), interpolation=cv2.INTER_AREA)
        image_ref = cv2.resize(image_ref, (image1.shape[1], image1.shape[
                               0]), interpolation=cv2.INTER_AREA)

        image1[:, :, 1:] = image_ref

    return image1.transpose(2, 0, 1), _image1.transpose(2, 0, 1)


if __name__ == '__main__':
    main()