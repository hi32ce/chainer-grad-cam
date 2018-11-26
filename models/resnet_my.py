import collections
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L

# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)
from chainer import initializers


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2, groups=1):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True,
                groups=groups)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('conv2', [self.conv2, self.bn2, F.relu]),
            ('conv3', [self.conv3, self.bn3]),
            ('proj4', [self.conv4, self.bn4]),
            ('relu5', [F.add, F.relu]),
        ])

    def __call__(self, x):
        h = x
        for key, funcs in self.functions.items():
            if key == 'proj4':
                for func in funcs:
                    x = func(x)
            else:
                for func in funcs:
                    if func is F.add:
                        h = func(h, x)
                    else:
                        h = func(h)
        return h


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch, groups=1):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True,
                groups=groups)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('conv2', [self.conv2, self.bn2, F.relu]),
            ('conv3', [self.conv3, self.bn3]),
            ('relu4', [F.add, F.relu]),
        ])

    def __call__(self, x):
        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                if func is F.add:
                    h = func(h, x)
                else:
                    h = func(h)
        return h


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2, groups=1):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride, groups))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch, groups))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):

    insize = 224
    size = 224

    def __init__(self):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(3, 64, 64, 256, 1)
            self.res3 = Block(4, 256, 128, 512)
            self.res4 = Block(6, 512, 256, 1024)
            self.res5 = Block(3, 1024, 512, 2048)
            self.fc = L.Linear(2048, 1000)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, 3, 2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool5', [lambda x: F.average_pooling_2d(x, x.shape[2:], 1)]),
            ('fc6', [self.fc]),
            ('prob', [F.softmax]),
        ])

    def __call__(self, x, layers=['prob']):
        h = chainer.Variable(x)
        activations = {'input': h}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def predict(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h


class ResNeXt50(ResNet50):

    insize = 224
    size = 224

    def __init__(self):
        chainer.Chain.__init__(self)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(3, 64, 128, 256, 1, groups=32)
            self.res3 = Block(4, 256, 256, 512, groups=32)
            self.res4 = Block(6, 512, 512, 1024, groups=32)
            self.res5 = Block(3, 1024, 1024, 2048, groups=32)
            self.fc = L.Linear(2048, 1000)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, 3, 2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool5', [lambda x: F.average_pooling_2d(x, x.shape[2:], 1)]),
            ('fc6', [self.fc]),
            ('prob', [F.softmax]),
        ])
