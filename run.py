import argparse

import chainer
import cv2
import numpy as np

import models
from lib import backprop


def load_model(arch, model_path):
    if arch == 'alex':
        model = models.Alex()
    elif arch == 'vgg':
        model = models.VGG16Layers()
    elif arch == 'resnet':
        model = models.ResNet152Layers()
    elif arch == 'resnext_my':
        model = models.ResNeXt50()
        chainer.serializers.load_npz(model_path, model)
    return model


def load_mean(mean_path):
    mean = None
    if mean_path is not None:
        mean = np.load(mean_path)
    return mean


def init_grad_cam(model):
    grad_cam = backprop.GradCAM(model)
    guided_backprop = backprop.GuidedBackprop(model)
    return (grad_cam, guided_backprop)


def load_image(input, size):
    src_img = cv2.imread(input, 1)
    src_img = cv2.resize(src_img, size)
    return src_img


def draw(src_img, output_path, model, mean, label, layer, grad_cam, guided_backprop):
    predict = None
    if mean is not None:
        x = src_img.transpose(2, 0, 1).astype(np.float32)
        _, h, w = x.shape
        top = (h - model.insize) // 2
        left = (w - model.insize) // 2
        bottom = top + model.insize
        right = left + model.insize
        x -= mean[:, top:bottom, left:right]
        x *= (1.0 / 255.0)  # Scale to [0, 1]
        x = x[np.newaxis, :, :, :]
        with chainer.using_config('train', False):
            predict = model.predict(x).data.argmax()
            output_path += "predict-" + str(predict) + "_"
    else:
        x = src_img.astype(np.float32) - np.float32([103.939, 116.779, 123.68])
        x = x.transpose(2, 0, 1)[np.newaxis, :, :, :]

    gcam = grad_cam.generate(x, label, layer)
    gcam = np.uint8(gcam * 255 / gcam.max())
    gcam = cv2.resize(gcam, (model.size, model.size))
    gbp = guided_backprop.generate(x, label, layer)

    ggcam = gbp * gcam[:, :, np.newaxis]
    ggcam -= ggcam.min()
    ggcam = 255 * ggcam / ggcam.max()
    cv2.imwrite(output_path + 'ggcam.png', ggcam)

    gbp -= gbp.min()
    gbp = 255 * gbp / gbp.max()
    cv2.imwrite(output_path + 'gbp.png', gbp)

    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = np.float32(src_img) + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()
    cv2.imwrite(output_path + 'gcam.png', gcam)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='images/dog_cat.png')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--arch', '-a',
                   choices=['alex', 'vgg', 'resnet', 'resnext_my'],
                   default='vgg')
    p.add_argument('--label', '-y', type=int, default=-1)
    p.add_argument('--layer', '-l', default='conv5_3')
    p.add_argument('--model', '-m')
    p.add_argument('--mean')
    p.add_argument('--use-ideep', action='store_true')
    p.set_defaults(use_ideep=False)
    args = p.parse_args()

    model = load_model(args.arch, args.model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.use_ideep:
        chainer.using_config('use_ideep', 'auto')
        model.to_intel64()

    mean = load_mean(args.mean)
    src_img = load_image(args.input, (model.size, model.size))
    (grad_cam, guided_backprop) = init_grad_cam(model)
    draw(src_img=src_img, output_path="", model=model, mean=mean, label=args.label, layer=args.layer, grad_cam=grad_cam,
         guided_backprop=guided_backprop)


if __name__ == '__main__':
    main()
