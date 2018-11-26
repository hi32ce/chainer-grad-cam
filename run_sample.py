from run import *
from path_utility import *
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--arch', '-a',
                   choices=['alex', 'vgg', 'resnet', 'resnext_my'],
                   default='vgg')
    # p.add_argument('--label', '-y', type=int, default=-1)
    p.add_argument('--layer', '-l', default='conv5_3')
    p.add_argument('--model', '-m')
    p.add_argument('--mean')
    p.add_argument('--use-ideep', action='store_true')
    # parser.add_argument('--root', type=str, default=os.path.join(os.path.dirname(__file__), u'sample/'),
    #                     help='content root')
    p.add_argument('--label-name', type=str, default='label.txt', help='label file name')
    p.add_argument('--input-paths', type=str, nargs='+', default=os.path.join(os.path.dirname(__file__), u'in'), help='dir or epub file path')
    p.add_argument('--output-path', type=str, nargs='?', default=os.path.join(os.path.dirname(__file__), u'out'), help='dir path')
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
    (grad_cam, guided_backprop) = init_grad_cam(model)

    label_paths = []
    for input_path in args.input_paths:
        label_paths.extend(
            list(filter(lambda f: f.endswith(args.label_name), list(PathUtility.get_all_file_list(input_path)))))

    targets = dict()
    for label_path in label_paths:
        target_name = os.path.split(os.path.dirname(label_path))[1]
        targets[target_name] = []
        with open(label_path, 'r') as f:
            for line in f:
                line_splited = line.strip().split(' ')
                path_org = line_splited[0].strip()
                path = os.path.join(os.path.dirname(label_path), path_org)
                label = line_splited[1].strip()
                targets[target_name].append((path, label, path_org))

    os.makedirs(args.output_path, exist_ok=True)
    labels = [0, 1, 2]
    for target, values in targets.items():
        print(target)
        for img_path, correct_label_str, path_org in values:
            for target_label in labels:
                print(path_org)
                src_img = load_image(img_path, (model.size, model.size))
                label = target_label
                output_path = "".join(
                    [os.path.splitext(os.path.join(args.output_path, target, path_org))[0], "_correct-", correct_label_str,
                     "_", "target-", str(label), "_"])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                draw(src_img=src_img, output_path=output_path, model=model, mean=mean, label=label, layer=args.layer, grad_cam=grad_cam,
                     guided_backprop=guided_backprop)


if __name__ == '__main__':
    main()
