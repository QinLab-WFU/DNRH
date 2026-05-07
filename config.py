import argparse
from os import path as osp


def get_config():
    parser = argparse.ArgumentParser(description=osp.basename(osp.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50", help="see _network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--multi-thread", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output-old", help="directory to output-old results")
    parser.add_argument("--n-bits", type=int, default=16, help="length of hashing binary")

    parser.add_argument("--graph-mode", type=str, default="dynamic",
                        choices=["full", "topk", "adaptive_once", "dynamic"],
                        help="graph construction strategy")
    parser.add_argument("--graph-k", type=int, default=30, help="neighbor number for sparse graph")
    parser.add_argument("--graph-t", type=int, default=3, help="iteration number for dynamic refinement")

    # special settings
    parser.add_argument("--alpha", type=float, default=0.01, help="hyper-parameter of Eq. 15")
    parser.add_argument('--noise_rate', type=float, default=0.2, help='标签噪声比例，0表示不加噪声')
    parser.add_argument('--mapping', type=float, default={0: 1, 1: 2}, help='asy noise')
    args = parser.parse_args()

    # args.rename = True

    # code
    # args.optimizer = "adam"
    # args.lr = 3e-5

    return args
