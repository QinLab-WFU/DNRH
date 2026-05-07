import configparser
import os.path as osp
import pickle
import platform

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def get_class_num(name):
    r = {"cifar": 10, "flickr": 38, "nuswide": 21, "coco": 80, "imagenet": 100}[name]
    return r


def get_topk(name):
    r = {"cifar": None, "flickr": None, "nuswide": 5000, "coco": None, "imagenet": 1000}[name]
    return r


def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r") as f:
        lines = f.read().splitlines()
    return np.array(lines)


def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.CenterCrop(crop_size)]
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            # T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loaders(name, root, **kwargs):
    train_trans = build_trans("train")
    other_trans = build_trans("other")

    data = init_dataset(name, root)

    train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    # generator=torch.Generator(): to keep torch.get_rng_state() unchanged!
    # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
    query_loader = DataLoader(ImageDataset(data.query, other_trans), generator=torch.Generator(), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), generator=torch.Generator(), **kwargs)

    return train_loader, query_loader, dbase_loader


class BaseDataset(object):
    """
    Base class of dataset
    """

    def __init__(self, name, txt_root, img_root, verbose=True):

        self.img_root = img_root

        self.train_txt = osp.join(txt_root, "train.txt")
        self.query_txt = osp.join(txt_root, "query.txt")
        self.dbase_txt = osp.join(txt_root, "dbase.txt")

        self.check_before_run()

        self.train = self.process(self.train_txt)
        self.query = self.process(self.query_txt)
        self.dbase = self.process(self.dbase_txt)

        self.set_img_abspath()  # 1.jpg -> /home/x/COCO/images/1.jpg

        if verbose:
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics()

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_txt):
            raise RuntimeError("'{}' is not available".format(self.train_txt))
        if not osp.exists(self.query_txt):
            raise RuntimeError("'{}' is not available".format(self.query_txt))
        if not osp.exists(self.dbase_txt):
            raise RuntimeError("'{}' is not available".format(self.dbase_txt))

    def get_imagedata_info(self, data):
        labs = data[1]
        n_cids = (labs.sum(axis=0) > 0).sum()
        n_imgs = len(data[0])
        return n_cids, n_imgs

    def print_dataset_statistics(self):
        n_train_cids, n_train_imgs = self.get_imagedata_info(self.train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(self.query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(self.dbase)

        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")

    def process(self, txt_path):
        imgs, labs = [], []
        for x in open(txt_path, "r").readlines():
            parts = x.split()
            imgs.append(parts[0])
            labs.append(parts[1:])
        imgs = np.array(imgs)
        labs = np.array(labs, dtype=np.float32)
        return (imgs, labs)

    def set_img_abspath(self):
        for x in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, x)
            # imgs = [osp.join(self.img_root, img) for img in imgs]
            imgs = np.char.add(f"{self.img_root}/", imgs)
            setattr(self, x, (imgs, labs))


class CIFAR(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    @staticmethod
    def unpickle(file):
        with open(file, "rb") as fo:
            dic = pickle.load(fo, encoding="latin1")
        return dic

    def set_img_abspath(self):
        # get all img data from data_batch_1~5 & test_batch
        data_list = [f"data_batch_{x}" for x in range(1, 5 + 1)]
        data_list.append("test_batch")
        imgs = []
        for x in data_list:
            data = self.unpickle(osp.join(self.img_root, x))
            imgs.append(data["data"])
            # labs.extend(data["labels"])
        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
        imgs = imgs.transpose((0, 2, 3, 1))

        # change image file name to image data
        for x in ["train", "query", "dbase"]:
            _imgs, _labs = getattr(self, x)
            idxes = [int(x.replace(".png", "")) for x in _imgs]
            setattr(self, x, (imgs[idxes], _labs))


_ds_factory = {
    "cifar": CIFAR,
    "nuswide": BaseDataset,
    "flickr": BaseDataset,
    "coco": BaseDataset,
    "imagenet": BaseDataset,
}


def init_dataset(name, root, **kwargs):

    if name not in list(_ds_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(_ds_factory.keys())))

    txt_root = osp.join(root, name)

    ini_loc = osp.join(root, name, "images", "location.ini")
    if osp.exists(ini_loc):
        config = configparser.ConfigParser()
        config.read(ini_loc)
        if "wfu.edu.cn" in platform.node():
            img_root = config["DEFAULT"]["SLURM"]
        else:
            img_root = config["DEFAULT"][platform.system()]
    else:
        img_root = osp.join(root, name)

    return _ds_factory[name](name, txt_root, img_root, **kwargs)


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        img, lab = self.data[0][idx], self.data[1][idx]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, lab, idx

    def get_all_labels(self):
        return torch.from_numpy(self.data[1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db_name = "imagenet"
    root = "./_datasets"

    dataset = init_dataset(db_name, root)

    trans = T.Compose(
        [
            # T.ToPILImage(),
            T.Resize([224, 224]),
            T.ToTensor(),
        ]
    )

    train_set = ImageDataset(dataset.dbase, trans)
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    concepts = get_concepts(db_name, root)

    for imgs, labs, _ in dataloader:
        print(imgs.shape, labs)
        plt.imshow(imgs[0].numpy().transpose(1, 2, 0))
        titles = concepts[labs[0].nonzero().squeeze(1)]
        plt.title(titles)
        plt.show()
        break
