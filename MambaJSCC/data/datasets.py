# import cv2
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image

### 2026/4/12 wyj: 语义重构
from torchvision.transforms import functional as TF
import random


NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


# class Datasets(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.imgs = []
#         dir = data_dir
#         self.imgs += glob(os.path.join(dir, "*.jpg"))
#         self.imgs += glob(os.path.join(dir, "*.png"))
#         self.imgs.sort()

#     def __getitem__(self, item):
#         image_ori = self.imgs[item]
#         name = os.path.basename(image_ori)
#         image = Image.open(image_ori).convert("RGB")
#         self.im_height, self.im_width = image.size
#         if self.im_height % 128 != 0 or self.im_width % 128 != 0:
#             self.im_height = self.im_height - self.im_height % 128
#             self.im_width = self.im_width - self.im_width % 128
#         self.transform = transforms.Compose(
#             [transforms.CenterCrop((self.im_width, self.im_height)), transforms.ToTensor()]
#         )
#         img = self.transform(image)
#         return img, name

#     def __len__(self):
#         return len(self.imgs)


### 2026/4/9 wyj:
class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        self.imgs += glob(os.path.join(data_dir, "*.jpg"))
        self.imgs += glob(os.path.join(data_dir, "*.png"))
        self.imgs += glob(os.path.join(data_dir, "*.jpeg"))
        self.imgs.sort()

    def __getitem__(self, item):
        image_path = self.imgs[item]
        name = os.path.splitext(os.path.basename(image_path))[0]

        image = Image.open(image_path).convert("RGB")
        w, h = image.size   # PIL返回的是(width, height)

        # 裁成不超过原图、且能被128整除的最大尺寸
        crop_w = w - (w % 128)
        crop_h = h - (h % 128)

        # 防止极端情况下出现0
        if crop_w == 0:
            crop_w = w
        if crop_h == 0:
            crop_h = h

        transform = transforms.Compose([
            transforms.CenterCrop((crop_h, crop_w)),  # 注意这里是(h, w)
            transforms.ToTensor(),
        ])

        img = transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


class Datasets_train(Dataset):
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        self.imgs = []
        dir = data_dir
        self.img_size = img_size
        self.imgs += glob(os.path.join(dir, "*.jpg"))
        self.imgs += glob(os.path.join(dir, "*.png"))
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert("RGB")
        self.im_height, self.im_width = image.size
        if self.im_height < self.img_size or self.im_width < self.img_size:
            crop_size = self.img_size
            self.transform = transforms.Compose(
                [transforms.Resize((crop_size, crop_size)), transforms.ToTensor()]
            )
        else:
            crop_size = self.img_size
            self.transform = transforms.Compose(
                [transforms.RandomCrop((crop_size, crop_size)), transforms.ToTensor()]
            )
        img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


### 2026/4/12 wyj: 语义加权
class MUSegSemanticTrainDataset(Dataset):
    def __init__(self, mine_root, img_size, sem_ids):
        self.mine_root = mine_root
        self.img_size = img_size
        self.sem_ids = set(sem_ids)

        self.image_dir = os.path.join(mine_root, "Image")
        self.label_dir = os.path.join(mine_root, "Label")

        self.img_paths = []
        self.img_paths += glob(os.path.join(self.image_dir, "*.jpg"))
        self.img_paths += glob(os.path.join(self.image_dir, "*.png"))
        self.img_paths += glob(os.path.join(self.image_dir, "*.jpeg"))
        self.img_paths.sort()

    def _get_label_path(self, img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.label_dir, name + "_label.png")

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self._get_label_path(img_path)

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)   # 不要 convert("RGB")

        # 同步随机裁剪
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.img_size, self.img_size)
        )
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # 同步随机翻转
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        image = TF.to_tensor(image)

        # label: [H, W], 每个像素是类别id
        label = torch.from_numpy(np.array(label)).long()

        # 生成二值重点区域 mask
        sem_mask = torch.zeros_like(label, dtype=torch.bool)
        for cls_id in self.sem_ids:
            sem_mask = sem_mask | (label == cls_id)

        sem_mask = sem_mask.float().unsqueeze(0)   # [1, H, W]

        # ## 2026/4/12 wyj调试打印
        #  if idx < 2:
        #     print("img_path =", img_path)
        #     print("label_path =", label_path)
        #     print("image shape =", image.shape)
        #     print("sem_mask shape =", sem_mask.shape)
        #     print("sem_mask min/max =", sem_mask.min().item(), sem_mask.max().item())

        return image, sem_mask

    def __len__(self):
        return len(self.img_paths)
    
    

def get_loader(config):

    if config.DATA.DATASET == "CIFAR10":
        transform_train = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )

        transform_test = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(
            root=config.DATA.train_data_dir, train=True, transform=transform_train, download=False
        )

        test_dataset = datasets.CIFAR10(
            root=config.DATA.test_data_dir, train=False, transform=transform_test, download=False
        )
###2026/4/2 wyj：增加 MUSeg 数据读取代码
    # elif config.DATA.DATASET == "MUSeg":
    #     transform_train = transforms.Compose(
    #         [
    #             transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.ToTensor(),
    #         ]
    #     )

    #     transform_test = transforms.Compose(
    #         [
    #             transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),  #
    #             transforms.ToTensor(),
    #         ]
    #     )

    #     train_dataset = datasets.ImageFolder(
    #         root=config.DATA.train_data_dir,
    #         transform=transform_train,
    #     )
    #     # test_dataset = Datasets(data_dir=config.DATA.test_data_dir)
    #     test_dataset = datasets.ImageFolder(
    #         root=config.DATA.test_data_dir, transform=transform_test
    #     )
    # elif config.DATA.DATASET == "MUSeg":
    #     transform_train = transforms.Compose(
    #         [
    #             transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.ToTensor(),
    #         ]
    #     )

    #     train_dataset = datasets.ImageFolder(
    #         root=config.DATA.train_data_dir,
    #         transform=transform_train,
    #     )
    #     # 整图测试：不再用ImageFolder+CenterCrop(128,128)
    #     test_dataset = Datasets(config.DATA.test_data_dir)


    elif config.DATA.DATASET == "MUSeg":
        if config.TRAIN.SEMANTIC_WEIGHT:
            train_dataset = MUSegSemanticTrainDataset(
                mine_root=config.DATA.train_data_dir,
                img_size=config.DATA.IMG_SIZE,
                sem_ids=config.TRAIN.SEM_IDS,
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
            train_dataset = datasets.ImageFolder(
                root=config.DATA.train_data_dir,
                transform=transform_train,
            )

    # 测试阶段仍然只读整图 Image 文件夹
        test_dataset = Datasets(config.DATA.test_data_dir)

    elif config.DATA.DATASET == "DIV2K":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),  #
                transforms.ToTensor(),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root=config.DATA.train_data_dir,
            transform=transform_train,
        )
        # test_dataset = Datasets(data_dir=config.DATA.test_data_dir)
        test_dataset = datasets.ImageFolder(
            root=config.DATA.test_data_dir, transform=transform_test
        )
    elif config.DATA.DATASET in ["CelebA", "CelebA-HQ", "AFHQ", "Bird"]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(
            root=config.DATA.train_data_dir, transform=transform_train
        )

        test_dataset = datasets.ImageFolder(
            root=config.DATA.test_data_dir, transform=transform_test
        )

    elif config.DATA.DATASET in ["Kodak", "CLIC2021", "OpenImg"]:
        #
        transform_train = transforms.Compose(
            [
                transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        #
        train_dataset = Datasets_train(
            data_dir=config.DATA.train_data_dir, img_size=config.DATA.IMG_SIZE
        )

        test_dataset = Datasets(config.DATA.test_data_dir)
    # print(NUM_DATASET_WORKERS)
    # seed_torch()
    if config.TRAIN.DATA_PARALLEL:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler_train,
            batch_size=config.DATA.TRAIN_BATCH,
            num_workers=NUM_DATASET_WORKERS,  # config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        batch_size=config.DATA.TRAIN_BATCH,
        # worker_init_fn=worker_init_fn_seed,
        shuffle=True,
        drop_last=False,
    )

    
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=config.DATA.TEST_BATCH, shuffle=False
    )

    return train_loader, test_loader



