import os
import pickle
import random
import time

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import check_integrity

import utils
from ppcls_randaugment import RandAugment


class CIFAR10Dataset(Dataset):
    # Based on torchvision
    # https://github.com/pytorch/vision/blob/v0.8.2/torchvision/datasets/cifar.py
    #
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root,
            train=True,
            supervised_count=0
    ):

        self.root = root
        self.train = train
        self.supervised_count = supervised_count
        self.default_transform = self.get_data_transform(self.train, True)
        self.transform = self.get_data_transform(self.train, False)
        self.target_transform = None

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.train and self.supervised_count != 0:
            # TODO: make the semantic meaning of supervised_count more clear
            # "supervised_count" is used to specify how many data to be used in semi or
            # normal supervised training.
            if self.supervised_count > 0:
                self.data = self.data[:self.supervised_count]
            elif self.supervised_count < 0:
                self.data = self.data[self.supervised_count * -1:]

        self._load_meta()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # load_enter = time.perf_counter()

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.train and (self.transform is not None):
            transformed_img = self.transform(img)
        else:
            transformed_img = self.default_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = self.default_transform(img)

        # load_elapsed = time.perf_counter() - load_enter
        # print("LOAD", "S" if self.supervised_count > 0 else "US", load_enter, load_elapsed)

        return img, transformed_img, target

    def __len__(self):
        return len(self.data)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    @staticmethod
    def get_data_transform(train, is_simple_aug=True):
        # CIFAR10 data statatistics from
        # https://github.com/google-research/uda/blob/master/image/randaugment/augmentation_transforms.py#L40-L43
        mean = (0.49139968, 0.48215841, 0.44653091)
        std = (0.24703223, 0.24348513, 0.26158784)
        transform = None
        transform_list = []
        if train:
            # Default agumentation
            transform_list += [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip()
            ]

        # Convert image to tensor
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        # Add CutOut augmentation
        if train and not is_simple_aug:
            transform_list += [
                transforms.RandomErasing(scale=(0.25, 0.25), ratio=(1.0, 1.0), inplace=False),
            ]

        transform = transforms.Compose(transform_list)

        return transform
