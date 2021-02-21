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


class CIFAR10Dataset(Dataset):
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
        self.default_transform = self.get_data_transform(False)
        self.transform = self.get_data_transform(self.train)
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

        if self.train:
            if self.supervised_count > 0:
                self.data = self.data[:self.supervised_count]
            else:
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

        # return index, self.targets[index]

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
    def get_data_transform(train):
        mean = (0.49139968, 0.48215841, 0.44653091)
        std = (0.24703223, 0.24348513, 0.26158784)
        # mean = np.array([125.3, 123.0, 113.9]) / 255.0
        # std = np.array([63.0, 62.1, 66.7]) / 255.0
        transform = None
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transform


if __name__ == "__main__":

    from itertools import cycle
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    SUPERVISED_COUNT = 4000
    EPOCHS = 50

    s_dataset = CIFAR10Dataset(root="/home/kuohsin/workspace/dataset", train=True, supervised_count=SUPERVISED_COUNT)
    us_dataset = CIFAR10Dataset(root="/home/kuohsin/workspace/dataset", train=True, supervised_count=SUPERVISED_COUNT * -1)
    # dataset = torch.utils.data.ConcatDataset([s_dataset, us_dataset])
    config_s = {
        "batch_size": 64,
        "num_workers": 0,
        "drop_last": False,
        "shuffle": True
    }

    config_us = {
        "batch_size": 448,
        "num_workers": 0,
        "drop_last": False,
        "shuffle": True
    }
    s_dataloader = torch.utils.data.DataLoader(s_dataset, **config_s)
    us_dataloader = torch.utils.data.DataLoader(us_dataset, **config_us)

    s_index_list = []
    us_index_list = []

    s_label_list = []
    us_label_list = []

    s_data_iter = iter(s_dataloader)
    us_data_iter = iter(us_dataloader)

    # for epoch in range(EPOCHS):
        # for item1, item2 in tqdm(zip(us_dataloader, cycle(s_dataloader))):
    for it in range(1000):
        # for img, label in s_dataloader:

        try:
            item1 = next(s_data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            s_data_iter = iter(s_dataloader)
            item1 = next(s_data_iter)

        try:
            item2 = next(us_data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            us_data_iter = iter(us_dataloader)
            item2 = next(us_data_iter)

        index_1, label_1 = item1
        index_2, label_2 = item2

        # print(img_1.shape, label_1.shape)
        # print(img_2.shape, label_2.shape)

        # img_1 = utils.tensor_to_cv2_image(img_1[0])
        # img_aug_1 = utils.tensor_to_cv2_image(img_aug_1[0])
        # img_2 = utils.tensor_to_cv2_image(img_2[0])
        # cv2.imshow("img_1", img_1)
        # cv2.imshow("img_aug_1", img_aug_1)
        # cv2.imshow("img_2", img_2)
        # cv2.waitKey(0)

        s_index_list.append(index_1)
        us_index_list.append(index_2)
        s_label_list.append(label_1)
        us_label_list.append(label_2)

    # s_index_array = np.array(s_index_list)
    s_index_array = np.concatenate(s_index_list)
    us_index_array = np.concatenate(us_index_list)

    s_label_array = np.concatenate(s_label_list)
    us_label_array = np.concatenate(us_label_list)

    s_frequency_list = [0] * len(s_dataset)
    us_frequency_list = [0] * len(us_dataset)
    for i in s_index_array:
        s_frequency_list[i] += 1
    for i in us_index_array:
        us_frequency_list[i] += 1

    print(s_index_array.shape, us_index_array.shape)

    index_list = [i for i in range(50000)]
    plt.scatter(index_list[:len(s_dataset)], s_frequency_list)
    plt.title("s_frequency_list")
    plt.show()

    plt.hist(s_label_array)
    plt.show()

    plt.scatter(index_list[:len(us_dataset)], us_frequency_list)
    plt.title("us_frequency_list")
    plt.show()

    plt.hist(us_label_array)
    plt.show()

    print(len(us_dataset))
