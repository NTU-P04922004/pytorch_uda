import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml


def read_yaml(file_path):
    yaml_data = None
    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def create_folders(config_name, base_path="runs", extra_folders=[]):
    date_str = datetime.now().strftime("%Y%m%d-%H%M")

    checkpoint_path = os.path.join(base_path,
                                       config_name + "_" + date_str)
    if not os.path.exists(checkpoint_path):
        print("Creating directory: {}".format(checkpoint_path))
        os.makedirs(checkpoint_path)

    for folder_name in extra_folders:
        dir_path = os.path.join(checkpoint_path, folder_name)
        if not os.path.exists(dir_path):
            print("Creating directory: {}".format(dir_path))
            os.makedirs(dir_path)

    return checkpoint_path


def cv2_image_to_tensor(img):
    img = img.astype(np.float32)
    img /= 255
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    return img_tensor


def tensor_to_cv2_image(img_tensor, size=(128, 128)):
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def seed_everything(seed, debug=False):
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

    torch.backends.cudnn.deterministic = debug
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results
    torch.backends.cudnn.benchmark = not debug
