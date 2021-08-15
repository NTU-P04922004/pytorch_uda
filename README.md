# PyTorch UDA

## Overview

This is an unofficial implementation of the NeurIPS 2020 paper ```Unsupervised Data Augmentation (UDA)```.

## Results

### Error rates on CIFAR-10 test set

Augmentation     | Paper        | Reproduced   |
---------------- | :----------: | :----------: |
Crop and flip    | 10.94        | 10.93        |
CutOut           | 5.43         | 6.00         |

* Setting: CIFAR-10 4000
* Training with 4,000 labeled samples and 46,000 unlabeled samples

## Requirements

* Python >= 3.6
* PyTorch >= 1.5
* torchvision >= 0.6
* numpy
* Pillow
* ruamel.yaml
* sklearn
* tqdm

## Usage

See ```train.py``` and config files in the ```config``` folder for more information

## References

- [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)

## License
GPLv3
