# PyTorch UDA

## Overview

This an unofficial implementation of the NeurIPS 2020 paper ```Unsupervised Data Augmentation (UDA)```.

## Results

### CIFAR-10 with 4,000 labeled samples

Augmentation     | Paper        | Reproduced   |
---------------- | :----------: | :----------: |
Crop and flip    | 10.94        | 10.93        |
CutOut           | 5.43         | 6.00         |

* The number means error rates on CIFAR-10 test set
* 46,000 unlabeled samples are also used for semi-supervised learning

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

See train.py for more information

## References

- [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)

## License
