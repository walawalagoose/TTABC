import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
from data.cifar_dataset import CIFAR10C, CIFAR100C, CIFAR10_Dataset, CIFAR100_Dataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

import ipdb

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'imagenetc': 'imagenet-c',
    'cifar10c': 'CIFAR-10-C',
    'cifar100c': 'CIFAR-100-C',
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
}

distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur',
                'zoom_blur', 'frost',
                'brightness', 'contrast', 'elastic_transform',
                'pixelate','fog','speckle_noise','saturate', 'spatter', 'gaussian_blur']

corruption_types = ["gaussian_noise", "shot_noise", "impulse_noise", 
                   "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                   "snow", "frost", "fog", "brightness", 
                   "contrast", "elastic_transform", "pixelate", "jpeg_compression",
                   "speckle_noise", "saturate", "spatter", "gaussian_blur",]


def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, corruption_type=None, corruption_level=None):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)

    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    elif 'imagenetc' in set_id:
        if corruption_type is None or corruption_type not in corruption_types:
            raise ValueError(f"Supported corruption types are: {corruption_types}")
        if corruption_level is None or corruption_level not in [1,2,3,4,5]:
            raise ValueError("corruption_level should be an integer between 1 and 5.")
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], corruption_type, str(corruption_level))
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif 'cifar10c' in set_id or 'cifar100c' in set_id:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        if set_id == 'cifar10c':
            testset = CIFAR10C(testdir, corruption_type, corruption_level, transform)
        elif set_id == 'cifar100c':
            testset = CIFAR100C(testdir, corruption_type, corruption_level, transform)
        else:
            raise NotImplementedError
    elif set_id in ['cifar10', 'cifar100']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        if set_id == 'cifar10':
            testset = CIFAR10_Dataset(testdir, split=mode, transform=transform, download=True)
        elif set_id == 'cifar100':
            testset = CIFAR100_Dataset(testdir, split=mode, transform=transform, download=True)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views



