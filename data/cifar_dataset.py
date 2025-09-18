import json
import os
import csv
import random
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Optional, Sequence, Union, Literal

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader

# debug dataset
import torchvision.transforms as transforms
from torchvision import datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC




CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


class CIFAR10_Dataset(Dataset):
    """
    通过 torchvision 自动下载并加载 CIFAR-10。
    - 默认合并 train+test；可用 split 指定 "train" 或 "test"。
    - 若 transform 为 None，则采用 ToTensor + Normalize(CIFAR-10 统计量)。
    """
    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        to_tensor_dtype: torch.dtype = torch.float32,
        split: Literal["all", "train", "test"] = "all",
        download: bool = True,
    ):
        self.root = str(root)
        self.user_transform = transform
        self.to_tensor_dtype = to_tensor_dtype
        self.split = split

        # 下载/加载
        ds_train = datasets.CIFAR10(self.root, train=True, download=download)
        ds_test  = datasets.CIFAR10(self.root, train=False, download=download)

        if split == "train":
            self.data = ds_train.data  # numpy (N,32,32,3), uint8
            self.targets = np.array(ds_train.targets, dtype=np.int64)
        elif split == "test":
            self.data = ds_test.data
            self.targets = np.array(ds_test.targets, dtype=np.int64)
        else:  # "all"
            self.data = np.concatenate([ds_train.data, ds_test.data], axis=0)
            self.targets = np.array(ds_train.targets + ds_test.targets, dtype=np.int64)

        # 默认变换
        if self.user_transform is None:
            self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
            self.std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
            self.default_to_tensor = True
        else:
            self.default_to_tensor = False

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_np = self.data[idx]  # HWC uint8
        target = int(self.targets[idx])

        if self.default_to_tensor:
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).to(self.to_tensor_dtype) / 255.0
            img = (img - self.mean) / self.std
        else:
            img = Image.fromarray(img_np)
            img = self.user_transform(img)
        return img, target


class CIFAR100_Dataset(Dataset):
    """
    通过 torchvision 自动下载并加载 CIFAR-100。
    - 默认合并 train+test；可用 split 指定 "train" 或 "test"。
    - 若 transform 为 None，则采用 ToTensor + Normalize(CIFAR-100 统计量)。
    """
    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        to_tensor_dtype: torch.dtype = torch.float32,
        split: Literal["all", "train", "test"] = "all",
        download: bool = True,
    ):
        self.root = str(root)
        self.user_transform = transform
        self.to_tensor_dtype = to_tensor_dtype
        self.split = split

        ds_train = datasets.CIFAR100(self.root, train=True, download=download)
        ds_test  = datasets.CIFAR100(self.root, train=False, download=download)

        if split == "train":
            self.data = ds_train.data  # numpy (N,32,32,3), uint8
            self.targets = np.array(ds_train.targets, dtype=np.int64)
        elif split == "test":
            self.data = ds_test.data
            self.targets = np.array(ds_test.targets, dtype=np.int64)
        else:  # "all"
            self.data = np.concatenate([ds_train.data, ds_test.data], axis=0)
            self.targets = np.array(ds_train.targets + ds_test.targets, dtype=np.int64)

        if self.user_transform is None:
            self.mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
            self.std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
            self.default_to_tensor = True
        else:
            self.default_to_tensor = False

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_np = self.data[idx]
        target = int(self.targets[idx])

        if self.default_to_tensor:
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).to(self.to_tensor_dtype) / 255.0
            img = (img - self.mean) / self.std
        else:
            img = Image.fromarray(img_np)
            img = self.user_transform(img)
        return img, target

class CIFAR10C(Dataset):
    """
    root: 目录包含 labels.npy 与如 brightness.npy, gaussian_noise.npy 等
    corruption: 字符串或序列，指定腐蚀类型；若为序列，将把多个腐蚀拼接在一起
    severity: 1..5，选择严重度；也可为序列，如 [1,3,5]，则按顺序拼接
    transform: 可选 torchvision.transforms，用于 PIL 或 tensor；若为 None，默认做 ToTensor+Normalize
    return_index: 是否返回样本在原测试集中的索引（0..9999），便于做配对评估
    """
    def __init__(self, 
                 root: Union[str, Path], 
                 corruption: Union[str, Sequence[str]] = "brightness", 
                 severity: Union[int, Sequence[int]] = 5, 
                 transform=None,
                 to_tensor_dtype=torch.float32,):
        self.root = Path(root)
        self.transform = transform
        self.to_tensor_dtype = to_tensor_dtype
        # 规范化输入
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        for s in severity:
            if s < 1 or s > 5:
                raise ValueError("severity 必须在 1..5 之间")
        # 读取标签（10000,）
        self.labels = np.load(self.root / "labels.npy")  # int64/ int32 均可

        self.samples = []  # list of (np.ndarray, label, idx)
        for c in corruption:
            arr = np.load(self.root / f"{c}.npy")  # 期望 shape (50000,32,32,3)
            # 如果你的文件是每严重度一个文件(10000,32,32,3)，请改为：
            # arr = np.load(self.root / f"{c}_{severity}.npy") 并去掉 start/end 切片。
            if arr.ndim != 4 or arr.shape[1:4] != (32, 32, 3):
                # 支持 (50000,32,32,3) 或 (10000,32,32,3)
                if arr.shape == (10000, 32, 32, 3):
                    # 当用户传的是单一严重度文件时，默认视为 severity=[1] 的切片
                    pass
                else:
                    raise ValueError(f"{c}.npy 的 shape 异常: {arr.shape}")
            for s in severity:
                if arr.shape[0] == 50000:
                    start = (s - 1) * 10000
                    end = s * 10000
                    x = arr[start:end]  # (10000,32,32,3) uint8
                else:
                    # 单严重度文件
                    x = arr  # (10000,32,32,3)
                # 与 labels 对齐
                for i in range(10000):
                    self.samples.append((x[i], int(self.labels[i]), i))

        # 预创建标准化张量的均值/方差
        self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, target, base_idx = self.samples[idx]  # img_np: (32,32,3) uint8
        if self.transform is None:
            # 默认：转为 tensor 并按 CIFAR-10 统计量标准化
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).to(self.to_tensor_dtype) / 255.0
            img = (img - self.mean) / self.std
        else:
            # 若使用 torchvision.transforms，需要 PIL.Image 或 Tensor
            img = Image.fromarray(img_np)
            img = self.transform(img)
        return img, target

class CIFAR100C(Dataset):
    """
    root: 目录包含 labels.npy 与如 brightness.npy, gaussian_noise.npy 等
    corruption: 字符串或序列，指定腐蚀类型；若为序列，将把多个腐蚀拼接在一起
    severity: 1..5，选择严重度；也可为序列，如 [1,3,5]，则按顺序拼接
    transform: 可选 torchvision.transforms，用于 PIL 或 tensor；若为 None，默认做 ToTensor+Normalize
    return_index: 是否返回样本在原测试集中的索引（0..9999），便于做配对评估
    """
    def __init__(self, 
                 root: Union[str, Path], 
                 corruption: Union[str, Sequence[str]] = "brightness", 
                 severity: Union[int, Sequence[int]] = 5, 
                 transform=None,
                 to_tensor_dtype=torch.float32,):
        self.root = Path(root)
        self.transform = transform
        self.to_tensor_dtype = to_tensor_dtype
        # 规范化输入
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        for s in severity:
            if s < 1 or s > 5:
                raise ValueError("severity 必须在 1..5 之间")
        # 读取标签（10000,）
        self.labels = np.load(self.root / "labels.npy")  # int64/ int32 均可

        self.samples = []  # list of (np.ndarray, label, idx)
        for c in corruption:
            arr = np.load(self.root / f"{c}.npy")  # 期望 shape (50000,32,32,3)
            # 如果你的文件是每严重度一个文件(10000,32,32,3)，请改为：
            # arr = np.load(self.root / f"{c}_{severity}.npy") 并去掉 start/end 切片。
            if arr.ndim != 4 or arr.shape[1:4] != (32, 32, 3):
                # 支持 (50000,32,32,3) 或 (10000,32,32,3)
                if arr.shape == (10000, 32, 32, 3):
                    # 当用户传的是单一严重度文件时，默认视为 severity=[1] 的切片
                    pass
                else:
                    raise ValueError(f"{c}.npy 的 shape 异常: {arr.shape}")
            for s in severity:
                if arr.shape[0] == 50000:
                    start = (s - 1) * 10000
                    end = s * 10000
                    x = arr[start:end]  # (10000,32,32,3) uint8
                else:
                    # 单严重度文件
                    x = arr  # (10000,32,32,3)
                # 与 labels 对齐
                for i in range(10000):
                    self.samples.append((x[i], int(self.labels[i]), i))

        # 预创建标准化张量的均值/方差（CIFAR-100 使用与 CIFAR-10 相同的常用统计量）
        self.mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(CIFAR100_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, target, base_idx = self.samples[idx]  # img_np: (32,32,3) uint8
        if self.transform is None:
            # 默认：转为 tensor 并按 CIFAR-100 统计量标准化
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).to(self.to_tensor_dtype) / 255.0
            img = (img - self.mean) / self.std
        else:
            # 若使用 torchvision.transforms，需要 PIL.Image 或 Tensor
            img = Image.fromarray(img_np)
            img = self.transform(img)
        return img, target

if __name__ == '__main__':
    pass

            