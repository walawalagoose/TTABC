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
    Automatically download and load CIFAR-10 using torchvision.
    - By default, train+test are merged; you can specify "train" or "test" using the split parameter.
    - If transform is None, ToTensor + Normalize (CIFAR-10 statistics) will be applied.
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

        # Download/Load
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

        # Default transformation
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
    Automatically download and load CIFAR-100 using torchvision.
    - By default, train+test are merged; you can specify "train" or "test" using the split parameter.
    - If transform is None, ToTensor + Normalize (CIFAR-100 statistics) will be applied.
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
    root: Directory containing labels.npy and files like brightness.npy, gaussian_noise.npy, etc.
    corruption: A string or sequence specifying the corruption types; if a sequence, multiple corruptions will be concatenated.
    severity: 1..5, specifying the severity level; can also be a sequence like [1,3,5], which will concatenate in order.
    transform: Optional torchvision.transforms, used for PIL or tensor; if None, defaults to ToTensor+Normalize.
    return_index: Whether to return the sample's index in the original test set (0..9999), useful for paired evaluations.
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
        # Normalize the input
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        for s in severity:
            if s < 1 or s > 5:
                raise ValueError("severity must be between 1 and 5")
        # Load labels (shape: 10000,)
        self.labels = np.load(self.root / "labels.npy")  # int64/ int32 are both acceptable

        self.samples = []  # list of (np.ndarray, label, idx)
        for c in corruption:
            arr = np.load(self.root / f"{c}.npy")  # Expected shape (50000,32,32,3)
            # If your files are split by severity (10000,32,32,3), replace with:
            # arr = np.load(self.root / f"{c}_{severity}.npy") and remove the start/end slicing.
            if arr.ndim != 4 or arr.shape[1:4] != (32, 32, 3):
                # Support (50000,32,32,3) or (10000,32,32,3)
                if arr.shape == (10000, 32, 32, 3):
                    # When the user provides a single severity file, treat it as severity=[1] by default
                    pass
                else:
                    raise ValueError(f"Shape of {c}.npy is invalid: {arr.shape}")
            for s in severity:
                if arr.shape[0] == 50000:
                    start = (s - 1) * 10000
                    end = s * 10000
                    x = arr[start:end]  # (10000,32,32,3) uint8
                else:
                    # Single severity file
                    x = arr  # (10000,32,32,3)
                # Align with labels
                for i in range(10000):
                    self.samples.append((x[i], int(self.labels[i]), i))

        # Pre-create mean/standard deviation tensors for normalization
        self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, target, base_idx = self.samples[idx]  # img_np: (32,32,3) uint8
        if self.transform is None:
            # Default: Convert to tensor and normalize using CIFAR-10 statistics
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).to(self.to_tensor_dtype) / 255.0
            img = (img - self.mean) / self.std
        else:
            # If using torchvision.transforms, requires PIL.Image or Tensor
            img = Image.fromarray(img_np)
            img = self.transform(img)
        return img, target

class CIFAR100C(Dataset):
    """
    root: Directory containing labels.npy and files like brightness.npy, gaussian_noise.npy, etc.
    corruption: A string or sequence specifying the corruption types; if a sequence, multiple corruptions will be concatenated.
    severity: 1..5, specifying the severity level; can also be a sequence like [1,3,5], which will concatenate in order.
    transform: Optional torchvision.transforms, used for PIL or tensor; if None, defaults to ToTensor+Normalize.
    return_index: Whether to return the sample's index in the original test set (0..9999), useful for paired evaluations.
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
        # Normalize the input
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        for s in severity:
            if s < 1 or s > 5:
                raise ValueError("severity must be between 1 and 5")
        # Load labels (shape: 10000,)
        self.labels = np.load(self.root / "labels.npy")  # int64/ int32 are both acceptable

        self.samples = []  # list of (np.ndarray, label, idx)
        for c in corruption:
            arr = np.load(self.root / f"{c}.npy")  # Expected shape (50000,32,32,3)
            # If your files are split by severity (10000,32,32,3), replace with:
            # arr = np.load(self.root / f"{c}_{severity}.npy") and remove the start/end slicing.
            if arr.ndim != 4 or arr.shape[1:4] != (32, 32, 3):
                # Support (50000,32,32,3) or (10000,32,32,3)
                if arr.shape == (10000, 32, 32, 3):
                    # When the user provides a single severity file, treat it as severity=[1] by default
                    pass
                else:
                    raise ValueError(f"Shape of {c}.npy is invalid: {arr.shape}")
            for s in severity:
                if arr.shape[0] == 50000:
                    start = (s - 1) * 10000
                    end = s * 10000
                    x = arr[start:end]  # (10000,32,32,3) uint8
                else:
                    # Single severity file
                    x = arr  # (10000,32,32,3)
                # Align with labels
                for i in range(10000):
                    self.samples.append((x[i], int(self.labels[i]), i))

        # Pre-create mean/standard deviation tensors for normalization
        self.mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(CIFAR100_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, target, base_idx = self.samples[idx]  # img_np: (32,32,3) uint8
        if self.transform is None:
            # Default: Convert to tensor and normalize using CIFAR-100 statistics
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).to(self.to_tensor_dtype) / 255.0
            img = (img - self.mean) / self.std
        else:
            # If using torchvision.transforms, requires PIL.Image or Tensor
            img = Image.fromarray(img_np)
            img = self.transform(img)
        return img, target

if __name__ == '__main__':
    pass

            