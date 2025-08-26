import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import ipdb
import math
import pickle
from datetime import datetime
import os

def create_dataloader(args, set_id):
    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    if args.tpt:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        if args.I_augmix:
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(set_id)>=1)
        else:
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(set_id)>1)
        batchsize = 1
    else:
        data_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = args.batch_size

    print("evaluating: {}".format(set_id))
    # reset the model
    # Reset classnames of custom CLIP model
    if len(set_id) > 1: 
        # fine-grained classification datasets
        classnames = eval("{}_classes".format(set_id.lower()))
    else:
        assert set_id in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if set_id in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
            if set_id == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        
        else:
            classnames = classnames_all

    val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
    
    if args.down_sample_ratio is not None:
        # 获取数据集大小和随机采样索引
        total_size = len(val_dataset)
        sample_size = int(args.down_sample_ratio * total_size)
        indices = np.random.choice(total_size, sample_size, replace=False)
        print("number of test samples: {}".format(sample_size))
        # 创建 SubsetRandomSampler
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        # 创建 DataLoader
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize,
            sampler=sampler,  # 使用自定义采样器
            num_workers=args.workers,
            pin_memory=True
        )
    else: 
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
    
    return val_dataset, val_loader, classnames