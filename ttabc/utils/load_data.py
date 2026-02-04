from PIL import Image
import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.imagenet_prompts import imagenet_classes
from data.cifar_prompts import cifar10_classes, cifar100_classes
from data.datautils import AugMixAugmenter, build_dataset
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import ipdb
import math
import pickle
from datetime import datetime
import os
from loguru import logger

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
        real_batch_size = 1
    else:
        data_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        real_batch_size = args.batch_size

    # reset the model
    # Reset classnames of custom CLIP model
    if len(set_id) > 1 and 'imagenet' not in set_id and 'cifar' not in set_id: 
        # fine-grained classification datasets
        classnames = eval("{}_classes".format(set_id.lower()))
    elif 'cifar' in set_id:
        if 'cifar100' in set_id:
            classnames = cifar100_classes
        elif 'cifar10' in set_id:
            classnames = cifar10_classes
        else:
            raise NotImplementedError
    else:
        assert set_id in ['A', 'R', 'K', 'V', 'I'] or 'imagenetc' in set_id 
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
    if 'imagenetc' in set_id or 'cifar10c' in set_id or 'cifar100c' in set_id:
        if '-' in set_id:
            # e.g., cifar10c-brightness-5
            parts = set_id.split('-')
            assert len(parts) == 3 and parts[0] in ['cifar10c', 'cifar100c', 'imagenetc']
            val_dataset = build_dataset(parts[0], data_transform, args.data, mode=args.dataset_mode, corruption_type=parts[1], corruption_level=int(parts[2]))
        else:
            val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode, corruption_type=args.corruption_type, corruption_level=args.corruption_level)
    else:
        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
    
    if args.down_sample_ratio is not None:
        # Get dataset size and random sampling indices
        total_size = len(val_dataset)
        sample_size = int(args.down_sample_ratio * total_size)
        indices = np.random.choice(total_size, sample_size, replace=False)
        logger.info("Number of test samples: {}".format(sample_size))
        # Create SubsetRandomSampler
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        # Create DataLoader
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=real_batch_size,
            sampler=sampler,  # Use custom sampler
            num_workers=args.workers,
            pin_memory=True
        )
    else: 
        logger.info("Number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=real_batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
    
    return val_dataset, val_loader, classnames