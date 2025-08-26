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
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, ece_calculator, ECE_Loss, select_confident_samples, avg_entropy
from ttabc.utils.model import create_model, set_optimizer
from ttabc.utils.load_data import create_dataloader
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import ipdb
import math
import pickle

class BaseMethod:
    def __init__(self, args):
        self.args = args
        self.temperature_value = {'ViT': 1.16, 'RN': 1.15} #for temperature scaling experiments 
        # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
        self.model, self.model_state = create_model(args)
        # define optimizer
        self.optimizer, self.optim_state = set_optimizer(args, self.model)
        # setup automatic mixed-precision (Amp) loss scaling
        self.scaler = torch.amp.GradScaler(init_scale=1000, device='cuda')
        print('=> Using native Torch AMP. Training in mixed precision.')

    def test_time_tuning(self, inputs):
        pass

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        pass