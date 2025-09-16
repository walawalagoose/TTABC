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
from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, load_model_weight
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import ipdb
import math
import pickle
from datetime import datetime
import os


def create_model(args):
    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if not args.tpt:
        # TODO: add zero-shot
        # model = get_zero_shot(args.arch, args.test_sets, args.gpu)
        # model_state = deepcopy(model.state_dict())
        pass
    elif args.prompt_type == 'T':
        if args.cocoop:
            model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
            assert args.load is not None
            load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
            model_state = deepcopy(model.state_dict())
        else:
            model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
            if args.load is not None:
                print("Use pre-trained soft prompt (CoOp) as initialization")
                pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
                assert pretrained_ctx.size()[0] == args.n_ctx
                with torch.no_grad():
                    #model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                    #model.prompt_learner[0].ctx_init_state = pretrained_ctx

                    model.prompt_learner.ctx.copy_(pretrained_ctx)
                    model.prompt_learner.ctx_init_state = pretrained_ctx

            model_state = None

        for name, param in model.named_parameters():
            if not args.cocoop:
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            else:
                if "text_encoder" not in name:
                    param.requires_grad_(False)
        
        print("=> Model created: visual backbone {}".format(args.arch))
    elif args.prompt_type == 'V': # visual prompt
        assert args.load is None and args.cocoop is False
        # TODO: add visual prompt model
        # model = get_visual_prompt_clip(args.arch, args.test_sets, args.gpu, args.vp_type)
        # model_state = deepcopy(model.state_dict())
        # require_grad_(False) for all except visual prompt generator
        pass
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    return model, model_state

def set_optimizer(args, model):
    if args.tpt is False:
        optimizer = None
        optim_state = None
    else:
        if args.prompt_type == 'T':
            if args.cocoop:
                optimizer = None
                optim_state = None
            else:
                trainable_param = model.prompt_learner.parameters()
                optimizer = torch.optim.AdamW(trainable_param, args.lr)
                optim_state = deepcopy(optimizer.state_dict())
        elif args.prompt_type == 'V':
            # TODO: add visual prompt optimizer
            # optimizer = None
            # optim_state = None
            pass
    
    return optimizer, optim_state
