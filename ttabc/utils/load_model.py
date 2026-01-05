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

from clip.custom_clip import get_coop, get_zero_shot, get_tta_norm
from clip.cocoop import get_cocoop
from clip.maple import get_maple
from clip.visual_prompting import get_visual_prompt_clip
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
    # create model (zero-shot clip model (ViT-L/14@px336) with prompt tuning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
        
    assert args.prompt_type in args.supported_list, \
        f"Prompting type {args.prompt_type} not supported for {args.run_type}"
        
    if args.prompt_type in ['no_prompt', None]:
        model = get_zero_shot(args.arch, args.test_sets, args.gpu)
        model_state = deepcopy(model.state_dict())
        
    elif args.prompt_type in ['norm']:
        model = get_tta_norm(args.arch, args.test_sets, args.gpu)
        model_state = deepcopy(model.state_dict())
        
    elif args.prompt_type == 'coop':
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load, weights_only=False)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx
                # model.backbone.prompt_learner.ctx.copy_(pretrained_ctx)
                # model.backbone.prompt_learner.ctx_init_state = pretrained_ctx
        model_state = None # beacuse coop already has ctx_init_state
        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            
    elif args.prompt_type == 'cocoop':
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
        for name, param in model.named_parameters():
            if "text_encoder" not in name:
                param.requires_grad_(False)
        
    elif args.prompt_type == 'vp': # visual prompt
        assert args.vp_mode in ['zs', None, 'coop'], "Not implemented: visual prompt with cocoop" # TODO, future work: vp with cocoop
        assert args.vp_type in ['pad_vp', 'resized_pad_vp', 'patch_vp', 'random_patch_vp', 'lor_vp'], "Visual prompt type not supported"
        vp_args = {'prompt_size': 30, 'image_size': args.resolution, 'rank': 4}
        coop_args = {'n_ctx': args.n_ctx, 'ctx_init': args.ctx_init, 'learned_cls': False}
        model = get_visual_prompt_clip(args.arch, args.test_sets, args.gpu, args.vp_type, args.vp_mode, vp_args=vp_args, coop_args=coop_args)
        model_state = deepcopy(model.state_dict())
        # require_grad_(False) for all except visual prompter
        for name, param in model.named_parameters():
            if "visual_prompter" not in name:
                param.requires_grad_(False)
        # when vp works with coop
        if args.vp_mode == 'coop':
            for p in model.backbone.prompt_learner.parameters():
                p.requires_grad_(True)
            if args.load is not None:
                print("Use pre-trained soft prompt (CoOp) as initialization")
                pretrained_ctx = torch.load(args.load, weights_only=False)['state_dict']['ctx']
                assert pretrained_ctx.size()[0] == args.n_ctx
                with torch.no_grad():
                    model.prompt_learner.ctx_init_state = pretrained_ctx
        # TODO: customize when vp works with cocoop
        elif args.vp_mode == 'cocoop':
            pass
        
    elif args.prompt_type == 'maple':
        # TODO: put this design_details setting in algorithm.py
        design_details = {
            'trainer': 'MaPLe',
            'vision_depth': getattr(args, 'vision_depth', 0),
            'language_depth': getattr(args, 'language_depth', 0), 
            'vision_ctx': getattr(args, 'vision_ctx', 0),
            'language_ctx': getattr(args, 'language_ctx', 0),
            'maple_length': getattr(args, 'maple_length', 4),  # MaPLe prompt length
            'prompt_depth': getattr(args, 'prompt_depth', 9),  # MaPLe prompt depth
            }
        if args.ctx_init is not None:
            assert len(args.ctx_init.split('_')) == design_details['maple_length'], "n_ctx should be equal to maple_length"
        else:
            assert args.n_ctx == design_details['maple_length'], "n_ctx should be equal to maple_length"
        model = get_maple(
            args.arch, args.test_sets, args.gpu,
            n_ctx=args.n_ctx,ctx_init=args.ctx_init,
            prompt_depth=design_details['prompt_depth'],
            design_details=design_details
        )
        model_state = model.state_dict()
       
    print(f"=> Model created: visual backbone {args.arch}, prompting type: {args.prompt_type}")
        
    if not torch.cuda.is_available():
        model = model.to(torch.device('cpu'))
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    return model, model_state

def set_optimizer(args, model):
    
    def _build_optimizer(args, params):
        opt_name = getattr(args, "optimizer", "Adam").lower()
        lr = getattr(args, "lr", 1e-4)
        wd = getattr(args, "weight_decay", 1e-4)
        if opt_name == "sgd":
            momentum = getattr(args, "momentum", 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        elif opt_name == "adam":
            betas = getattr(args, "betas", (0.9, 0.999))
            return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
        else:  # adamw
            betas = getattr(args, "betas", (0.9, 0.999))
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
        
    if args.prompt_type in ['no_prompt', None]:
        optimizer = None
        optim_state = None
    
    # Since certain methods requires special parameter optimization, details will be implemented in the method file.
    elif args.prompt_type in ['norm']:
        optimizer = None
        optim_state = None
        
    elif args.prompt_type == 'coop':
        trainable_param = model.prompt_learner.parameters()
        # optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optimizer = _build_optimizer(args, trainable_param)
        optim_state = deepcopy(optimizer.state_dict())
        
    elif args.prompt_type == 'cocoop': # set dynamiclly during adaptation
        # TODO, set a function for cocoop to set optimizer during adaptation
        optimizer = None
        optim_state = None      
        
    elif args.prompt_type == 'vp':
        trainable_param = model.get_trainable_parameters()
        optimizer = _build_optimizer(args, trainable_param)
        optim_state = deepcopy(optimizer.state_dict())
        
    elif args.prompt_type == 'maple':
        trainable_param = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = _build_optimizer(args, trainable_param)
        optim_state = optimizer.state_dict()
    
    return optimizer, optim_state
