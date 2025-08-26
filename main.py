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
from ttabc.model_selection import get_method
from ttabc.utils.configs_tool import config_hparams
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import ipdb
import math
import pickle
from datetime import datetime
import os

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(args, file_path=None):

    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    
    tpt_methods = get_method(args.run_type)(args)

    cudnn.benchmark = True
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        val_dataset, val_loader, classnames = create_dataloader(args, set_id)
        if args.cocoop:
            tpt_methods.model.prompt_generator.reset_classnames(classnames, args.arch)
            tpt_methods.model = tpt_methods.model.cpu()
            tpt_methods.model_state = tpt_methods.model.state_dict()
            tpt_methods.model = tpt_methods.model.cuda(args.gpu)
        else:
            tpt_methods.model.reset_classnames(classnames, args.arch)

        if args.is_ece:
            result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
        else:
            result_dict = None
        calibration_methods_name = ['ctpt', 'otpt', 'ntpt']
        if args.run_type in calibration_methods_name:
            assert len(datasets) == 1
            args.is_ece = True
            result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
            results[set_id] = tpt_methods.test_time_adapt_eval(val_loader, result_dict)
            acc, ece = ece_calculator(result_dict)
        else:
            if args.is_ece:
                results[set_id] = tpt_methods.test_time_adapt_eval(val_loader, result_dict)
                acc, ece = ece_calculator(result_dict)
            else:
                results[set_id] = tpt_methods.test_time_adapt_eval(val_loader)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("[set_id] \t Top-1 acc. \t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end=" \t\t")
    # print("\n")
    # for id in results.keys():
        print("{:.2f}".format(results[id][0]), end=" \t\t")
        print("{:.2f}".format(results[id][1]), end=" \t\t")
        print("\n")
    if args.is_ece:
        print('acc: ', acc)
        print('ece: ', ece)
    print("============== End =============")
    if args.log_dir is not None:
        # Create a folder named with the current date
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_path = os.path.join(args.log_dir, date_str)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        with open(file_path, 'a') as f:
            f.write("\n======== Result Summary ========\n")
            f.write("params: nstep\tlr\tbs\n")
            f.write(f"params: {args.tta_steps}\t{args.lr}\t{args.batch_size}\n")
            f.write("[set_id] \t Top-1 acc. \t Top-5 acc.\n")
            for id in results.keys():
                f.write(f"{id} \t\t")
            f.write("\n")
            for id in results.keys():
                f.write(f"{results[id][0]:.2f} \t\t")
                f.write(f"{results[id][1]:.2f} \t\t")
                f.write("\n")
            if args.is_ece:
                f.write(f'acc: {acc}\n')
                f.write(f'ece: {ece}\n')
            f.write("============== End =============\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    # added args for c-tpt --------------------------------
    parser.add_argument('--lambda_term' , type=float, default=0.0, help='lambda for c-tpt')
    parser.add_argument('--run_type' , type=str, default='baseline_tpt', choices=['baseline', 'tpt', 'ctpt', 'tpt_ts', 'otpt', 'ntpt'],)
    parser.add_argument('--two_step', action='store_true', default=False, help='two step training')
    parser.add_argument('--I_augmix', action='store_true', default=False, help='augmix for I')
    # ------------------------------------------------


    # added args for n-tpt --------------------------------
    parser.add_argument('--down_sample_ratio' , type=float, default=None, help='down sample ratio for dataset')
    parser.add_argument('--log_dir', type=str, default=None, help='log directory')
    parser.add_argument('--is_ece', type=str, default=False, help='whether to calculate ece')

    # ------------------------------------------------

    args = parser.parse_args()
    args = config_hparams(args)

    if args.log_dir is not None:
        # Create a folder named with the current date
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_path = os.path.join(args.log_dir, date_str)
        os.makedirs(log_path, exist_ok=True)
        
        # Create a file to store args and printed parameters
        file_name = f"{args.run_type}_{args.test_sets}_{args.arch}.txt"
        file_path = os.path.join(log_path, file_name)
        
        with open(file_path, 'a') as f:
            # Write args to the file
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        main(args, file_path)
    else:
        main(args)






