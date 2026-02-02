import argparse

from PIL import Image

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import torchvision.models as models

from ttabc.utils.tools import set_random_seed, reset_classnames
from ttabc.utils.metrics_tools import ece_calculator, accuracy_writer
from ttabc.utils.load_data import create_dataloader
from ttabc.model_selection import get_method
from ttabc.utils.configs_tool import config_hparams
from data.cls_to_names import *

from datetime import datetime
import os

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(args):

    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))
    # print the run_type and prompt_type
    print(f"TTA Method: {args.run_type}, Prompting type: {args.prompt_type}")
    
    tta_methods = get_method(args.run_type)(args)

    cudnn.benchmark = True
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    results_for_ece = {}
    ece_res = {}
    for set_id in datasets:
        if args.log_dir is not None:
            # Create a folder named with the current date
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_path = os.path.join(args.log_dir, date_str)
            os.makedirs(log_path, exist_ok=True)
            
            # Create a file to store args and printed parameters
            file_name = f"{args.run_type}_{set_id}_{args.arch.replace('/', '-')}.txt"
            file_path = os.path.join(log_path, file_name)

            with open(file_path, 'a') as f:
                # Write args to the file
                for arg, value in vars(args).items():
                    f.write(f"{arg}: {value}\n")

        val_dataset, val_loader, classnames = create_dataloader(args, set_id)
        
        reset_classnames(args, tta_methods, classnames)

        results_for_ece[set_id] = {'max_confidence': [], 'prediction': [], 'label': []}
        results[set_id] = tta_methods.test_time_adapt_eval(val_loader, results_for_ece[set_id])
        _, ece_res[set_id] = ece_calculator(results_for_ece[set_id])
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
    if args.log_dir is not None:
        accuracy_writer(args, results, ece_res, log_path=log_path, file_path=file_path)
    else:
        accuracy_writer(args, results, ece_res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    # data
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I/imagenetc', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('--down_sample_ratio' , type=float, default=None, help='down sample ratio for dataset')
    parser.add_argument('--I_augmix', action='store_true', default=False, help='augmix for I')
    # specified for corrupted datasets (imagenetc, cifar10c, cifar100c, etc)
    parser.add_argument('--corruption_type', default='frost', type=str, help='corruption type for imagenetc, cifar10c, cifar100c')
    parser.add_argument('--corruption_level', default=5, type=int, help='corruption level for imagenetc, cifar10c, cifar100c')
    
    # model
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50') # backbone architecture
    # TTA method and prompting
    parser.add_argument('--prompt_type' , type=str, default='coop', choices=['no_prompt', 'coop', 'cocoop', 'vp', 'maple', 'norm', 'prototype'], help='type of prompt, support no_prompt/coop/cocoop/vp/maple/norm/prototype')
    parser.add_argument('--run_type' , type=str, default='tpt', choices=['zero_shot', 'baseline', 'tpt', 'ctpt', 'otpt', 'tda', 'boostadapter', 'difftpt', 'zero', 'batclip', 'dmn', 'histpt', 'oga', 'dota', 'bca', 'tent', 'sar', 'deyo', 'promptalign', 'tps', 'dpe'], help='which TTA method to use')
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning (sample-wise with augmentations)')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--episodic', action='store_true', default=True, help='whether to reset model after each sample/batch')
    # prompting config
    parser.add_argument('--vp_type' , type=str, default='lor_vp', choices=['pad_vp', 'resized_pad_vp', 'patch_vp', 'random_patch_vp', 'lor_vp'], help='type of visual prompt')
    parser.add_argument('--vp_mode' , type=str, default='zs', choices=['zs', 'coop', 'cocoop'], help='multimodal prompt for visual prompt')
    # parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization") # having been removed
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--prompt_setting', default=None, type=str)
    
    # parameters and device
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    # logging
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--log_dir', type=str, default=None, help='log directory')


    args = parser.parse_args()
    # Note: some critical hyperparameters will be replaced by the defaults in configs/algorithms.py
    args = config_hparams(args)

    main(args)






