import argparse
import json
import os
import sys
from datetime import datetime
from loguru import logger

from PIL import Image

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

from ttabc.utils.tools import set_random_seed, reset_classnames
from ttabc.utils.metrics_tools import ece_calculator
from ttabc.utils.load_data import create_dataloader
from ttabc.model_selection import get_method
from ttabc.utils.configs_tool import config_hparams
from data.cls_to_names import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def _get_dataset_dir_name(args, set_id):
    if set_id.lower() in {"imagenetc", "cifar10c", "cifar100c"}:
        return f"{set_id}_{args.corruption_type}_{args.corruption_level}"
    return set_id


def _setup_logger(log_file_path):
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file_path, level="INFO", enqueue=True)


def main(args):

    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    set_random_seed(args.seed)
    tta_methods = get_method(args.run_type)(args)

    cudnn.benchmark = True
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    results_for_ece = {}
    ece_res = {}
    for set_id in datasets:
        arch_str = args.arch.replace('/', '-').lower()
        dataset_str = _get_dataset_dir_name(args, set_id)
        base_dir = os.path.join(args.log_dir, arch_str, dataset_str)
        date_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir_name = f"{args.run_type}_{args.prompt_type}_{date_dir}"
        run_dir = os.path.join(base_dir, run_dir_name)
        os.makedirs(run_dir, exist_ok=True)

        json_path = os.path.join(run_dir, f"{run_dir_name}.json")
        log_path = os.path.join(run_dir, f"{run_dir_name}.log")
        _setup_logger(log_path)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

        logger.info("Use GPU: {} for training", args.gpu)
        logger.info("Architecture: {}", args.arch)
        logger.info("TTA Method: {}, Prompting type: {}", args.run_type, args.prompt_type)
        logger.info("Evaluating on dataset: {}", dataset_str)
        
        val_dataset, val_loader, classnames = create_dataloader(args, set_id)
        
        reset_classnames(args, tta_methods, classnames)

        results_for_ece[set_id] = {'max_confidence': [], 'prediction': [], 'label': []}
        results[set_id] = tta_methods.test_time_adapt_eval(val_loader, results_for_ece[set_id])
        _, ece_res[set_id] = ece_calculator(results_for_ece[set_id])
        del val_dataset, val_loader
        # try:
        #     logger.info("=> Acc. on testset [{}]: @1 {:.2f}/ @5 {:.2f}", set_id, results[set_id][0], results[set_id][1])
        # except Exception:
        #     logger.info("=> Acc. on testset [{}]: {}", set_id, results[set_id])

        logger.info("============= Result Summary =============")
        logger.info("Params:         nstep\tlr\tbs")
        logger.info("Params:         {}\t{}\t{}", args.tta_steps, args.lr, args.batch_size)
        logger.info("Dataset:        {}", set_id)
        logger.info("Top-1 Acc.:     {:.2f}", results[set_id][0])
        logger.info("Top-5 Acc.:     {:.2f}", results[set_id][1])
        logger.info("ECE:            {:.2f}", ece_res[set_id])
        # logger.info("{} \t {:.2f} \t {:.2f} \t {:.2f}", set_id, results[set_id][0], results[set_id][1], ece_res[set_id])
        logger.info("=================== End ==================")

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
    parser.add_argument('--run_type' , type=str, default='tpt', choices=['zero_shot', 'baseline', 'tpt', 'ctpt', 'otpt', 'tda', 'boostadapter', 'difftpt', 'zero', 'batclip', 'dmn', 'histpt', 'oga', 'dota', 'bca', 'tent', 'sar', 'deyo', 'promptalign', 'tps', 'dpe', 'rlcf', 'calip', 'atpt', 'panda', 'mta', 'rtpt', 'onzeta', 'ecalp', 'dynaprompt', 'stata', 'moon'], help='which TTA method to use')
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
    
    # StatA config
    parser.add_argument('--stata_alpha', type=float, default=1.0, help='StatA anchor weighting hyper-parameter')
    parser.add_argument('--stata_lambda_y_hat', type=float, default=1.0, help='StatA zero-shot prediction weighting')
    parser.add_argument('--stata_lambda_laplacian', type=float, default=1.0, help='StatA Laplacian weighting')
    parser.add_argument('--stata_n_neighbors', type=int, default=3, help='number of neighbors for StatA affinity graph')
    parser.add_argument('--stata_max_iter', type=int, default=10, help='number of StatA outer iterations')
    parser.add_argument('--stata_soft_beta', action='store_true', default=False, help='use soft StatA beta computation')
    # Label shift config
    parser.add_argument('--label_shift', action='store_true', default=False,
                        help='enable label-shifted sampling in test dataloader')
    parser.add_argument('--label_shift_mode', type=str, default='batch',
                        choices=['batch', 'online'],
                        help='label shift mode: original batch sampler or original online sampler')
    parser.add_argument('--label_shift_gamma', type=float, default=1.0,
                        help='Dirichlet parameter for original online label shift sampler')
    parser.add_argument('--num_class_eff', type=int, default=None,
                        help='original batch sampler: number of effective classes in each batch')
    parser.add_argument('--num_class_eff_min', type=int, default=None,
                        help='original batch sampler: minimum number of effective classes per batch')
    parser.add_argument('--num_class_eff_max', type=int, default=None,
                        help='original batch sampler: maximum number of effective classes per batch')
    parser.add_argument('--label_shift_num_slots', type=int, default=None,
                        help='original online sampler: number of slots; default is min(num_batches, num_classes)')
    parser.add_argument('--label_shift_scope', type=int, default=-1,
                        help='scope size for original batch/online label shift; -1 means global scope')
    # MOON config
    parser.add_argument('--moon_alpha', type=float, default=1.0, help='MOON anchor weighting')
    parser.add_argument('--moon_lambda_y_hat', type=float, default=1.0, help='MOON zero-shot prediction weighting')
    parser.add_argument('--moon_lambda_laplacian', type=float, default=1.0, help='MOON Laplacian weighting')
    parser.add_argument('--moon_n_neighbors', type=int, default=3, help='number of MOON graph neighbors')
    parser.add_argument('--moon_max_iter', type=int, default=10, help='number of MOON outer iterations')
    parser.add_argument('--moon_soft_beta', action='store_true', default=False, help='use soft MOON class counts')

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






