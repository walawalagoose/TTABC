# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "zero_shot": {
        "tpt": False,
        "prompt_type": 'zs',
        "load": None,
    },
    "baseline": {
        "tpt": False,
        # "prompt_type": 'coop', # actually it can be epuipped with any prompt type
        "supported_list": ['zs', 'coop', 'cocoop', 'vp', 'maple'],
        # "load": None,
    },
    "tpt": {
        "tpt": True,
        "prompt_type": 'coop', # can also be cocoop
        "supported_list": ['coop', 'cocoop', 'vp'],
        # "vp_type": 'lor_vp',
        # "load": 'checkpoints/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner',
        # "load": None,
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 64,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
        "selection_p": 0.1,  # confidence threshold for online shot.
        "tta_steps": 1,  # number of test-time adaptation steps
        "lr": 5e-3,  # learning rate
        "n_ctx": 4,  # number of prompt tokens
    },
    "ctpt": {
        "tpt": True,
        "prompt_type": 'coop',
        "supported_list": ['coop', 'cocoop', 'vp'],
        # "vp_type": 'lor_vp',
        "load": None,
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 64,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
        "selection_p": 0.1,  # confidence threshold for online shot.
        "tta_steps": 1,  # number of test-time adaptation steps
        "lr": 5e-3,  # learning rate
        "n_ctx": 4,  # number of prompt tokens
        "lambda_term": 2.0,  # lambda for c-tpt
        "two_step": False,  # whether to use two-step training
    },
    "otpt": {
        "tpt": True,
        "prompt_type": 'coop',
        "supported_list": ['coop', 'cocoop', 'vp'],
        # "vp_type": 'lor_vp',
        "load": None,
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 64,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
        "selection_p": 0.1,  # confidence threshold for online shot.
        "tta_steps": 1,  # number of test-time adaptation steps
        "lr": 5e-3,  # learning rate
        "n_ctx": 4,  # number of prompt tokens
        "lambda_term": 2.0,  # lambda for o-tpt
        "two_step": False,  # whether to use two-step training
    },
    "ntpt": {
        "tpt": True,
        "prompt_type": 'coop',
        "supported_list": ['coop', 'cocoop', 'vp'],
        # "vp_type": 'lor_vp',
        "load": None,
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 64,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
        "selection_p": 0.1,  # confidence threshold for online shot.
        "tta_steps": 1,  # number of test-time adaptation steps
        "lr": 5e-3,  # learning rate
        "n_ctx": 4,  # number of prompt tokens
        "lambda_term": 10.0,  # lambda for n-tpt
        "two_step": False,  # whether to use two-step training
    },

}