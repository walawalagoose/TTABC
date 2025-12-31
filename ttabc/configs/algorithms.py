# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "zero_shot": {
        "tpt": False,
        "prompt_type": 'no_prompt',
        "load": None,
    },
    "baseline": {
        "tpt": False,
        # "prompt_type": 'coop', # actually it can be epuipped with any prompt type
        "supported_list": ['coop', 'cocoop', 'vp', 'maple'],
        # "load": None,
    },
    "tpt": {
        "tpt": True,
        "prompt_type": 'coop', # can also be cocoop
        "supported_list": ['coop', 'cocoop', 'vp'],
        "vp_type": 'lor_vp',
        "vp_mode": 'coop', # 'zs' or 'coop', 'cocoop' not supported yet
        # "load": 'checkpoints/coop/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner',
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
    
    "tda": {
        # tpt=true, bs=aug_size or tpt=false, bs=1
        "tpt": True,
        "batch_size": 64,
        
        "prompt_type": 'no_prompt',
        "supported_list": ['no_prompt'],
        "load": None,
        "selection_p": 0.1,  # confidence threshold for online shot.
        
        # config for the caches
        "pos_config": {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 2.0,
            'beta': 5.0
        },
        "neg_config": {
            'enabled': True,
            'shot_capacity': 2,
            'alpha': 0.117,
            'beta': 1.0,
            'entropy_threshold': {'lower': 0.2, 'upper': 0.5},
            'mask_threshold': {'lower': 0.03, 'upper': 1.0}
        }
    },
    
    "boostadapter": {
        # tpt=true, bs=aug_size or tpt=false, bs=1
        "tpt": True,
        "batch_size": 64,
        
        "prompt_type": 'no_prompt',
        "supported_list": ['no_prompt'],
        "load": None,
        "selection_p": 0.1,  # confidence threshold for online shot.
        
        "infer_ori_image": True,
        "delta": 0,
        
        # config for the caches
        "pos_config": {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 2.0,
            'beta': 5.0
        },  
        "neg_config": {
            'enabled': True,
            'shot_capacity': 2,
            'alpha': 0.117,
            'beta': 1.0,
            'entropy_threshold': {'lower': 0.2, 'upper': 0.5},
            'mask_threshold': {'lower': 0.03, 'upper': 1.0}
        }
    },
    
    "difftpt": {
        "tpt": True,
        "prompt_type": 'coop', # can also be cocoop
        "supported_list": ['coop', 'cocoop'],
        # "load": 'checkpoints/coop/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner',
        # "load": None,
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 32,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
        "tta_steps": 1,  # number of test-time adaptation steps
        "lr": 5e-3,  # learning rate
        "n_ctx": 4,  # number of prompt tokens
        
        "selection_cosine": 0.8, # selection ratio based on cosine similarity
        "selection_selfentro": 0.3, # selection ratio based on self-entropy
        "diff_aug_size": 32, # kept the same as batch size
        "diff_guidance_scale": 3.0, # SD guidance scale
        "diff_times": 10, # diffusion steps
    },

    "zero": {
        "tpt": True,
        "prompt_type": 'no_prompt',
        "supported_list": ['no_prompt'],
        "batch_size": 64, # quals to aug_size
        "selection_p": 0.3,
    },

    "promptalign": {
        "tpt": True,
        "prompt_type": 'maple', # PromptAlign uses MaPLe architecture
        "supported_list": ['maple'],
        "load": None,
        "optimizer": "AdamW",  # Changed from SGD - PromptAlign uses AdamW for TTA
        "batch_size": 64,
        "ctx_init": "a photo of a",
        "tta_steps": 1,
        "lr": 4e-2, # TPT.LR from PromptAlign config (was 5e-4, should be 4e-2)
        "n_ctx": 2, # TRAINER.PROMPTALIGN.N_CTX (was 4, should be 2 for DG)
        "prompt_depth": 3, # TRAINER.PROMPTALIGN.PROMPT_DEPTH (was 9, should be 3 for DG)
        "tpt_threshold": 0.1, # TPT.TPT_THRESHOLD
        "align_threshold": 0.1, # TPT.ALIGN_THRESHOLD
        "align_layer_from": 0, # TPT.ALIGN_LAYER_FROM
        "align_layer_to": 3, # TPT.ALIGN_LAYER_TO (was 12, should be 3 for DG)
        "distr_loss_w": 100.0, # TPT.DISTR_LOSS_W
        "tpt_loss": True, # TPT.TPT_LOSS (usually True)
        "distr_align": True, # TPT.DISTR_ALIGN (usually True)
        "vis_means_path": "./outputs/features/Ipre_vis_means.pt",
        "vis_vars_path": "./outputs/features/Ipre_vis_vars.pt",
    },
}