# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "zero_shot": {
        "tpt": False, # True is also supported
        "prompt_type": 'no_prompt',
        "supported_list": ['no_prompt'],
        "load": None,
    },
    "baseline": {
        "tpt": True,
        "prompt_type": 'coop',
        "supported_list": ['coop', 'cocoop', 'vp', 'maple'],
        "load": None,
        # "load": 'checkpoints/coop/16shots/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50',
        # "load": 'checkpoints/cocoop/16shots/rn50_c4_ep10_batch1/seed1/prompt_learner/model.pth.tar-10',
    },
    "tpt": {
        "tpt": True,
        "prompt_type": 'coop', # can also be cocoop
        "supported_list": ['coop', 'cocoop', 'vp'],
        "vp_type": 'lor_vp',
        "vp_mode": 'coop', # 'zs' or 'coop', 'cocoop' not supported yet
        "load": None,
        # "load": 'checkpoints/coop/16shots/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50',
        # "load": 'checkpoints/cocoop/16shots/rn50_c4_ep10_batch1/seed1/prompt_learner/model.pth.tar-10',
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 64,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
        "selection_p": 0.1,  # confidence threshold for online shot.
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
        "batch_size": 64, # equals to aug_size
        "selection_p": 0.3,
    },
    
    "batclip": {
        # tpt=true, bs=aug_size or tpt=false, bs=batch_size
        "tpt": False,
        "prompt_type": 'norm',
        "supported_list": ['norm'],
        "optimizer": "AdamW",
        "batch_size": 64,
        "episodic": False,
    },
    
    "histpt": {
        # tpt=true/false are both supported, false is original HisTPT setting
        "tpt": False,
        "prompt_type": 'coop',
        "supported_list": ['coop'],
        "load": None,
        "selection_p": 0.1,
        "optimizer": "SGD",
        "batch_size": 1,
        "n_ctx": 4,
        "ctx_init": 'a_photo_of_a',
        
        "memory_size": 32,  # Local knowledge bank size L=H=32
        "hard_topk": 16,  # Number of hard-sample features K=16
        "ema_momentum": 0.99,  # Update coefficient γ=0.99 for global knowledge bank
        "episodic": False,
    },
    
    "dmn": {
        "tpt": True,
        "prompt_type": 'no_prompt',
        "supported_list": ['no_prompt'],
        "batch_size": 64,
        "selection_p": 0.1,
        
        "memory_size": 50,
        "beta": 5.5,
        "text_weight": 1.0, # beta2 in the paper
        "mem_weight": 0.1, # beta3 in the paper
        "shared_param": False,
        "mapping": "bias", # 'bias', 'affine', 'all'
        "position": "all", # 'query', 'key', 'value', 'qkv', 'output', 'all'
    },
    
    "oga": {
        "tpt": False,
        "prompt_type": "no_prompt",
        "supported_list": ["no_prompt"],
        "batch_size": 1,
        "shot_capacity": 8,
        "tau": 0.05,
        "sig_type": "RidgeMoorePenrose",
        "normalize_mu": False,
    },
    
    "bca": {
        "tpt": True,
        "prompt_type": "no_prompt",
        "supported_list": ["no_prompt"],
        "batch_size": 64, 
        "selection_p": 0.1,

        "threshold1": 0.05,
        "init_count1": 20000,
        "threshold2": 0.65,
        "init_count2": 1,
    },
    
    "dota": {
        "tpt": True, 
        "prompt_type": "no_prompt",
        "supported_list": ["no_prompt"],
        "batch_size": 64,  
        "selection_p": 0.1,
        
        "epsilon": 1e-4,
        "sigma": 2e-3,
        "eta": 0.3,
        "rho": 0.02,
        "init_mu": "clip_text",  # or "constant"
        "init_mu_value": 1e-3,
    },

    "tent": {
        "tpt": False,
        "prompt_type": 'norm',
        "supported_list": ['norm'],
        "optimizer": "AdamW",
        "batch_size": 64,
        "episodic": False,
    },
    "sar": {
        "tpt": False,
        "prompt_type": 'norm',
        "supported_list": ['norm'],
        "batch_size": 64,
        "episodic": False,
        
        "sar_margin_e0": 0.40,  # remember to multiply math.log(num_classes)
        "reset_constant_em": 0.2,  # threshold e_m for model recovery scheme
    },
    "deyo": {
        "tpt": False,
        "prompt_type": 'norm',
        "supported_list": ['norm'],
        "optimizer": "AdamW",
        "batch_size": 64,
        "episodic": False,
        
        "deyo_margin": 0.50, # remember to multiply math.log(num_classes)
        "aug_type": "patch", # the augmentation type for prime
        "occlusion_size": 112, # choises for occ
        "row_start": 56, # choises for occ
        "column_start": 56, # choises for occ
        "patch_len": 4, # choises for patch
        "plpd_threshold": 0.3, # plpd threshold for DeYO
        "reweight_ent": 1, # reweight entropy loss
        "reweight_plpd": 1, # reweight plpd loss
        "margin": 0.40, # remember to multiply math.log(num_classes)
    },
    
    "tps": {        
        "tpt": True,
        "prompt_type": 'prototype',
        "supported_list": ['prototype'],
        "optimizer": "SGD",
        "batch_size": 64, 
        "selection_p": 0.1, 
        
        "per_label": True,
        "prototype_type": "gpt4_x_templates",  # ['basic', 'templates', 'gpt4', 'gpt4_x_templates']
        "prototype_dir": ".caches/prototypes", 
        # Note: Before running TPS, generate prototypes using:
        # python3 generate_gpt_concepts.py --arch ViT-B/16 --datasets I/A/R/V --concepts_json /path/to/gpt4.json
    },

    "promptalign": {
        "tpt": True,
        "prompt_type": "maple",
        "supported_list": ["maple"],
        "load": None,
        "optimizer": "SGD",
        "batch_size": 64,
        # "ctx_init": "a_photo_of_a",
        "n_ctx": 2,
        "prompt_depth": 3,
        
        "tpt_loss": True,
        "distr_align": True,
        "tpt_threshold": 0.1,
        "align_threshold": 0.1,
        "distr_loss_w": 100.0,
        "align_layer_from": 0,
        "align_layer_to": 3,
        "vis_means_path": 'checkpoints/promptalign/ImgNet_vis_means.pt',
        "vis_vars_path": 'checkpoints/promptalign/ImgNet_vis_vars.pt',
    },
    
    "dpe": {
        "tpt": True,
        "prompt_type": 'prototype',
        "supported_list": ['prototype'],
        "optimizer": "SGD",
        "batch_size": 64,
        "selection_p": 0.1,
        
        "pos_config": {
            "enabled": True,
            "shot_capacity": 3,  # Maximum samples per class in cache
            "alpha": 6.0,  # Cache weight for logits computation
            "beta": 5.0,  # Temperature parameter for cache similarity
        },
        "lr_config": {
            "text": 5e-3,  # Learning rate for text prototype residuals
            "image": 5e-3,  # Learning rate for image prototype residuals
            "align": 0.5,  # Weight for InfoNCE alignment loss
        },
        "global_update_threshold": 0.1, 
      },
    
    "rlcf": {
        "tpt": True,
        "prompt_type": 'coop', # can also be cocoop
        "supported_list": ['coop', 'cocoop', 'maple'],
        "load": None,
        # "load": 'checkpoints/coop/16shots/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50',
        # "load": 'checkpoints/cocoop/16shots/rn50_c4_ep10_batch1/seed1/prompt_learner/model.pth.tar-10',
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 64,
        "ctx_init": 'a_photo_of_a',
        "selection_p": 0.1,
        "n_ctx": 4,
        
        
        # RLCF-specific parameters
        "sample_k": 3,  # Number of classes to sample from top-K
        "reward_arch": "ViT-L/14",  # Reward model architecture (can be different from main model)
        "reward_process": True,  # Enable reward post-processing (centering)
        "reward_amplify": False,  # Standardize rewards (divide by std, default: False)
        "process_batch": False,  # Process rewards per sample (not across batch)
        "multiple_reward_models": False,  # Use ensemble of multiple CLIP models as rewards
        "weighted_scores": True,  # Weighted average for ensemble (only if multiple_reward_models=True)
        # Optional regularization
        "min_entropy_reg": False,  # Add entropy regularization to RL loss
        "min_entropy_w": 0.1,  # Entropy regularization weight
    },
      
      
    "calip": {
        "tpt": False,
        "prompt_type": "no_prompt",
        "supported_list": ["no_prompt"],
        "calip_beta2": 1.0,
        "calip_beta3": 0.01,
    },
}