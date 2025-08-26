# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "tpt": {
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 64,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
        "selection_p": 0.1,  # confidence threshold for online shot.
        "tta_steps": 1,  # number of test-time adaptation steps
        "lr": 5e-3,  # learning rate
        "n_ctx": 4,  # number of prompt tokens
    },
    "ctpt": {
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
