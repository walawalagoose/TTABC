# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "tpt": {
        "optimizer": "SGD",  # Adam for officehome
        "batch_size": 32,
        "ctx_init": 'a_photo_of_a',  # confidence threshold for online shot.
    },

}
