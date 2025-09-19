# -*- coding: utf-8 -*-
from .zero_shot import ZEROSHOT
from .baseline import BASELINE
from .tpt import TPT
from .ctpt import CTPT
from .otpt import OTPT
from .ntpt import NTPT


def get_method(method_name):
    method_dict = {
        'baseline': BASELINE,
        'zero_shot': ZEROSHOT,
        'tpt': TPT,
        'ctpt': CTPT,
        'otpt': OTPT,
        'ntpt': NTPT,
    }
    
    if method_name not in method_dict:
        raise ValueError(f"Unknown method: {method_name}. Available methods: {list(method_dict.keys())}")
    
    return method_dict[method_name]
