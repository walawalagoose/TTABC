# -*- coding: utf-8 -*-
from .zero_shot import ZEROSHOT
from .baseline import BASELINE
from .tpt import TPT
from .ctpt import CTPT
from .otpt import OTPT
from .tda import TDA
from .boostadapter import BoostAdapter
from .difftpt import DiffTPT
from .zero import ZERO
from .batclip import BATCLIP
from .dmn import DMN
from .histpt import HisTPT
from .oga import OGA
from .bca import BCA
from .dota import DOTA
from .tent import Tent
from .sar import SAR
from .deyo import DeYO
from .tps import TPS
from .dpe import DPE_CLIP
from .rlcf import RLCF
from .calip import CALIP


def get_method(method_name):
    method_dict = {
        'baseline': BASELINE,
        'zero_shot': ZEROSHOT,
        'tpt': TPT,
        'ctpt': CTPT,
        'otpt': OTPT,
        'tda': TDA,
        'boostadapter': BoostAdapter,
        'difftpt': DiffTPT,
        'zero': ZERO,
    'batclip': BATCLIP,
    'dmn': DMN,
        'histpt': HisTPT,
        'oga': OGA,
        'bca': BCA,
        'dota': DOTA,
        'tent': Tent,
        'sar': SAR,
        'deyo': DeYO,
        'tps': TPS,
        'dpe': DPE_CLIP,
        'rlcf': RLCF,
        'calip': CALIP,
    }
    
    if method_name not in method_dict:
        raise ValueError(f"Unknown method: {method_name}. Available methods: {list(method_dict.keys())}")
    
    return method_dict[method_name]
