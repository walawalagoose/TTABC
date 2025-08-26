# -*- coding: utf-8 -*-
from .baseline import BASELINE
from .tpt import TPT
from .ctpt import CTPT
from .otpt import OTPT
from .ntpt import NTPT


def get_method(method_name):
    return {
        "baseline": BASELINE,
        "tpt": TPT,
        "ctpt": CTPT,
        "otpt": OTPT,
        "ntpt": NTPT,
    }[method_name]
