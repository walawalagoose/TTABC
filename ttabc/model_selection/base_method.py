import torch
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from ttabc.utils.load_model import create_model, set_optimizer
from data.cls_to_names import *

class BaseMethod:
    def __init__(self, args):
        self.args = args
        self.model, self.model_state = create_model(args)
        # define optimizer
        self.optimizer, self.optim_state = set_optimizer(args, self.model)
        # setup automatic mixed-precision (Amp) loss scaling
        self.scaler = torch.amp.GradScaler(init_scale=1000, device='cuda')
        print('=> Using native Torch AMP. Training in mixed precision.')

    def test_time_tuning(self, inputs):
        pass

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        pass
