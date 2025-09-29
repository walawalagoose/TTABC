import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torchvision.transforms import Resize

from clip.custom_clip import get_coop, get_zero_shot
from clip.cocoop import get_cocoop


class PadPrompter(nn.Module): # Default: 69840 Params
    def __init__(self, image_size=224, prompt_size=30):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size
        
        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class ResizedPadPrompter(nn.Module):
    def __init__(self, image_size=224, prompt_size=30):
        super().__init__()
        self.image_size = image_size
        self.prompt_size = prompt_size
        
        self.inner_image_size = image_size - 2 * prompt_size
        self.perturbation = nn.Parameter(torch.zeros(1, 3, image_size, image_size))
        
        mask = torch.ones(1, 3, image_size, image_size)
        mask[:, :, prompt_size : image_size - prompt_size, prompt_size : image_size - prompt_size] = 0
        self.register_buffer('mask', mask) 

    def forward(self, images):
        # make sure the image has already been shirnked
        if images.size(2) != self.inner_image_size or images.size(3) != self.inner_image_size:
            raise ValueError(f"The image size shold be {self.inner_image_size}x{self.inner_image_size}, "
                             f"but got {images.size(2)}x{images.size(3)}")

        padded_images = F.pad(images,
            (self.prompt_size, self.prompt_size, self.prompt_size, self.prompt_size),
            "constant",0
        )
        # self.perturbation * self.mask, make sure only the border area is updated
        prompted_images = padded_images + (self.perturbation * self.mask)
        return prompted_images
    

class FixedPatchPrompter(nn.Module): # Default: 30000 Params
    def __init__(self, image_size=224, prompt_size=100):
        super(FixedPatchPrompter, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize])
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module): # Default: 187500 Params
    def __init__(self, image_size=224, prompt_size=250):
        super(RandomPatchPrompter, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize])
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt
    

class LoR_VP(nn.Module):
    def __init__(self, rank, image_size=224, init_methods=['zero','random'], normalize=None):
        super(LoR_VP, self).__init__()
        self.normalize=normalize
        self.left_bar = torch.nn.Parameter(torch.empty(3, image_size, rank)) # B in ori paper
        self.get_init(init_methods[0], self.left_bar)
        self.right_bar = torch.nn.Parameter(torch.empty(3, rank, image_size)) # A in ori paper
        self.get_init(init_methods[1], self.right_bar)
        self.program = torch.bmm(self.left_bar, self.right_bar)

    def get_init(self, init_method, params):
        if init_method == 'zero':
            params.data.fill_(0)
        elif init_method == 'random':
            params.data.normal_(0, 1)
        elif init_method == 'xavier':
            torch.nn.init.xavier_uniform_(params)
        elif init_method == 'kaiming':
            torch.nn.init.kaiming_uniform_(params, nonlinearity='relu')
        elif init_method == 'uniform':
            torch.nn.init.uniform_(params, a=-0.1, b=0.1)
        elif init_method == 'normal':
            torch.nn.init.normal_(params, mean=0.0, std=0.01)

    def forward(self, x):
        self.program = torch.bmm(self.left_bar, self.right_bar)
        x = x + self.program
        if self.normalize is not None:
            x = self.normalize(x)
        return x
    
class VisualPrompter(nn.Module):
    # unified interface for different visual prompters
    def __init__(self, method, prompt_size=30, image_size=224, rank=4, device='cuda', **args):
        super(VisualPrompter, self).__init__()
        
        self.vp_type = method
        self.prompters_dict = {
            'pad_vp': PadPrompter,
            'resized_pad_vp': ResizedPadPrompter,
            'patch_vp': FixedPatchPrompter,
            'random_patch_vp': RandomPatchPrompter,
            'lor_vp': LoR_VP,
        }
        if method not in self.prompters_dict:
            raise ValueError(f"Unknown prompting method: {method}, available methods: {list(self.prompters_dict.keys())}")
        
        if method == 'lor_vp':
            self.prompter = self.prompters_dict[method](rank=rank, image_size=image_size).to(device)
        else:
            self.prompter = self.prompters_dict[method](image_size=image_size, prompt_size=prompt_size).to(device)
    
    def forward(self, images):
        if self.vp_type == 'resized_pad_vp':
            images = Resize(self.prompter.inner_image_size)(images)
        return self.prompter(images)

    def get_trainable_parameters(self):
        return self.prompter.parameters()
    
    def reset(self):
        if self.vp_type == 'lor_vp':
            self.prompter.get_init('zero', self.prompter.left_bar)
            self.prompter.get_init('zero', self.prompter.right_bar)
        else:
            # reset to zero or random initialization, TODO
            for param in self.prompter.parameters():
                param.data.fill_(0)
                # nn.init.normal_(param, 0, 0.02)
    
class VPCLIP(nn.Module):
    def __init__(self, backbone, visual_prompter, backbone_type='zs'):
        super(VPCLIP, self).__init__()
        self.backbone = backbone
        self.visual_prompter = visual_prompter
        self.backbone_type = backbone_type
    
    def forward(self, images):
        prompted_images = self.visual_prompter(images)
        logits, _ = self.backbone(prompted_images)
        return logits
    
    def reset_backbone(self):
        if self.backbone_type != 'zs' and hasattr(self.backbone, 'reset'):
            with torch.no_grad():
                self.backbone.reset()
            
    def reset_vp(self):
        self.visual_prompter.reset()
        
    def reset(self):
        # self.reset_backbone()
        self.reset_vp()
    
    def reset_classnames(self, classnames, arch):
        if hasattr(self.backbone, 'reset_classnames'):
            self.backbone.reset_classnames(classnames, arch)
    
    def get_text_features(self):
        if hasattr(self.backbone, 'get_text_features'):
            return self.backbone.get_text_features()
    
    # Not implemented yet, TODO
    # def gen_ctx(self, images, aug=False):
    #     if self.backbone_type == 'cocoop':
    #         # TODO, 后续看一下是否需要在生成上下文时使用vp（按照经验来说一般不需要更好）
    #         # if self.visual_prompter is not None:
    #         #     prompted_images = self.visual_prompter(images)
    #         # else:
    #         prompted_images = images
    #         return self.backbone.gen_ctx(prompted_images, aug)
    #     else:
    #         raise NotImplementedError(f"Backbone {self.backbone_type} doesn't support gen_ctx")
    
    def get_vp_parameters(self):
        if self.visual_prompter is not None:
            return list(self.visual_prompter.get_trainable_parameters())
        return []
    
    # Not implemented yet, TODO
    # def get_text_parameters(self):
    #     if self.backbone_type == 'coop':
    #         return list(self.backbone.prompt_learner.parameters())
    #     # elif self.backbone_type == 'cocoop':
    #     #     return list(self.backbone.prompt_generator.parameters())
    #     else:
    #         return []
    
    def get_trainable_parameters(self):
        params = []
        params.extend(self.get_vp_parameters())
        # params.extend(self.get_text_parameters())
        return params
    
def get_visual_prompt_clip(clip_arch, test_set, device, vp_type, backbone_type='zs', vp_args=None, coop_args=None):
    # Handle None arguments by setting default empty dict
    if vp_args is None:
        vp_args = {}
    if coop_args is None:
        coop_args = {}
        
    # Set hyperparameters for vp
    prompt_size = vp_args.get('prompt_size', 30)
    image_size = vp_args.get('image_size', 224)
    rank = vp_args.get('rank', 16)
    
    # Set hyperparameters for coop/cocoop, TODO
    n_ctx = coop_args.get('n_ctx', 16)
    ctx_init = coop_args.get('ctx_init', None)
    learned_cls = coop_args.get('learned_cls', False)
    
    # Initialize vp
    visual_prompter = VisualPrompter(vp_type, prompt_size=prompt_size, image_size=image_size, rank=rank, device=device)
        
    # get backbone model
    if backbone_type == 'zs':
        backbone = get_zero_shot(clip_arch, test_set, device)
    # # Not implemented yet, TODO
    # elif backbone_type == 'coop':
    #     backbone = get_coop(
    #         clip_arch,test_set,device,
    #         n_ctx=n_ctx, ctx_init=ctx_init,
    #         learned_cls=learned_cls
    #     )
    # elif backbone_type == 'cocoop':
    #     backbone = get_cocoop(clip_arch, test_set, device, n_ctx=n_ctx)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}, supported: ['zs', 'coop', 'cocoop']")

    return VPCLIP(backbone, visual_prompter, backbone_type)