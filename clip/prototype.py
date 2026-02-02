import torch
import torch.nn as nn
import torch.nn.functional as F

class Shifter(nn.Module):
    def __init__(self, 
            embed_dim=512, 
            num_classes=1000,
            per_label=False,
            device='cuda',
            dtype=None, 
            do_shift=True, 
            do_scale=False,
            text_embeds=None,
            num_components=-1,
            class2concepts=None,
            
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.per_label = per_label
        self.num_classes = num_classes
        self.device = device
        
        self.dtype = dtype
        self.do_shift = do_shift
        self.do_scale = do_scale

        self.num_classes_per_concept = [len(v) for _,v in class2concepts.items()] if class2concepts is not None else None

        self.text_embeds = text_embeds

        if num_components == -1:
            num_components = num_classes
        self.num_components = num_components

        if per_label:
            shift_init = torch.zeros((num_classes, embed_dim), dtype=self.dtype, device=self.device)
            scale_init = torch.ones((num_classes, embed_dim), dtype=self.dtype, device=self.device)
        else:
            shift_init = torch.zeros(embed_dim, dtype=self.dtype, device=self.device)
            scale_init = torch.ones(embed_dim, dtype=self.dtype, device=self.device)

        if self.do_shift:
            self.shift_init_state_original = shift_init.detach().clone()
            self.shift_init_state = shift_init.detach().clone()
            self.shift = nn.Parameter(shift_init)

        if self.do_scale:
            self.scale_init_state_original = scale_init.detach().clone()
            self.scale_init_state = scale_init.detach().clone()
            self.scale = nn.Parameter(scale_init)


    def reset(self):
        if self.do_shift:
            self.shift.copy_(self.shift_init_state)
        
        if self.do_scale:
            self.scale.copy_(self.scale_init_state)

    def forward(self, img_embed, test=False):
        x = img_embed

        if self.do_scale:
            x = self.scale * x

        if self.do_shift:
            if test or self.num_classes_per_concept is None:
                x += self.shift
            else: # need to expand per concept
                shift = [self.shift[i].repeat(n, 1) for i,n in enumerate(self.num_classes_per_concept)]
                shift = torch.cat(shift, dim=0)

                x += shift
                
        return x


class PrototypeShifterWrapper(nn.Module):
    def __init__(
            self,
            clip_model,
            prototypes,
            per_label=False,
            device=None,
            do_shift=True,
            do_scale=False,
        ):
        super().__init__()

        if device is None:
            device = clip_model.visual.conv1.weight.device

        self.clip_model = clip_model
        self.device = device
        self.logit_scale = clip_model.logit_scale.data
        prototypes = prototypes.to(device)
        self.register_buffer("prototypes", prototypes)

        embed_dim = prototypes.shape[-1]
        num_classes = prototypes.shape[0]

        self.shifter = Shifter(
            embed_dim=embed_dim,
            num_classes=num_classes,
            per_label=per_label,
            device=device,
            dtype=self.dtype,
            do_shift=do_shift,
            do_scale=do_scale,
            text_embeds=prototypes,
        )
        self.shifter.to(device)

    @property
    def dtype(self):
        return self.clip_model.dtype

    def reset(self):
        self.shifter.reset()

    def forward(self, image, test=False, return_features=False):
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)

        base_text = F.normalize(self.prototypes, dim=-1)
        text_features = self.shifter(base_text, test=test)
        # text_features = self.shifter(self.prototypes, test=test)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.t())

        if return_features:
            return logits, image_features, text_features
        return logits


def wrap_prototype_shifter(backbone, prototypes, **kwargs):
    clip_model = backbone.clip_model if hasattr(backbone, "clip_model") else backbone
    return PrototypeShifterWrapper(clip_model=clip_model, prototypes=prototypes, **kwargs)
        